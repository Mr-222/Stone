#include "OpaqueDirectLightingPass.h"

#include <array>

#include "Core/Buffer.h"
#include "Core/MetalContext.h"
#include "Core/RenderGraph.h"
#include "Core/Texture.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kOpaqueDirectLightingShaderLibrary = STONE_SHADER_DIR "/OpaqueDirectLighting.metallib";

constexpr const char* kGBufferAlbedoMetallicImageName = "gbuffer_albedo_metallic";
constexpr const char* kGBufferNormalImageName = "gbuffer_normal";
constexpr const char* kGBufferDepthRoughnessImageName = "gbuffer_depth_roughness";
constexpr MTL::PixelFormat kGBufferAlbedoMetallicFormat = MTL::PixelFormatRGBA8Unorm_sRGB; // rgb = base color, a = metallic
constexpr MTL::PixelFormat kGBufferNormalFormat = MTL::PixelFormatRGBA16Snorm; // xy = octahedral shading normal, zw = octahedral geometric normal
constexpr MTL::PixelFormat kGBufferDepthRoughnessFormat = MTL::PixelFormatRG32Float; // x = linear view Z, y = perceptual roughness

constexpr NS::UInteger AlignThreadgroupMemory(NS::UInteger size) { return (size + 15) & ~NS::UInteger(15); }
constexpr NS::UInteger kTileLightListThreadgroupBytes = AlignThreadgroupMemory(sizeof(TileLightList));
constexpr NS::UInteger kTileCullingScratchThreadgroupBytes = AlignThreadgroupMemory(sizeof(TileCullingScratch));

// All pipelines in the deferred pass share the scene color + G-buffer attachment layout.
template <typename ColorAttachmentArray>
void SetDeferredColorAttachmentFormats(ColorAttachmentArray* attachments, MTL::PixelFormat sceneColorFormat) {
    attachments->object(0)->setPixelFormat(sceneColorFormat);
    attachments->object(1)->setPixelFormat(kGBufferAlbedoMetallicFormat);
    attachments->object(2)->setPixelFormat(kGBufferNormalFormat);
    attachments->object(3)->setPixelFormat(kGBufferDepthRoughnessFormat);
}

struct OpaqueDirectLightingVertexArgumentData {
    MTL::GPUAddress vertices;
    MTL::GPUAddress renderPrimitives;
};

struct OpaqueDirectLightingFragmentArgumentData {
    MTL::GPUAddress materials;
    MTL::GPUAddress lightListInfo;
    MTL::GPUAddress directionalLights;
    MTL::GPUAddress pointLights;
    MTL::GPUAddress spotLights;
    MTL::ResourceID accelerationStructure;
    MTL::GPUAddress opaqueVertices;
    MTL::GPUAddress opaqueIndices;
    MTL::GPUAddress opaqueSubmeshes;
    std::array<MTL::ResourceID, kMaxBindlessTextureCount> textures;
};

struct OpaqueDirectLightingPassData {
    RenderGraphColorAttachment colorAttachment;
    RenderGraphColorAttachment gbufferAlbedoMetallicAttachment;
    RenderGraphColorAttachment gbufferNormalAttachment;
    RenderGraphColorAttachment gbufferDepthRoughnessAttachment;
    RenderGraphDepthAttachment depthAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::RenderPipelineState* maskedPipelineState = nullptr;
    MTL::RenderPipelineState* tileCullingPipelineState = nullptr;
    MTL::RenderPipelineState* deferredLightingPipelineState = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL::DepthStencilState* deferredLightingDepthStencilState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Buffer* vertexArgumentBuffer = nullptr;
    MTL::Buffer* fragmentArgumentBuffer = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
    RenderGraphResourceHandle opaqueVertexBufferHandle;
    RenderGraphResourceHandle opaqueIndexBufferHandle;
    RenderGraphResourceHandle opaqueSubmeshBufferHandle;
    RenderGraphResourceHandle opaqueRenderPrimitiveBufferHandle;
    RenderGraphResourceHandle materialBufferHandle;
    RenderGraphResourceHandle lightListInfoBufferHandle;
    RenderGraphResourceHandle directionalLightBufferHandle;
    RenderGraphResourceHandle pointLightBufferHandle;
    RenderGraphResourceHandle spotLightBufferHandle;
    RenderGraphResourceHandle transparentTileLightListBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
    std::vector<MTL::Texture*> textures;
    MTL::ResidencySet* accelResidencySet = nullptr;
    int numPrimitives;
    int maskedPrimitiveStart;
};

OpaqueDirectLightingPass::~OpaqueDirectLightingPass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_deferredLightingDepthStencilState)
        m_deferredLightingDepthStencilState->release();
    if (m_depthStencilState)
        m_depthStencilState->release();
    if (m_deferredLightingPipelineState)
        m_deferredLightingPipelineState->release();
    if (m_tileCullingPipelineState)
        m_tileCullingPipelineState->release();
    if (m_maskedPipelineState)
        m_maskedPipelineState->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void OpaqueDirectLightingPass::Setup(
    MetalContext& context,
    const int numPrimitives,
    const int maskedPrimitiveStart,
    const std::vector<Texture>& textures,
    MTL::AccelerationStructure* tlas,
    MTL::ResidencySet* accelResidencySet)
{
    m_numPrimitives = numPrimitives;
    m_maskedPrimitiveStart = maskedPrimitiveStart;
    m_tlas = tlas;
    m_accelResidencySet = accelResidencySet;
    LOG_ERROR_IF(!m_tlas, "OpaqueDirectLighting requires a valid acceleration structure.");
    LOG_ERROR_IF(!m_accelResidencySet, "OpaqueDirectLighting requires a valid acceleration structure residency set.");
    m_textures.reserve(textures.size());
    for (size_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        MTL::Texture* texture = textures[textureIndex].GetNative();
        LOG_ERROR_IF(!texture, "OpaqueDirectLighting texture {} is null", textureIndex);
        LOG_ERROR_IF(texture->textureType() != MTL::TextureType2D,
            "OpaqueDirectLighting texture {} has Metal texture type {}, expected MTLTextureType2D ({})",
            textureIndex,
            static_cast<NS::UInteger>(texture->textureType()),
            static_cast<NS::UInteger>(MTL::TextureType2D));
        m_textures.push_back(texture);
    }

    LOG_ERROR_IF(m_textures.empty(), "OpaqueDirectLighting requires at least the fallback texture");

    LOG_ERROR_IF(m_textures.size() > kMaxBindlessTextureCount,
        "OpaqueDirectLighting needs {} textures, but the bindless table supports {}.",
        m_textures.size(),
        kMaxBindlessTextureCount);

    MTL::Device* device = context.GetDevice();

    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kOpaqueDirectLightingShaderLibrary, {
        "gbuffer_vertex", "gbuffer_fragment", "gbufferMasked_fragment",
        "tileLightCulling_kernel", "deferredLighting_vertex", "deferredLighting_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("OpaqueGBuffer", NS::UTF8StringEncoding));

    MTL4::LibraryFunctionDescriptor* vertexFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "gbuffer_vertex");
    MTL4::LibraryFunctionDescriptor* fragmentFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "gbuffer_fragment");
    MTL4::LibraryFunctionDescriptor* maskedFragmentFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "gbufferMasked_fragment");

    pipelineDescriptor->setVertexFunctionDescriptor(vertexFunc);
    pipelineDescriptor->setFragmentFunctionDescriptor(fragmentFunc);
    SetDeferredColorAttachmentFormats(pipelineDescriptor->colorAttachments(), context.GetSwapchainPixelFormat());
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
    pipelineDescriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create G-buffer render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    pipelineDescriptor->setLabel(NS::String::string("OpaqueGBufferMasked", NS::UTF8StringEncoding));
    pipelineDescriptor->setFragmentFunctionDescriptor(maskedFragmentFunc);
    m_maskedPipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_maskedPipelineState, "Failed to create masked G-buffer render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL4::TileRenderPipelineDescriptor* tileDesc = MTL4::TileRenderPipelineDescriptor::alloc()->init()->autorelease();
    tileDesc->setLabel(NS::String::string("OpaqueDeferredTileLightCulling", NS::UTF8StringEncoding));
    tileDesc->setTileFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "tileLightCulling_kernel"));
    tileDesc->setThreadgroupSizeMatchesTileSize(true);
    SetDeferredColorAttachmentFormats(tileDesc->colorAttachments(), context.GetSwapchainPixelFormat());
    m_tileCullingPipelineState = compiler->newRenderPipelineState(tileDesc, taskOptions, &error);
    LOG_ERROR_IF(!m_tileCullingPipelineState, "Failed to create tile light culling pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL4::RenderPipelineDescriptor* deferredDesc = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    deferredDesc->setLabel(NS::String::string("OpaqueDeferredLighting", NS::UTF8StringEncoding));
    deferredDesc->setVertexFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "deferredLighting_vertex"));
    deferredDesc->setFragmentFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "deferredLighting_fragment"));
    SetDeferredColorAttachmentFormats(deferredDesc->colorAttachments(), context.GetSwapchainPixelFormat());
    deferredDesc->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
    m_deferredLightingPipelineState = compiler->newRenderPipelineState(deferredDesc, taskOptions, &error);
    LOG_ERROR_IF(!m_deferredLightingPipelineState, "Failed to create deferred lighting render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    depthStencilDescriptor->setLabel(NS::String::string("OpaqueDirectLighting Depth State", NS::UTF8StringEncoding));
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthStencilDescriptor->setDepthWriteEnabled(true);
    m_depthStencilState = device->newDepthStencilState(depthStencilDescriptor);
    LOG_ERROR_IF(!m_depthStencilState, "Failed to create opaque direct lighting depth-stencil state.");

    MTL::DepthStencilDescriptor* deferredDepthDesc = MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    deferredDepthDesc->setLabel(NS::String::string("OpaqueDeferredLighting Depth State", NS::UTF8StringEncoding));
    deferredDepthDesc->setDepthCompareFunction(MTL::CompareFunctionAlways);
    deferredDepthDesc->setDepthWriteEnabled(false);
    m_deferredLightingDepthStencilState = device->newDepthStencilState(deferredDepthDesc);
    LOG_ERROR_IF(!m_deferredLightingDepthStencilState, "Failed to create deferred lighting depth-stencil state.");

    const OpaqueDirectLightingVertexArgumentData vertexArguments{};
    m_vertexArgumentBuffer = std::make_unique<Buffer>(
        device,
        &vertexArguments,
        sizeof(vertexArguments),
        MTL::ResourceStorageModeShared);
    m_vertexArgumentBuffer->GetNative()->setLabel(NS::String::string("OpaqueDirectLighting Vertex Argument Buffer", NS::UTF8StringEncoding));

    const OpaqueDirectLightingFragmentArgumentData fragmentArguments{};
    m_fragmentArgumentBuffer = std::make_unique<Buffer>(
        device,
        &fragmentArguments,
        sizeof(fragmentArguments),
        MTL::ResourceStorageModeShared);
    m_fragmentArgumentBuffer->GetNative()->setLabel(
        NS::String::string("OpaqueDirectLighting Fragment Argument Buffer", NS::UTF8StringEncoding));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("OpaqueDirectLighting Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(
        m_vertexArgumentBuffer->GetGPUAddress(),
        static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::VertexArguments));
    m_argumentTable->setAddress(
        m_fragmentArgumentBuffer->GetGPUAddress(),
        static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::FragmentArguments));

    const uint32_t frameSlotCount = context.GetFrameSlotCount();
    const uint32_t width = context.GetDrawableWidth();
    const uint32_t height = context.GetDrawableHeight();
    const size_t tileCount = static_cast<size_t>((width + kLightTileSize - 1) / kLightTileSize) * ((height + kLightTileSize - 1) / kLightTileSize);
    const auto createGBufferTexture = [&](MTL::PixelFormat format, const char* label) {
        MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::texture2DDescriptor(format, width, height, false);
        descriptor->setStorageMode(MTL::StorageModeMemoryless);
        descriptor->setUsage(MTL::TextureUsageRenderTarget);
        auto texture = std::make_unique<Texture>(device, descriptor);
        texture->GetNative()->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
        return texture;
    };

    m_gbufferAlbedoMetallicTextures.resize(frameSlotCount);
    m_gbufferNormalTextures.resize(frameSlotCount);
    m_gbufferDepthRoughnessTextures.resize(frameSlotCount);
    m_transparentTileLightListBuffers.resize(frameSlotCount);
    for (uint32_t frameSlot = 0; frameSlot < frameSlotCount; ++frameSlot) {
        m_gbufferAlbedoMetallicTextures[frameSlot] = createGBufferTexture(kGBufferAlbedoMetallicFormat, "GBuffer Albedo Metallic");
        m_gbufferNormalTextures[frameSlot] = createGBufferTexture(kGBufferNormalFormat, "GBuffer Normal");
        m_gbufferDepthRoughnessTextures[frameSlot] = createGBufferTexture(kGBufferDepthRoughnessFormat, "GBuffer Depth Roughness");
        m_transparentTileLightListBuffers[frameSlot] = std::make_unique<Buffer>(device, tileCount * sizeof(TileLightList), MTL::ResourceStorageModePrivate);
        m_transparentTileLightListBuffers[frameSlot]->GetNative()->setLabel(NS::String::string("Transparent Tile Light List Buffer", NS::UTF8StringEncoding));
    }

    compiler->release();
}

void OpaqueDirectLightingPass::AddToGraph(RenderGraph& graph) {
    for (uint32_t frameSlot = 0; frameSlot < m_transparentTileLightListBuffers.size(); ++frameSlot) {
        graph.RegisterFrameLocalTexture(kGBufferAlbedoMetallicImageName, frameSlot, *m_gbufferAlbedoMetallicTextures[frameSlot]);
        graph.RegisterFrameLocalTexture(kGBufferNormalImageName, frameSlot, *m_gbufferNormalTextures[frameSlot]);
        graph.RegisterFrameLocalTexture(kGBufferDepthRoughnessImageName, frameSlot, *m_gbufferDepthRoughnessTextures[frameSlot]);
        graph.RegisterFrameLocalBuffer(kTransparentTileLightListBufferName, frameSlot, *m_transparentTileLightListBuffers[frameSlot]);
    }

    RenderGraphResourceHandle sceneColorHandle = graph.DeclareTexture(kSceneColorImageName);
    RenderGraphResourceHandle gbufferAlbedoMetallicHandle = graph.DeclareTexture(kGBufferAlbedoMetallicImageName);
    RenderGraphResourceHandle gbufferNormalHandle = graph.DeclareTexture(kGBufferNormalImageName);
    RenderGraphResourceHandle gbufferDepthRoughnessHandle = graph.DeclareTexture(kGBufferDepthRoughnessImageName);
    RenderGraphResourceHandle depthHandle = graph.DeclareTexture(kSceneDepthImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");
    RenderGraphResourceHandle transparentTileLightListBufferHandle = graph.DeclareBuffer(kTransparentTileLightListBufferName);
    RenderGraphResourceHandle opaqueVertexBufferHandle = graph.DeclareBuffer("OpaqueVertexBuffer");
    RenderGraphResourceHandle opaqueIndexBufferHandle = graph.DeclareBuffer("OpaqueIndexBuffer");
    RenderGraphResourceHandle opaqueSubmeshBufferHandle = graph.DeclareBuffer("OpaqueSubmeshBuffer");
    RenderGraphResourceHandle opaqueRenderPrimitiveBufferHandle = graph.DeclareBuffer("OpaqueRenderPrimitiveBuffer");
    RenderGraphResourceHandle materialBufferHandle = graph.DeclareBuffer("MaterialBuffer");
    RenderGraphResourceHandle lightListInfoBufferHandle = graph.DeclareBuffer("LightListInfoBuffer");
    RenderGraphResourceHandle directionalLightBufferHandle = graph.DeclareBuffer("DirectionalLightBuffer");
    RenderGraphResourceHandle pointLightBufferHandle = graph.DeclareBuffer("PointLightBuffer");
    RenderGraphResourceHandle spotLightBufferHandle = graph.DeclareBuffer("SpotLightBuffer");
    RenderGraphResourceHandle indirectCBHandle = graph.DeclareIndirectCommandBuffer("IndirectCommandBuffer");

    graph.AddPass<OpaqueDirectLightingPassData>(
        "OpaqueDirectLighting",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, OpaqueDirectLightingPassData& data, RenderGraphResources& resources) {
            data.colorAttachment = builder.WriteColor(sceneColorHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0),
            });
            data.gbufferAlbedoMetallicAttachment = builder.WriteColor(gbufferAlbedoMetallicHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionDontCare,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0),
            });
            data.gbufferNormalAttachment = builder.WriteColor(gbufferNormalHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionDontCare,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0),
            });
            data.gbufferDepthRoughnessAttachment = builder.WriteColor(gbufferDepthRoughnessHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionDontCare,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0),
            });
            data.depthAttachment = builder.WriteDepth(depthHandle, RenderGraphDepthAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearDepth = 1.0,
            });
            data.pipelineState = m_pipelineState;
            data.maskedPipelineState = m_maskedPipelineState;
            data.tileCullingPipelineState = m_tileCullingPipelineState;
            data.deferredLightingPipelineState = m_deferredLightingPipelineState;
            data.depthStencilState = m_depthStencilState;
            data.deferredLightingDepthStencilState = m_deferredLightingDepthStencilState;
            data.argumentTable = m_argumentTable;
            data.vertexArgumentBuffer = m_vertexArgumentBuffer->GetNative();
            data.fragmentArgumentBuffer = m_fragmentArgumentBuffer->GetNative();
            data.frameUniformHandle = frameUniformHandle;
            data.opaqueVertexBufferHandle = opaqueVertexBufferHandle;
            data.opaqueIndexBufferHandle = opaqueIndexBufferHandle;
            data.opaqueSubmeshBufferHandle = opaqueSubmeshBufferHandle;
            data.opaqueRenderPrimitiveBufferHandle = opaqueRenderPrimitiveBufferHandle;
            data.materialBufferHandle = materialBufferHandle;
            data.lightListInfoBufferHandle = lightListInfoBufferHandle;
            data.directionalLightBufferHandle = directionalLightBufferHandle;
            data.pointLightBufferHandle = pointLightBufferHandle;
            data.spotLightBufferHandle = spotLightBufferHandle;
            data.transparentTileLightListBufferHandle = transparentTileLightListBufferHandle;
            data.indirectCBHandle = indirectCBHandle;
            data.textures = m_textures;
            data.accelResidencySet = m_accelResidencySet;
            data.numPrimitives = m_numPrimitives;
            data.maskedPrimitiveStart = m_maskedPrimitiveStart;

            builder.ReadBuffer(frameUniformHandle);
            builder.ReadBuffer(opaqueVertexBufferHandle);
            builder.ReadBuffer(opaqueIndexBufferHandle);
            builder.ReadBuffer(opaqueSubmeshBufferHandle);
            builder.ReadBuffer(opaqueRenderPrimitiveBufferHandle);
            builder.ReadBuffer(materialBufferHandle);
            builder.ReadBuffer(lightListInfoBufferHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
            builder.ReadBuffer(pointLightBufferHandle);
            builder.ReadBuffer(spotLightBufferHandle);
            builder.WriteBuffer(transparentTileLightListBufferHandle);
            builder.ReadIndirectCommandBuffer(indirectCBHandle);

            MTL::Buffer* opaqueVertexBuffer = resources.GetBuffer(opaqueVertexBufferHandle);
            LOG_ERROR_IF(!opaqueVertexBuffer, "OpaqueDirectLighting: Failed to get opaque vertex buffer");

            MTL::Buffer* opaqueRenderPrimitiveBuffer = resources.GetBuffer(opaqueRenderPrimitiveBufferHandle);
            LOG_ERROR_IF(!opaqueRenderPrimitiveBuffer, "OpaqueDirectLighting: Failed to get opaque render primitive buffer");

            const OpaqueDirectLightingVertexArgumentData vertexArguments {
                .vertices = opaqueVertexBuffer->gpuAddress(),
                .renderPrimitives = opaqueRenderPrimitiveBuffer->gpuAddress(),
            };
            m_vertexArgumentBuffer->Update(&vertexArguments, sizeof(vertexArguments));

            MTL::Buffer* opaqueIndexBuffer = resources.GetBuffer(opaqueIndexBufferHandle);
            LOG_ERROR_IF(!opaqueIndexBuffer, "OpaqueDirectLighting: Failed to get opaque index buffer");
            MTL::Buffer* opaqueSubmeshBuffer = resources.GetBuffer(opaqueSubmeshBufferHandle);
            LOG_ERROR_IF(!opaqueSubmeshBuffer, "OpaqueDirectLighting: Failed to get opaque submesh buffer");
            MTL::Buffer* materialBuffer = resources.GetBuffer(materialBufferHandle);
            LOG_ERROR_IF(!materialBuffer, "OpaqueDirectLighting: Failed to get material buffer");
            MTL::Buffer* lightListInfoBuffer = resources.GetBuffer(lightListInfoBufferHandle);
            LOG_ERROR_IF(!lightListInfoBuffer, "OpaqueDirectLighting: Failed to get light list info buffer");
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(directionalLightBufferHandle);
            LOG_ERROR_IF(!directionalLightBuffer, "OpaqueDirectLighting: Failed to get directional light buffer");
            MTL::Buffer* pointLightBuffer = resources.GetBuffer(pointLightBufferHandle);
            LOG_ERROR_IF(!pointLightBuffer, "OpaqueDirectLighting: Failed to get point light buffer");
            MTL::Buffer* spotLightBuffer = resources.GetBuffer(spotLightBufferHandle);
            LOG_ERROR_IF(!spotLightBuffer, "OpaqueDirectLighting: Failed to get spot light buffer");

            OpaqueDirectLightingFragmentArgumentData fragmentArguments {
                .materials = materialBuffer->gpuAddress(),
                .lightListInfo = lightListInfoBuffer->gpuAddress(),
                .directionalLights = directionalLightBuffer->gpuAddress(),
                .pointLights = pointLightBuffer->gpuAddress(),
                .spotLights = spotLightBuffer->gpuAddress(),
                .accelerationStructure = m_tlas->gpuResourceID(),
                .opaqueVertices = opaqueVertexBuffer->gpuAddress(),
                .opaqueIndices = opaqueIndexBuffer->gpuAddress(),
                .opaqueSubmeshes = opaqueSubmeshBuffer->gpuAddress(),
            };
            for (size_t textureIndex = 0; textureIndex < fragmentArguments.textures.size(); ++textureIndex) {
                MTL::Texture* texture = textureIndex < m_textures.size() ? m_textures[textureIndex] : m_textures.front();
                fragmentArguments.textures[textureIndex] = texture->gpuResourceID();
            }
            m_fragmentArgumentBuffer->Update(&fragmentArguments, sizeof(fragmentArguments));
        },
        [](const OpaqueDirectLightingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            MTL::Buffer* opaqueVertexBuffer = resources.GetBuffer(data.opaqueVertexBufferHandle);
            MTL::Buffer* opaqueIndexBuffer = resources.GetBuffer(data.opaqueIndexBufferHandle);
            MTL::Buffer* opaqueSubmeshBuffer = resources.GetBuffer(data.opaqueSubmeshBufferHandle);
            MTL::Buffer* opaqueRenderPrimitiveBuffer = resources.GetBuffer(data.opaqueRenderPrimitiveBufferHandle);
            MTL::Buffer* materialBuffer = resources.GetBuffer(data.materialBufferHandle);
            MTL::Buffer* lightListInfoBuffer = resources.GetBuffer(data.lightListInfoBufferHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(data.directionalLightBufferHandle);
            MTL::Buffer* pointLightBuffer = resources.GetBuffer(data.pointLightBufferHandle);
            MTL::Buffer* spotLightBuffer = resources.GetBuffer(data.spotLightBufferHandle);
            MTL::Buffer* transparentTileLightListBuffer = resources.GetBuffer(data.transparentTileLightListBufferHandle);
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);

            LOG_ERROR_IF(!frameUniformBuffer, "OpaqueDirectLighting: Failed to get frame uniform buffer");
            LOG_ERROR_IF(!opaqueVertexBuffer, "OpaqueDirectLighting: Failed to get opaque vertex buffer");
            LOG_ERROR_IF(!opaqueIndexBuffer, "OpaqueDirectLighting: Failed to get opaque index buffer");
            LOG_ERROR_IF(!opaqueSubmeshBuffer, "OpaqueDirectLighting: Failed to get opaque submesh buffer");
            LOG_ERROR_IF(!opaqueRenderPrimitiveBuffer, "OpaqueDirectLighting: Failed to get opaque render primitive buffer");
            LOG_ERROR_IF(!materialBuffer, "OpaqueDirectLighting: Failed to get material buffer");
            LOG_ERROR_IF(!lightListInfoBuffer, "OpaqueDirectLighting: Failed to get light list info buffer");
            LOG_ERROR_IF(!directionalLightBuffer, "OpaqueDirectLighting: Failed to get directional light buffer");
            LOG_ERROR_IF(!pointLightBuffer, "OpaqueDirectLighting: Failed to get point light buffer");
            LOG_ERROR_IF(!spotLightBuffer, "OpaqueDirectLighting: Failed to get spot light buffer");
            LOG_ERROR_IF(!transparentTileLightListBuffer, "OpaqueDirectLighting: Failed to get transparent tile light list buffer");
            LOG_ERROR_IF(!indirectCB, "OpaqueDirectLighting: Failed to get indirect command buffer");

            data.argumentTable->setAddress(frameUniformBuffer->gpuAddress(), static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::FrameUniform));
            data.argumentTable->setAddress(transparentTileLightListBuffer->gpuAddress(), static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::TransparentTileLightLists));

            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(opaqueVertexBuffer);
            cmd.AddResource(opaqueIndexBuffer);
            cmd.AddResource(opaqueSubmeshBuffer);
            cmd.AddResource(opaqueRenderPrimitiveBuffer);
            cmd.AddResource(materialBuffer);
            cmd.AddResource(lightListInfoBuffer);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(pointLightBuffer);
            cmd.AddResource(spotLightBuffer);
            cmd.AddResource(transparentTileLightListBuffer);
            cmd.AddResource(indirectCB);
            cmd.AddResource(data.vertexArgumentBuffer);
            cmd.AddResource(data.fragmentArgumentBuffer);
            for (MTL::Texture* texture : data.textures)
                cmd.AddResource(texture);
            cmd.AddResidencySet(data.accelResidencySet);

            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            MTL::Texture* gbufferAlbedoMetallicTexture = resources.GetTexture(data.gbufferAlbedoMetallicAttachment.texture);
            MTL::Texture* gbufferNormalTexture = resources.GetTexture(data.gbufferNormalAttachment.texture);
            MTL::Texture* gbufferDepthRoughnessTexture = resources.GetTexture(data.gbufferDepthRoughnessAttachment.texture);
            MTL::Texture* depthTexture = resources.GetTexture(data.depthAttachment.texture);

            LOG_ERROR_IF(!colorTexture, "OpaqueDirectLighting: No color target.");
            LOG_ERROR_IF(!gbufferAlbedoMetallicTexture, "OpaqueDirectLighting: No G-buffer albedo/metallic target.");
            LOG_ERROR_IF(!gbufferNormalTexture, "OpaqueDirectLighting: No G-buffer normal target.");
            LOG_ERROR_IF(!gbufferDepthRoughnessTexture, "OpaqueDirectLighting: No G-buffer depth/roughness target.");
            LOG_ERROR_IF(!depthTexture, "OpaqueDirectLighting: No depth target.");
            LOG_ERROR_IF(depthTexture->pixelFormat() != MTL::PixelFormatDepth32Float, "OpaqueDirectLighting: Depth target has pixel format {}, expected Depth32Float ({}).", static_cast<NS::UInteger>(depthTexture->pixelFormat()), static_cast<NS::UInteger>(MTL::PixelFormatDepth32Float));
            LOG_ERROR_IF(depthTexture->width() != colorTexture->width() || depthTexture->height() != colorTexture->height(), "OpaqueDirectLighting: Depth target size {}x{} does not match color target size {}x{}.", depthTexture->width(), depthTexture->height(), colorTexture->width(), colorTexture->height());
            LOG_ERROR_IF(!data.pipelineState, "OpaqueDirectLighting: Pipeline state is null.");
            LOG_ERROR_IF(!data.maskedPipelineState, "OpaqueDirectLighting: Masked pipeline state is null.");
            LOG_ERROR_IF(!data.tileCullingPipelineState, "OpaqueDirectLighting: Tile culling pipeline state is null.");
            LOG_ERROR_IF(!data.deferredLightingPipelineState, "OpaqueDirectLighting: Deferred lighting pipeline state is null.");
            LOG_ERROR_IF(!data.depthStencilState, "OpaqueDirectLighting: Depth-stencil state is null.");
            LOG_ERROR_IF(!data.deferredLightingDepthStencilState, "OpaqueDirectLighting: Deferred lighting depth-stencil state is null.");
            LOG_ERROR_IF(!data.argumentTable, "OpaqueDirectLighting: Argument table is null.");

            cmd.AddResource(colorTexture);
            cmd.AddResource(depthTexture);

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            const auto configureColorAttachment = [&](NS::UInteger index, MTL::Texture* texture, const RenderGraphColorAttachmentDesc& desc) {
                MTL::RenderPassColorAttachmentDescriptor* attachment = passDescriptor->colorAttachments()->object(index);
                attachment->setTexture(texture);
                attachment->setLoadAction(desc.loadAction);
                attachment->setClearColor(desc.clearColor);
                attachment->setStoreAction(desc.storeAction);
            };
            configureColorAttachment(0, colorTexture, data.colorAttachment.desc);
            configureColorAttachment(1, gbufferAlbedoMetallicTexture, data.gbufferAlbedoMetallicAttachment.desc);
            configureColorAttachment(2, gbufferNormalTexture, data.gbufferNormalAttachment.desc);
            configureColorAttachment(3, gbufferDepthRoughnessTexture, data.gbufferDepthRoughnessAttachment.desc);

            MTL::RenderPassDepthAttachmentDescriptor* depthAttachment = passDescriptor->depthAttachment();
            depthAttachment->setTexture(depthTexture);
            depthAttachment->setLoadAction(data.depthAttachment.desc.loadAction);
            depthAttachment->setStoreAction(data.depthAttachment.desc.storeAction);
            depthAttachment->setClearDepth(data.depthAttachment.desc.clearDepth);

            passDescriptor->setTileWidth(kLightTileSize);
            passDescriptor->setTileHeight(kLightTileSize);
            passDescriptor->setThreadgroupMemoryLength(kTileLightListThreadgroupBytes + kTileCullingScratchThreadgroupBytes);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "OpaqueDirectLighting: Failed to create render command encoder");

            MTL::Viewport viewport { 0.0, 0.0, static_cast<double>(colorTexture->width()), static_cast<double>(colorTexture->height()), 0.0, 1.0 };

            renderEncoder->setDepthStencilState(data.depthStencilState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setFrontFacingWinding(MTL::WindingClockwise);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex | MTL::RenderStageFragment | MTL::RenderStageTile);

            if (data.maskedPrimitiveStart > 0) {
                renderEncoder->setRenderPipelineState(data.pipelineState);
                renderEncoder->setCullMode(MTL::CullModeBack);
                renderEncoder->executeCommandsInBuffer(indirectCB, NS::Range(0, data.maskedPrimitiveStart));
            }
            const int maskedCount = data.numPrimitives - data.maskedPrimitiveStart;
            if (maskedCount > 0) {
                renderEncoder->setRenderPipelineState(data.maskedPipelineState);
                renderEncoder->setCullMode(MTL::CullModeNone);
                renderEncoder->executeCommandsInBuffer(indirectCB, NS::Range(data.maskedPrimitiveStart, maskedCount));
            }

            renderEncoder->setRenderPipelineState(data.tileCullingPipelineState);
            renderEncoder->setThreadgroupMemoryLength(kTileLightListThreadgroupBytes, 0, 0);
            renderEncoder->setThreadgroupMemoryLength(kTileCullingScratchThreadgroupBytes, kTileLightListThreadgroupBytes, 1);

            renderEncoder->dispatchThreadsPerTile(MTL::Size(kLightTileSize, kLightTileSize, 1));

            renderEncoder->setRenderPipelineState(data.deferredLightingPipelineState);
            renderEncoder->setDepthStencilState(data.deferredLightingDepthStencilState);
            renderEncoder->setCullMode(MTL::CullModeNone);
            renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

            renderEncoder->endEncoding();
        });
}
