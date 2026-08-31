#include "OpaqueDirectLightingPass.h"

#include <array>

#include "Core/Buffer.h"
#include "Core/RenderGraph.h"
#include "Core/Texture.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kOpaqueDirectLightingShaderLibrary = STONE_SHADER_DIR "/OpaqueDirectLighting.metallib";

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
    std::array<MTL::ResourceID, kMaxBindlessTextureCount> textures;
};

struct OpaqueDirectLightingPassData {
    RenderGraphColorAttachment colorAttachment;
    RenderGraphDepthAttachment depthAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Buffer* vertexArgumentBuffer = nullptr;
    MTL::Buffer* fragmentArgumentBuffer = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
    RenderGraphResourceHandle opaqueVertexBufferHandle;
    RenderGraphResourceHandle opaqueIndexBufferHandle;
    RenderGraphResourceHandle opaqueRenderPrimitiveBufferHandle;
    RenderGraphResourceHandle materialBufferHandle;
    RenderGraphResourceHandle lightListInfoBufferHandle;
    RenderGraphResourceHandle directionalLightBufferHandle;
    RenderGraphResourceHandle pointLightBufferHandle;
    RenderGraphResourceHandle spotLightBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
    std::vector<MTL::Texture*> textures;
    int numPrimitives;
};

OpaqueDirectLightingPass::~OpaqueDirectLightingPass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_depthStencilState)
        m_depthStencilState->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void OpaqueDirectLightingPass::Setup(
    MetalContext& context,
    const int numPrimitives,
    const std::vector<Texture>& textures)
{
    m_numPrimitives = numPrimitives;
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
        "opaqueDirect_vertex",
        "opaqueDirect_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("OpaqueDirectLighting", NS::UTF8StringEncoding));

    MTL4::LibraryFunctionDescriptor* vertexFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "opaqueDirect_vertex");
    MTL4::LibraryFunctionDescriptor* fragmentFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "opaqueDirect_fragment");

    pipelineDescriptor->setVertexFunctionDescriptor(vertexFunc);
    pipelineDescriptor->setFragmentFunctionDescriptor(fragmentFunc);
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(context.GetSwapchainPixelFormat());
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
    pipelineDescriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create opaque direct lighting render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    depthStencilDescriptor->setLabel(NS::String::string("OpaqueDirectLighting Depth State", NS::UTF8StringEncoding));
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthStencilDescriptor->setDepthWriteEnabled(true);
    m_depthStencilState = device->newDepthStencilState(depthStencilDescriptor);
    LOG_ERROR_IF(!m_depthStencilState, "Failed to create opaque direct lighting depth-stencil state.");

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

    compiler->release();
}

void OpaqueDirectLightingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle swapchainHandle = graph.DeclareTexture(kSwapchainImageName);
    RenderGraphResourceHandle depthHandle = graph.DeclareTexture(kSceneDepthImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");
    RenderGraphResourceHandle opaqueVertexBufferHandle = graph.DeclareBuffer("OpaqueVertexBuffer");
    RenderGraphResourceHandle opaqueIndexBufferHandle = graph.DeclareBuffer("OpaqueIndexBuffer");
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
            data.colorAttachment = builder.WriteColor(swapchainHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0),
            });
            data.depthAttachment = builder.WriteDepth(depthHandle, RenderGraphDepthAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearDepth = 1.0,
            });
            data.pipelineState = m_pipelineState;
            data.depthStencilState = m_depthStencilState;
            data.argumentTable = m_argumentTable;
            data.vertexArgumentBuffer = m_vertexArgumentBuffer->GetNative();
            data.fragmentArgumentBuffer = m_fragmentArgumentBuffer->GetNative();
            data.frameUniformHandle = frameUniformHandle;
            data.opaqueVertexBufferHandle = opaqueVertexBufferHandle;
            data.opaqueIndexBufferHandle = opaqueIndexBufferHandle;
            data.opaqueRenderPrimitiveBufferHandle = opaqueRenderPrimitiveBufferHandle;
            data.materialBufferHandle = materialBufferHandle;
            data.lightListInfoBufferHandle = lightListInfoBufferHandle;
            data.directionalLightBufferHandle = directionalLightBufferHandle;
            data.pointLightBufferHandle = pointLightBufferHandle;
            data.spotLightBufferHandle = spotLightBufferHandle;
            data.indirectCBHandle = indirectCBHandle;
            data.textures = m_textures;
            data.numPrimitives = m_numPrimitives;

            builder.ReadBuffer(frameUniformHandle);
            builder.ReadBuffer(opaqueVertexBufferHandle);
            builder.ReadBuffer(opaqueIndexBufferHandle);
            builder.ReadBuffer(opaqueRenderPrimitiveBufferHandle);
            builder.ReadBuffer(materialBufferHandle);
            builder.ReadBuffer(lightListInfoBufferHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
            builder.ReadBuffer(pointLightBufferHandle);
            builder.ReadBuffer(spotLightBufferHandle);
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
            MTL::Buffer* opaqueRenderPrimitiveBuffer = resources.GetBuffer(data.opaqueRenderPrimitiveBufferHandle);
            MTL::Buffer* materialBuffer = resources.GetBuffer(data.materialBufferHandle);
            MTL::Buffer* lightListInfoBuffer = resources.GetBuffer(data.lightListInfoBufferHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(data.directionalLightBufferHandle);
            MTL::Buffer* pointLightBuffer = resources.GetBuffer(data.pointLightBufferHandle);
            MTL::Buffer* spotLightBuffer = resources.GetBuffer(data.spotLightBufferHandle);
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);

            LOG_ERROR_IF(!frameUniformBuffer, "OpaqueDirectLighting: Failed to get frame uniform buffer");
            LOG_ERROR_IF(!opaqueVertexBuffer, "OpaqueDirectLighting: Failed to get opaque vertex buffer");
            LOG_ERROR_IF(!opaqueIndexBuffer, "OpaqueDirectLighting: Failed to get opaque index buffer");
            LOG_ERROR_IF(!opaqueRenderPrimitiveBuffer, "OpaqueDirectLighting: Failed to get opaque render primitive buffer");
            LOG_ERROR_IF(!materialBuffer, "OpaqueDirectLighting: Failed to get material buffer");
            LOG_ERROR_IF(!lightListInfoBuffer, "OpaqueDirectLighting: Failed to get light list info buffer");
            LOG_ERROR_IF(!directionalLightBuffer, "OpaqueDirectLighting: Failed to get directional light buffer");
            LOG_ERROR_IF(!pointLightBuffer, "OpaqueDirectLighting: Failed to get point light buffer");
            LOG_ERROR_IF(!spotLightBuffer, "OpaqueDirectLighting: Failed to get spot light buffer");
            LOG_ERROR_IF(!indirectCB, "OpaqueDirectLighting: Failed to get indirect command buffer");

            data.argumentTable->setAddress(
                frameUniformBuffer->gpuAddress(),
                static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::FrameUniform));

            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(opaqueVertexBuffer);
            cmd.AddResource(opaqueIndexBuffer);
            cmd.AddResource(opaqueRenderPrimitiveBuffer);
            cmd.AddResource(materialBuffer);
            cmd.AddResource(lightListInfoBuffer);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(pointLightBuffer);
            cmd.AddResource(spotLightBuffer);
            cmd.AddResource(indirectCB);
            cmd.AddResource(data.vertexArgumentBuffer);
            cmd.AddResource(data.fragmentArgumentBuffer);
            for (MTL::Texture* texture : data.textures)
                cmd.AddResource(texture);

            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            MTL::Texture* depthTexture = resources.GetTexture(data.depthAttachment.texture);
            LOG_ERROR_IF(!colorTexture, "OpaqueDirectLighting: No color target.");
            LOG_ERROR_IF(!depthTexture, "OpaqueDirectLighting: No depth target.");
            LOG_ERROR_IF(depthTexture->pixelFormat() != MTL::PixelFormatDepth32Float,
                "OpaqueDirectLighting: Depth target has pixel format {}, expected Depth32Float ({}).",
                static_cast<NS::UInteger>(depthTexture->pixelFormat()),
                static_cast<NS::UInteger>(MTL::PixelFormatDepth32Float));
            LOG_ERROR_IF(
                depthTexture->width() != colorTexture->width() || depthTexture->height() != colorTexture->height(),
                "OpaqueDirectLighting: Depth target size {}x{} does not match color target size {}x{}.",
                depthTexture->width(),
                depthTexture->height(),
                colorTexture->width(),
                colorTexture->height());
            LOG_ERROR_IF(!data.pipelineState, "OpaqueDirectLighting: Pipeline state is null.");
            LOG_ERROR_IF(!data.depthStencilState, "OpaqueDirectLighting: Depth-stencil state is null.");
            LOG_ERROR_IF(!data.argumentTable, "OpaqueDirectLighting: Argument table is null.");

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);
            colorAttachment->setTexture(colorTexture);
            colorAttachment->setLoadAction(data.colorAttachment.desc.loadAction);
            colorAttachment->setClearColor(data.colorAttachment.desc.clearColor);
            colorAttachment->setStoreAction(data.colorAttachment.desc.storeAction);

            MTL::RenderPassDepthAttachmentDescriptor* depthAttachment = passDescriptor->depthAttachment();
            depthAttachment->setTexture(depthTexture);
            depthAttachment->setLoadAction(data.depthAttachment.desc.loadAction);
            depthAttachment->setStoreAction(data.depthAttachment.desc.storeAction);
            depthAttachment->setClearDepth(data.depthAttachment.desc.clearDepth);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "OpaqueDirectLighting: Failed to create render command encoder");

            MTL::Viewport viewport {
                0.0,
                0.0,
                static_cast<double>(colorTexture->width()),
                static_cast<double>(colorTexture->height()),
                0.0,
                1.0
            };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setDepthStencilState(data.depthStencilState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setCullMode(MTL::CullModeBack);
            renderEncoder->setFrontFacingWinding(MTL::WindingClockwise);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex | MTL::RenderStageFragment);

            renderEncoder->executeCommandsInBuffer(indirectCB, NS::Range(0, data.numPrimitives));

            renderEncoder->endEncoding();
        });
}
