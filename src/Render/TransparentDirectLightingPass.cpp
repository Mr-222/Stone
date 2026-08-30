#include "TransparentDirectLightingPass.h"

#include <array>

#include "Core/Buffer.h"
#include "Core/RenderGraph.h"
#include "Core/Texture.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kTransparentDirectLightingShaderLibrary = STONE_SHADER_DIR "/TransparentDirectLighting.metallib";

struct TransparentDirectLightingVertexArgumentData {
    MTL::GPUAddress vertices;
    MTL::GPUAddress renderPrimitives;
};

struct TransparentDirectLightingFragmentArgumentData {
    MTL::GPUAddress materials;
    MTL::GPUAddress lightListInfo;
    MTL::GPUAddress directionalLights;
    std::array<MTL::ResourceID, kMaxBindlessTextureCount> textures;
};

struct TransparentDirectLightingPassData {
    RenderGraphColorAttachment accumAttachment;
    RenderGraphColorAttachment revealAttachment;
    RenderGraphDepthAttachment depthAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Buffer* vertexArgumentBuffer = nullptr;
    MTL::Buffer* fragmentArgumentBuffer = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
    RenderGraphResourceHandle transparentVertexBufferHandle;
    RenderGraphResourceHandle transparentIndexBufferHandle;
    RenderGraphResourceHandle transparentRenderPrimitiveBufferHandle;
    RenderGraphResourceHandle materialBufferHandle;
    RenderGraphResourceHandle lightListInfoBufferHandle;
    RenderGraphResourceHandle directionalLightBufferHandle;
    RenderGraphResourceHandle transparentIndirectCBHandle;
    std::vector<MTL::Texture*> textures;
    int numPrimitives;
};

TransparentDirectLightingPass::~TransparentDirectLightingPass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_depthStencilState)
        m_depthStencilState->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void TransparentDirectLightingPass::Setup(
    MetalContext& context,
    const int numPrimitives,
    const std::vector<Texture>& textures)
{
    m_numPrimitives = numPrimitives;
    m_textures.reserve(textures.size());
    for (size_t i = 0; i < textures.size(); ++i) {
        MTL::Texture* texture = textures[i].GetNative();
        LOG_ERROR_IF(!texture, "TransparentDirectLighting texture {} is null", i);
        m_textures.push_back(texture);
    }

    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kTransparentDirectLightingShaderLibrary, {
        "transparentDirect_vertex",
        "transparentDirect_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("TransparentDirectLighting", NS::UTF8StringEncoding));
    pipelineDescriptor->setVertexFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transparentDirect_vertex"));
    pipelineDescriptor->setFragmentFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transparentDirect_fragment"));

    // ── Color attachment 0: accumulation (RGBA16Float, additive blend) ──
    auto* accumAttach = pipelineDescriptor->colorAttachments()->object(0);
    accumAttach->setPixelFormat(MTL::PixelFormatRGBA16Float);
    accumAttach->setBlendingState(MTL4::BlendStateEnabled);
    accumAttach->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    accumAttach->setDestinationRGBBlendFactor(MTL::BlendFactorOne);
    accumAttach->setRgbBlendOperation(MTL::BlendOperationAdd);
    accumAttach->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    accumAttach->setDestinationAlphaBlendFactor(MTL::BlendFactorOne);
    accumAttach->setAlphaBlendOperation(MTL::BlendOperationAdd);

    // ── Color attachment 1: revealage (R8Unorm, multiplicative via 1-src) ──
    auto* revealAttach = pipelineDescriptor->colorAttachments()->object(1);
    revealAttach->setPixelFormat(MTL::PixelFormatR8Unorm);
    revealAttach->setBlendingState(MTL4::BlendStateEnabled);
    revealAttach->setSourceRGBBlendFactor(MTL::BlendFactorZero);
    revealAttach->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceColor);
    revealAttach->setRgbBlendOperation(MTL::BlendOperationAdd);

    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
    pipelineDescriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create transparent direct lighting pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    // Depth test enabled, depth write DISABLED
    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    depthStencilDescriptor->setLabel(NS::String::string("TransparentDirectLighting Depth State", NS::UTF8StringEncoding));
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLess);
    depthStencilDescriptor->setDepthWriteEnabled(false);
    m_depthStencilState = device->newDepthStencilState(depthStencilDescriptor);
    LOG_ERROR_IF(!m_depthStencilState, "Failed to create transparent direct lighting depth-stencil state.");

    const TransparentDirectLightingVertexArgumentData vertexArguments{};
    m_vertexArgumentBuffer = std::make_unique<Buffer>(device, &vertexArguments, sizeof(vertexArguments), MTL::ResourceStorageModeShared);
    m_vertexArgumentBuffer->GetNative()->setLabel(NS::String::string("TransparentDirectLighting Vertex Argument Buffer", NS::UTF8StringEncoding));

    const TransparentDirectLightingFragmentArgumentData fragmentArguments{};
    m_fragmentArgumentBuffer = std::make_unique<Buffer>(device, &fragmentArguments, sizeof(fragmentArguments), MTL::ResourceStorageModeShared);
    m_fragmentArgumentBuffer->GetNative()->setLabel(NS::String::string("TransparentDirectLighting Fragment Argument Buffer", NS::UTF8StringEncoding));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("TransparentDirectLighting Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(TransparentDirectLightingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_vertexArgumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TransparentDirectLightingBufferIndex::VertexArguments));
    m_argumentTable->setAddress(m_fragmentArgumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TransparentDirectLightingBufferIndex::FragmentArguments));

    const uint32_t frameSlotCount = context.GetFrameSlotCount();
    const uint32_t width = context.GetDrawableWidth();
    const uint32_t height = context.GetDrawableHeight();

    m_oitAccumTextures.resize(frameSlotCount);
    m_oitRevealTextures.resize(frameSlotCount);
    for (uint32_t frameSlot = 0; frameSlot < frameSlotCount; ++frameSlot) {
        MTL::TextureDescriptor* accumDescriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, width, height, false);
        accumDescriptor->setStorageMode(MTL::StorageModePrivate);
        accumDescriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        m_oitAccumTextures[frameSlot] = std::make_unique<Texture>(device, accumDescriptor);
        m_oitAccumTextures[frameSlot]->GetNative()->setLabel(NS::String::string("OIT Accum Texture", NS::UTF8StringEncoding));

        MTL::TextureDescriptor* revealDescriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR8Unorm, width, height, false);
        revealDescriptor->setStorageMode(MTL::StorageModePrivate);
        revealDescriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        m_oitRevealTextures[frameSlot] = std::make_unique<Texture>(device, revealDescriptor);
        m_oitRevealTextures[frameSlot]->GetNative()->setLabel(NS::String::string("OIT Reveal Texture", NS::UTF8StringEncoding));
    }

    compiler->release();
}

void TransparentDirectLightingPass::AddToGraph(RenderGraph& graph) {
    for (uint32_t frameSlot = 0; frameSlot < m_oitAccumTextures.size(); ++frameSlot) {
        graph.RegisterFrameLocalTexture("OITAccumTexture", frameSlot, *m_oitAccumTextures[frameSlot]);
        graph.RegisterFrameLocalTexture("OITRevealTexture", frameSlot, *m_oitRevealTextures[frameSlot]);
    }

    RenderGraphResourceHandle accumHandle = graph.DeclareTexture("OITAccumTexture");
    RenderGraphResourceHandle revealHandle = graph.DeclareTexture("OITRevealTexture");
    RenderGraphResourceHandle depthHandle = graph.DeclareTexture(kSceneDepthImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");
    RenderGraphResourceHandle transparentVertexBufferHandle = graph.DeclareBuffer("TransparentVertexBuffer");
    RenderGraphResourceHandle transparentIndexBufferHandle = graph.DeclareBuffer("TransparentIndexBuffer");
    RenderGraphResourceHandle transparentRenderPrimitiveBufferHandle = graph.DeclareBuffer("TransparentRenderPrimitiveBuffer");
    RenderGraphResourceHandle materialBufferHandle = graph.DeclareBuffer("MaterialBuffer");
    RenderGraphResourceHandle lightListInfoBufferHandle = graph.DeclareBuffer("LightListInfoBuffer");
    RenderGraphResourceHandle directionalLightBufferHandle = graph.DeclareBuffer("DirectionalLightBuffer");
    RenderGraphResourceHandle transparentIndirectCBHandle = graph.DeclareIndirectCommandBuffer("TransparentIndirectCommandBuffer");

    graph.AddPass<TransparentDirectLightingPassData>(
        "TransparentDirectLighting",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, TransparentDirectLightingPassData& data, RenderGraphResources& resources) {
            data.accumAttachment = builder.WriteColor(accumHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0),
            });
            data.revealAttachment = builder.WriteColor(revealHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(1.0, 1.0, 1.0, 1.0),
            });
            data.depthAttachment = builder.ReadDepth(depthHandle, RenderGraphDepthAttachmentDesc{
                .loadAction = MTL::LoadActionLoad,
                .storeAction = MTL::StoreActionDontCare,
                .clearDepth = 1.0,
            });
            data.pipelineState = m_pipelineState;
            data.depthStencilState = m_depthStencilState;
            data.argumentTable = m_argumentTable;
            data.vertexArgumentBuffer = m_vertexArgumentBuffer->GetNative();
            data.fragmentArgumentBuffer = m_fragmentArgumentBuffer->GetNative();
            data.frameUniformHandle = frameUniformHandle;
            data.transparentVertexBufferHandle = transparentVertexBufferHandle;
            data.transparentIndexBufferHandle = transparentIndexBufferHandle;
            data.transparentRenderPrimitiveBufferHandle = transparentRenderPrimitiveBufferHandle;
            data.materialBufferHandle = materialBufferHandle;
            data.lightListInfoBufferHandle = lightListInfoBufferHandle;
            data.directionalLightBufferHandle = directionalLightBufferHandle;
            data.transparentIndirectCBHandle = transparentIndirectCBHandle;
            data.textures = m_textures;
            data.numPrimitives = m_numPrimitives;

            builder.ReadBuffer(frameUniformHandle);
            builder.ReadBuffer(transparentVertexBufferHandle);
            builder.ReadBuffer(transparentIndexBufferHandle);
            builder.ReadBuffer(transparentRenderPrimitiveBufferHandle);
            builder.ReadBuffer(materialBufferHandle);
            builder.ReadBuffer(lightListInfoBufferHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
            builder.ReadIndirectCommandBuffer(transparentIndirectCBHandle);

            MTL::Buffer* transparentVertexBuffer = resources.GetBuffer(transparentVertexBufferHandle);
            MTL::Buffer* transparentRenderPrimitiveBuffer = resources.GetBuffer(transparentRenderPrimitiveBufferHandle);

            if (transparentVertexBuffer && transparentRenderPrimitiveBuffer) {
                const TransparentDirectLightingVertexArgumentData vertexArguments {
                    .vertices = transparentVertexBuffer->gpuAddress(),
                    .renderPrimitives = transparentRenderPrimitiveBuffer->gpuAddress(),
                };
                m_vertexArgumentBuffer->Update(&vertexArguments, sizeof(vertexArguments));
            }

            MTL::Buffer* materialBuffer = resources.GetBuffer(materialBufferHandle);
            MTL::Buffer* lightListInfoBuffer = resources.GetBuffer(lightListInfoBufferHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(directionalLightBufferHandle);

            if (materialBuffer && lightListInfoBuffer && directionalLightBuffer && !m_textures.empty()) {
                TransparentDirectLightingFragmentArgumentData fragmentArguments {
                    .materials = materialBuffer->gpuAddress(),
                    .lightListInfo = lightListInfoBuffer->gpuAddress(),
                    .directionalLights = directionalLightBuffer->gpuAddress(),
                };
                for (size_t i = 0; i < fragmentArguments.textures.size(); ++i) {
                    MTL::Texture* texture = i < m_textures.size()
                        ? m_textures[i]
                        : m_textures.front();
                    fragmentArguments.textures[i] = texture->gpuResourceID();
                }
                m_fragmentArgumentBuffer->Update(&fragmentArguments, sizeof(fragmentArguments));
            }
        },
        [](const TransparentDirectLightingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            MTL::Buffer* transparentVertexBuffer = resources.GetBuffer(data.transparentVertexBufferHandle);
            MTL::Buffer* transparentIndexBuffer = resources.GetBuffer(data.transparentIndexBufferHandle);
            MTL::Buffer* transparentRenderPrimitiveBuffer = resources.GetBuffer(data.transparentRenderPrimitiveBufferHandle);
            MTL::Buffer* materialBuffer = resources.GetBuffer(data.materialBufferHandle);
            MTL::Buffer* lightListInfoBuffer = resources.GetBuffer(data.lightListInfoBufferHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(data.directionalLightBufferHandle);
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.transparentIndirectCBHandle);

            LOG_ERROR_IF(!frameUniformBuffer, "TransparentDirectLighting: Failed to get frame uniform buffer");
            LOG_ERROR_IF(!transparentVertexBuffer, "TransparentDirectLighting: Failed to get transparent vertex buffer");
            LOG_ERROR_IF(!transparentIndexBuffer, "TransparentDirectLighting: Failed to get transparent index buffer");
            LOG_ERROR_IF(!transparentRenderPrimitiveBuffer, "TransparentDirectLighting: Failed to get transparent render primitive buffer");
            LOG_ERROR_IF(!materialBuffer, "TransparentDirectLighting: Failed to get material buffer");
            LOG_ERROR_IF(!lightListInfoBuffer, "TransparentDirectLighting: Failed to get light list info buffer");
            LOG_ERROR_IF(!directionalLightBuffer, "TransparentDirectLighting: Failed to get directional light buffer");
            LOG_ERROR_IF(!indirectCB, "TransparentDirectLighting: Failed to get indirect command buffer");

            data.argumentTable->setAddress(frameUniformBuffer->gpuAddress(), static_cast<NS::UInteger>(TransparentDirectLightingBufferIndex::FrameUniform));

            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(transparentVertexBuffer);
            cmd.AddResource(transparentIndexBuffer);
            cmd.AddResource(transparentRenderPrimitiveBuffer);
            cmd.AddResource(materialBuffer);
            cmd.AddResource(lightListInfoBuffer);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(indirectCB);
            cmd.AddResource(data.vertexArgumentBuffer);
            cmd.AddResource(data.fragmentArgumentBuffer);
            for (MTL::Texture* texture : data.textures)
                cmd.AddResource(texture);

            MTL::Texture* accumTexture = resources.GetTexture(data.accumAttachment.texture);
            MTL::Texture* revealTexture = resources.GetTexture(data.revealAttachment.texture);
            MTL::Texture* depthTexture = resources.GetTexture(data.depthAttachment.texture);
            LOG_ERROR_IF(!accumTexture, "TransparentDirectLighting: No accum target.");
            LOG_ERROR_IF(!revealTexture, "TransparentDirectLighting: No reveal target.");
            LOG_ERROR_IF(!depthTexture, "TransparentDirectLighting: No depth target.");

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();

            // Color attachment 0: accumulation
            MTL::RenderPassColorAttachmentDescriptor* accumColorAttachment = passDescriptor->colorAttachments()->object(0);
            accumColorAttachment->setTexture(accumTexture);
            accumColorAttachment->setLoadAction(data.accumAttachment.desc.loadAction);
            accumColorAttachment->setClearColor(data.accumAttachment.desc.clearColor);
            accumColorAttachment->setStoreAction(data.accumAttachment.desc.storeAction);

            // Color attachment 1: revealage
            MTL::RenderPassColorAttachmentDescriptor* revealColorAttachment = passDescriptor->colorAttachments()->object(1);
            revealColorAttachment->setTexture(revealTexture);
            revealColorAttachment->setLoadAction(data.revealAttachment.desc.loadAction);
            revealColorAttachment->setClearColor(data.revealAttachment.desc.clearColor);
            revealColorAttachment->setStoreAction(data.revealAttachment.desc.storeAction);

            // Depth attachment (read-only: test only, no write)
            MTL::RenderPassDepthAttachmentDescriptor* depthAttachment = passDescriptor->depthAttachment();
            depthAttachment->setTexture(depthTexture);
            depthAttachment->setLoadAction(data.depthAttachment.desc.loadAction);
            depthAttachment->setStoreAction(data.depthAttachment.desc.storeAction);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "TransparentDirectLighting: Failed to create render command encoder");

            MTL::Viewport viewport {
                0.0, 0.0,
                static_cast<double>(accumTexture->width()),
                static_cast<double>(accumTexture->height()),
                0.0, 1.0
            };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setDepthStencilState(data.depthStencilState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setCullMode(MTL::CullModeNone);
            renderEncoder->setFrontFacingWinding(MTL::WindingClockwise);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex | MTL::RenderStageFragment);

            if (data.numPrimitives > 0)
                renderEncoder->executeCommandsInBuffer(indirectCB, NS::Range(0, data.numPrimitives));

            renderEncoder->endEncoding();
        });
}
