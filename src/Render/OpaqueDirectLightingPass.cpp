#include "OpaqueDirectLightingPass.h"

#include "Core/Buffer.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kOpaqueDirectLightingShaderLibrary = STONE_SHADER_DIR "/OpaqueDirectLighting.metallib";

struct OpaqueDirectLightingPassData {
    RenderGraphColorAttachment colorAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Buffer* argumentBuffer = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
    RenderGraphResourceHandle globalVertexBufferHandle;
    RenderGraphResourceHandle globalIndexBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
    int numPrimitives;
};

OpaqueDirectLightingPass::OpaqueDirectLightingPass() = default;

OpaqueDirectLightingPass::~OpaqueDirectLightingPass() {
    if (m_argumentEncoder)
        m_argumentEncoder->release();
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void OpaqueDirectLightingPass::Setup(MetalContext& context, const int numPrimitives) {
    m_numPrimitives = numPrimitives;

    MTL::Device* device = context.GetDevice();

    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kOpaqueDirectLightingShaderLibrary, {
        "opaqueDirect_vertex",
        "opaqueDirect_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();

    MTL4::LibraryFunctionDescriptor* vertexFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "opaqueDirect_vertex");
    MTL4::LibraryFunctionDescriptor* fragmentFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "opaqueDirect_fragment");

    pipelineDescriptor->setVertexFunctionDescriptor(vertexFunc);
    pipelineDescriptor->setFragmentFunctionDescriptor(fragmentFunc);
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);
    pipelineDescriptor->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create opaque direct lighting render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::Function* vertexFunction = shaderLibrary.GetFunction("opaqueDirect_vertex");

    m_argumentEncoder = vertexFunction->newArgumentEncoder(static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::BindlessArguments));
    LOG_ERROR_IF(!m_argumentEncoder, "Failed to create argument encoder for opaque direct lighting bindings");

    m_argumentBuffer = std::make_unique<Buffer>(
        device,
        m_argumentEncoder->encodedLength(),
        MTL::ResourceStorageModeShared
    );
    m_argumentBuffer->GetNative()->setLabel(NS::String::string("OpaqueDirectLighting Argument Buffer", NS::UTF8StringEncoding));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("OpaqueDirectLighting Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_argumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::BindlessArguments));

    compiler->release();
}

void OpaqueDirectLightingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle swapchainHandle = graph.DeclareTexture(kSwapchainImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");
    RenderGraphResourceHandle globalVertexBufferHandle = graph.DeclareBuffer("GlobalVertexBuffer");
    RenderGraphResourceHandle globalIndexBufferHandle = graph.DeclareBuffer("GlobalIndexBuffer");
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
            data.pipelineState = m_pipelineState;
            data.argumentTable = m_argumentTable;
            data.argumentBuffer = m_argumentBuffer->GetNative();
            data.frameUniformHandle = frameUniformHandle;
            data.globalVertexBufferHandle = globalVertexBufferHandle;
            data.globalIndexBufferHandle = globalIndexBufferHandle;
            data.indirectCBHandle = indirectCBHandle;
            data.numPrimitives = m_numPrimitives;

            builder.ReadBuffer(frameUniformHandle);
            builder.ReadBuffer(globalVertexBufferHandle);
            builder.ReadBuffer(globalIndexBufferHandle);
            builder.ReadIndirectCommandBuffer(indirectCBHandle);

            // Encode the global vertex buffer into the shader argument-buffer layout.
            MTL::Buffer* globalVertexBuffer = resources.GetBuffer(globalVertexBufferHandle);
            LOG_ERROR_IF(!globalVertexBuffer, "OpaqueDirectLighting: Failed to get global vertex buffer");
            m_argumentEncoder->setArgumentBuffer(m_argumentBuffer->GetNative(), 0);
            m_argumentEncoder->setBuffer(
                globalVertexBuffer,
                0,
                static_cast<NS::UInteger>(OpaqueDirectLightingBindlessArgumentID::Vertices));

            // Bind frame uniform into the argument table
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(frameUniformHandle);
            LOG_ERROR_IF(!frameUniformBuffer, "OpaqueDirectLighting: Failed to get frame uniform buffer");
            data.argumentTable->setAddress(frameUniformBuffer->gpuAddress(), static_cast<NS::UInteger>(OpaqueDirectLightingBufferIndex::FrameUniform));
        },
        [](const OpaqueDirectLightingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            MTL::Buffer* globalVertexBuffer = resources.GetBuffer(data.globalVertexBufferHandle);
            MTL::Buffer* globalIndexBuffer = resources.GetBuffer(data.globalIndexBufferHandle);
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);

            LOG_ERROR_IF(!globalVertexBuffer, "OpaqueDirectLighting: Failed to get global vertex buffer");
            LOG_ERROR_IF(!globalIndexBuffer, "OpaqueDirectLighting: Failed to get global index buffer");
            LOG_ERROR_IF(!indirectCB, "OpaqueDirectLighting: Failed to get indirect command buffer");

            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(globalVertexBuffer);
            cmd.AddResource(globalIndexBuffer);
            cmd.AddResource(indirectCB);
            cmd.AddResource(data.argumentBuffer);

            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            LOG_ERROR_IF(!colorTexture, "OpaqueDirectLighting: No color target.");
            LOG_ERROR_IF(!data.pipelineState, "OpaqueDirectLighting: Pipeline state is null.");
            LOG_ERROR_IF(!data.argumentTable, "OpaqueDirectLighting: Argument table is null.");

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);
            colorAttachment->setTexture(colorTexture);
            colorAttachment->setLoadAction(data.colorAttachment.desc.loadAction);
            colorAttachment->setClearColor(data.colorAttachment.desc.clearColor);
            colorAttachment->setStoreAction(data.colorAttachment.desc.storeAction);

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
            renderEncoder->setViewport(viewport);
            renderEncoder->setCullMode(MTL::CullModeBack);
            renderEncoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex);

            renderEncoder->executeCommandsInBuffer(indirectCB, NS::Range(0, data.numPrimitives));

            renderEncoder->endEncoding();
        });
}
