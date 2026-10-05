#include "TonemappingPass.h"

#include "Core/Buffer.h"
#include "Core/MetalContext.h"
#include "Core/RenderGraph.h"
#include "Core/Window.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kTonemappingShaderLibrary = STONE_SHADER_DIR "/Tonemapping.metallib";

struct TonemappingFragmentArgumentData {
    MTL::ResourceID sceneColorTexture;
    float edrHeadroom;
    float padding = 0.0f;
};
static_assert(sizeof(TonemappingFragmentArgumentData) == 16);

struct TonemappingPassData {
    RenderGraphColorAttachment colorAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    RenderGraphResourceHandle sceneColorHandle;
};

TonemappingPass::~TonemappingPass() {
    if (m_argumentTable) m_argumentTable->release();
    if (m_pipelineState) m_pipelineState->release();
}

void TonemappingPass::Setup(MetalContext& context, const Window& window) {
    m_context = &context;
    m_window = &window;
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kTonemappingShaderLibrary, {
        "tonemapping_vertex",
        "tonemapping_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("Tonemapping", NS::UTF8StringEncoding));
    pipelineDescriptor->setVertexFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "tonemapping_vertex"));
    pipelineDescriptor->setFragmentFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "tonemapping_fragment"));
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(context.GetSwapchainPixelFormat());
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create tonemapping pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    const uint32_t frameSlotCount = context.GetFrameSlotCount();
    m_fragmentArgumentBuffers.resize(frameSlotCount);
    for (uint32_t i = 0; i < frameSlotCount; ++i) {
        const TonemappingFragmentArgumentData fragmentArguments{};
        m_fragmentArgumentBuffers[i] = std::make_unique<Buffer>(device, &fragmentArguments, sizeof(fragmentArguments), MTL::ResourceStorageModeShared);
        m_fragmentArgumentBuffers[i]->GetNative()->setLabel(NS::String::string("Tonemapping Fragment Argument Buffer", NS::UTF8StringEncoding));
    }

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Tonemapping Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(TonemappingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    compiler->release();
}

void TonemappingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle sceneColorHandle = graph.DeclareTexture(kSceneColorImageName);
    RenderGraphResourceHandle swapchainHandle = graph.DeclareTexture(kSwapchainImageName);

    graph.AddPass<TonemappingPassData>(
        "Tonemapping",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, TonemappingPassData& data, RenderGraphResources&) {
            data.colorAttachment = builder.WriteColor(swapchainHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionDontCare,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0),
            });
            data.sceneColorHandle = sceneColorHandle;
            data.pipelineState = m_pipelineState;
            data.argumentTable = m_argumentTable;

            builder.ReadTexture(sceneColorHandle);
        },
        [this](const TonemappingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            const uint32_t frameSlot = m_context->GetCurrentFrameSlot();
            MTL::Texture* sceneColorTexture = resources.GetTexture(data.sceneColorHandle);
            MTL::Texture* swapchainTexture = resources.GetTexture(data.colorAttachment.texture);

            LOG_ERROR_IF(!sceneColorTexture, "Tonemapping: No scene color texture.");
            LOG_ERROR_IF(!swapchainTexture, "Tonemapping: No swapchain texture.");

            const float edrHeadroom = m_window ? m_window->GetEDRHeadroom() : 1.0f;
            TonemappingFragmentArgumentData fragmentArguments {
                .sceneColorTexture = sceneColorTexture->gpuResourceID(),
                .edrHeadroom = edrHeadroom,
                .padding = 0.0f,
            };
            m_fragmentArgumentBuffers[frameSlot]->Update(&fragmentArguments, sizeof(fragmentArguments));

            data.argumentTable->setAddress(m_fragmentArgumentBuffers[frameSlot]->GetGPUAddress(), static_cast<NS::UInteger>(TonemappingBufferIndex::FragmentArguments));

            cmd.AddResource(m_fragmentArgumentBuffers[frameSlot]->GetNative());
            cmd.AddResource(sceneColorTexture);
            cmd.AddResource(swapchainTexture);

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);
            colorAttachment->setTexture(swapchainTexture);
            colorAttachment->setLoadAction(data.colorAttachment.desc.loadAction);
            colorAttachment->setStoreAction(data.colorAttachment.desc.storeAction);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "Tonemapping: Failed to create render encoder");

            MTL::Viewport viewport {
                0.0, 0.0,
                static_cast<double>(swapchainTexture->width()),
                static_cast<double>(swapchainTexture->height()),
                0.0, 1.0
            };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageFragment);

            renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            renderEncoder->endEncoding();
        });
}
