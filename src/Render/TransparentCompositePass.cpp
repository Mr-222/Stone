#include "TransparentCompositePass.h"

#include "Core/Buffer.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kTransparentCompositeShaderLibrary = STONE_SHADER_DIR "/TransparentComposite.metallib";

struct TransparentCompositeFragmentArgumentData {
    MTL::ResourceID accumTexture;
    MTL::ResourceID revealTexture;
};

struct TransparentCompositePassData {
    RenderGraphColorAttachment colorAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Buffer* fragmentArgumentBuffer = nullptr;
    RenderGraphResourceHandle accumHandle;
    RenderGraphResourceHandle revealHandle;
};

TransparentCompositePass::~TransparentCompositePass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void TransparentCompositePass::Setup(MetalContext& context) {
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kTransparentCompositeShaderLibrary, {
        "transparentComposite_vertex",
        "transparentComposite_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("TransparentComposite", NS::UTF8StringEncoding));
    pipelineDescriptor->setVertexFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transparentComposite_vertex"));
    pipelineDescriptor->setFragmentFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transparentComposite_fragment"));

    // Standard alpha blending onto the opaque result
    auto* colorAttachment = pipelineDescriptor->colorAttachments()->object(0);
    colorAttachment->setPixelFormat(context.GetSwapchainPixelFormat());
    colorAttachment->setBlendingState(MTL4::BlendStateEnabled);
    colorAttachment->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    colorAttachment->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAttachment->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAttachment->setSourceAlphaBlendFactor(MTL::BlendFactorZero);
    colorAttachment->setDestinationAlphaBlendFactor(MTL::BlendFactorOne);
    colorAttachment->setAlphaBlendOperation(MTL::BlendOperationAdd);

    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create transparent composite pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    const TransparentCompositeFragmentArgumentData fragmentArguments{};
    m_fragmentArgumentBuffer = std::make_unique<Buffer>(device, &fragmentArguments, sizeof(fragmentArguments), MTL::ResourceStorageModeShared);
    m_fragmentArgumentBuffer->GetNative()->setLabel(NS::String::string("TransparentComposite Fragment Argument Buffer", NS::UTF8StringEncoding));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("TransparentComposite Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(TransparentCompositeBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_fragmentArgumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TransparentCompositeBufferIndex::FragmentArguments));

    compiler->release();
}

void TransparentCompositePass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle swapchainHandle = graph.DeclareTexture(kSwapchainImageName);
    RenderGraphResourceHandle accumHandle = graph.DeclareTexture("OITAccumTexture");
    RenderGraphResourceHandle revealHandle = graph.DeclareTexture("OITRevealTexture");

    graph.AddPass<TransparentCompositePassData>(
        "TransparentComposite",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, TransparentCompositePassData& data, RenderGraphResources&) {
            data.colorAttachment = builder.WriteColor(swapchainHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionLoad,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0),
            });
            data.accumHandle = accumHandle;
            data.revealHandle = revealHandle;
            data.pipelineState = m_pipelineState;
            data.argumentTable = m_argumentTable;
            data.fragmentArgumentBuffer = m_fragmentArgumentBuffer->GetNative();

            builder.ReadTexture(accumHandle);
            builder.ReadTexture(revealHandle);
        },
        [this](const TransparentCompositePassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            MTL::Texture* accumTexture = resources.GetTexture(data.accumHandle);
            MTL::Texture* revealTexture = resources.GetTexture(data.revealHandle);

            LOG_ERROR_IF(!colorTexture, "TransparentComposite: No color target.");
            LOG_ERROR_IF(!accumTexture, "TransparentComposite: No accum texture.");
            LOG_ERROR_IF(!revealTexture, "TransparentComposite: No reveal texture.");

            // Update the argument buffer with current frame's OIT textures
            TransparentCompositeFragmentArgumentData fragmentArguments {
                .accumTexture = accumTexture->gpuResourceID(),
                .revealTexture = revealTexture->gpuResourceID(),
            };
            m_fragmentArgumentBuffer->Update(&fragmentArguments, sizeof(fragmentArguments));

            cmd.AddResource(data.fragmentArgumentBuffer);
            cmd.AddResource(accumTexture);
            cmd.AddResource(revealTexture);

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);
            colorAttachment->setTexture(colorTexture);
            colorAttachment->setLoadAction(data.colorAttachment.desc.loadAction);
            colorAttachment->setStoreAction(data.colorAttachment.desc.storeAction);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "TransparentComposite: Failed to create render encoder");

            MTL::Viewport viewport {
                0.0, 0.0,
                static_cast<double>(colorTexture->width()),
                static_cast<double>(colorTexture->height()),
                0.0, 1.0
            };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageFragment);

            renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            renderEncoder->endEncoding();
        });
}
