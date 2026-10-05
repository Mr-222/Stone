#include "AtmosphereScatteringPass.h"

#include "Core/Buffer.h"
#include "Core/MetalContext.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kAtmosphereScatteringShaderLibrary = STONE_SHADER_DIR "/AtmosphereScattering.metallib";

// Must match AtmosphereScatteringFragmentArguments in ShaderTypes.h
struct AtmosphereScatteringFragmentArgumentData {
    MTL::ResourceID skyViewLUT;
    MTL::ResourceID transmittanceLUT;
    MTL::GPUAddress directionalLights;
    MTL::GPUAddress param;
};

struct AtmosphereScatteringPassData {
    RenderGraphColorAttachment colorAttachment;
    RenderGraphDepthAttachment depthAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL::DepthStencilState* depthStencilState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
    RenderGraphResourceHandle skyViewLUTHandle;
    RenderGraphResourceHandle transmittanceLUTHandle;
    RenderGraphResourceHandle directionalLightBufferHandle;
    RenderGraphResourceHandle atmosphereUniformsHandle;
};

AtmosphereScatteringPass::~AtmosphereScatteringPass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_depthStencilState)
        m_depthStencilState->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void AtmosphereScatteringPass::Setup(MetalContext& context) {
    m_context = &context;
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kAtmosphereScatteringShaderLibrary, {
        "atmosphereScattering_vertex",
        "atmosphereScattering_fragment",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("AtmosphereScattering", NS::UTF8StringEncoding));
    pipelineDescriptor->setVertexFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "atmosphereScattering_vertex"));
    pipelineDescriptor->setFragmentFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "atmosphereScattering_fragment"));
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(context.GetSwapchainPixelFormat());
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create atmosphere scattering pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    // Sky is drawn at the far plane (z = 1): LessEqual passes only where no opaque geometry was written, depth write DISABLED
    MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init()->autorelease();
    depthStencilDescriptor->setLabel(NS::String::string("AtmosphereScattering Depth State", NS::UTF8StringEncoding));
    depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    depthStencilDescriptor->setDepthWriteEnabled(false);
    m_depthStencilState = device->newDepthStencilState(depthStencilDescriptor);
    LOG_ERROR_IF(!m_depthStencilState, "Failed to create atmosphere scattering depth-stencil state.");

    const uint32_t frameSlotCount = context.GetFrameSlotCount();
    m_fragmentArgumentBuffers.resize(frameSlotCount);
    for (uint32_t frameSlot = 0; frameSlot < frameSlotCount; ++frameSlot) {
        const AtmosphereScatteringFragmentArgumentData fragmentArguments{};
        m_fragmentArgumentBuffers[frameSlot] = std::make_unique<Buffer>(device, &fragmentArguments, sizeof(fragmentArguments), MTL::ResourceStorageModeShared);
        m_fragmentArgumentBuffers[frameSlot]->GetNative()->setLabel(NS::String::string("AtmosphereScattering Fragment Argument Buffer", NS::UTF8StringEncoding));
    }

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("AtmosphereScattering Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(AtmosphereScatteringBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    compiler->release();
}

void AtmosphereScatteringPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle sceneColorHandle = graph.DeclareTexture(kSceneColorImageName);
    RenderGraphResourceHandle depthHandle = graph.DeclareTexture(kSceneDepthImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");
    RenderGraphResourceHandle skyViewLUTHandle = graph.DeclareTexture("AtmosphereSkyViewLUT");
    RenderGraphResourceHandle transmittanceLUTHandle = graph.DeclareTexture("AtmosphereTransmittanceLUT");
    RenderGraphResourceHandle directionalLightBufferHandle = graph.DeclareBuffer("DirectionalLightBuffer");
    RenderGraphResourceHandle atmosphereUniformsHandle = graph.DeclareBuffer("AtmosphereUniformsBuffer");

    graph.AddPass<AtmosphereScatteringPassData>(
        "AtmosphereScattering",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, AtmosphereScatteringPassData& data, RenderGraphResources&) {
            data.colorAttachment = builder.WriteColor(sceneColorHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionLoad,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0),
            });
            // Depth is test-only here but must be stored, TransparentDirectLighting loads it afterwards
            data.depthAttachment = builder.ReadDepth(depthHandle, RenderGraphDepthAttachmentDesc{
                .loadAction = MTL::LoadActionLoad,
                .storeAction = MTL::StoreActionStore,
                .clearDepth = 1.0,
            });
            data.pipelineState = m_pipelineState;
            data.depthStencilState = m_depthStencilState;
            data.argumentTable = m_argumentTable;
            data.frameUniformHandle = frameUniformHandle;
            data.skyViewLUTHandle = skyViewLUTHandle;
            data.transmittanceLUTHandle = transmittanceLUTHandle;
            data.directionalLightBufferHandle = directionalLightBufferHandle;
            data.atmosphereUniformsHandle = atmosphereUniformsHandle;

            builder.ReadBuffer(frameUniformHandle);
            builder.ReadTexture(skyViewLUTHandle);
            builder.ReadTexture(transmittanceLUTHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
            builder.ReadBuffer(atmosphereUniformsHandle);
        },
        [this](const AtmosphereScatteringPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            MTL::Texture* skyViewLUT = resources.GetTexture(data.skyViewLUTHandle);
            MTL::Texture* transmittanceLUT = resources.GetTexture(data.transmittanceLUTHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(data.directionalLightBufferHandle);
            MTL::Buffer* atmosphereUniformsBuffer = resources.GetBuffer(data.atmosphereUniformsHandle);
            MTL::Buffer* fragmentArgumentBuffer = m_fragmentArgumentBuffers[m_context->GetCurrentFrameSlot()]->GetNative();

            LOG_ERROR_IF(!frameUniformBuffer, "AtmosphereScattering: Failed to get frame uniform buffer");
            LOG_ERROR_IF(!skyViewLUT, "AtmosphereScattering: Failed to get sky view LUT");
            LOG_ERROR_IF(!transmittanceLUT, "AtmosphereScattering: Failed to get transmittance LUT");
            LOG_ERROR_IF(!directionalLightBuffer, "AtmosphereScattering: Failed to get directional light buffer");
            LOG_ERROR_IF(!atmosphereUniformsBuffer, "AtmosphereScattering: Failed to get atmosphere uniforms buffer");

            const AtmosphereScatteringFragmentArgumentData fragmentArguments {
                .skyViewLUT = skyViewLUT->gpuResourceID(),
                .transmittanceLUT = transmittanceLUT->gpuResourceID(),
                .directionalLights = directionalLightBuffer->gpuAddress(),
                .param = atmosphereUniformsBuffer->gpuAddress(),
            };
            memcpy(fragmentArgumentBuffer->contents(), &fragmentArguments, sizeof(fragmentArguments));

            data.argumentTable->setAddress(frameUniformBuffer->gpuAddress(), static_cast<NS::UInteger>(AtmosphereScatteringBufferIndex::FrameUniform));
            data.argumentTable->setAddress(fragmentArgumentBuffer->gpuAddress(), static_cast<NS::UInteger>(AtmosphereScatteringBufferIndex::FragmentArguments));

            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            MTL::Texture* depthTexture = resources.GetTexture(data.depthAttachment.texture);
            LOG_ERROR_IF(!colorTexture, "AtmosphereScattering: No color target.");
            LOG_ERROR_IF(!depthTexture, "AtmosphereScattering: No depth target.");

            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(skyViewLUT);
            cmd.AddResource(transmittanceLUT);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(atmosphereUniformsBuffer);
            cmd.AddResource(fragmentArgumentBuffer);
            cmd.AddResource(colorTexture);
            cmd.AddResource(depthTexture);

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
            LOG_ERROR_IF(!renderEncoder, "AtmosphereScattering: Failed to create render command encoder");

            MTL::Viewport viewport { 0.0, 0.0, static_cast<double>(colorTexture->width()), static_cast<double>(colorTexture->height()), 0.0, 1.0 };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setDepthStencilState(data.depthStencilState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setCullMode(MTL::CullModeNone);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex | MTL::RenderStageFragment);

            // Fullscreen triangle
            renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
            renderEncoder->endEncoding();
        });
}
