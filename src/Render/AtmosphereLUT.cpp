#include "AtmosphereLUT.h"

#include <Core/Buffer.h>
#include <Core/Texture.h>
#include <Core/MetalContext.h>
#include <Core/RenderGraph.h>
#include <Core/RenderGraphResources.h>
#include <Shader/ShaderTypes.h>
#include <Utility/Logger.h>
#include <Utility/ShaderLibrary.h>
#include <glm/glm.hpp>

#define TRANS_WIDTH 256
#define TRANS_HEIGHT 64

#define SKYVIEW_WIDTH 200
#define SKYVIEW_HEIGHT 100

constexpr const char* kAtmosphereLUTShaderLibrary = STONE_SHADER_DIR "/AtmosphereLUT.metallib";

struct TransmittanceArgumentData {
    MTL::ResourceID transmittanceLUT;
};

struct SkyViewArgumentData {
    MTL::ResourceID skyViewLUT;
    MTL::ResourceID transmittanceLUT;
    MTL::GPUAddress directionalLights;
    MTL::GPUAddress param;
    MTL::GPUAddress frameUniform;
};

struct PassData {
    MTL::ComputePipelineState* transmittancePiplineState = nullptr;
    MTL4::ArgumentTable*      transmittanceArgumentTable = nullptr;
    MTL::Buffer*             transmittanceArgumentBuffer = nullptr;
    RenderGraphResourceHandle transmittanceLUTHandle;

    MTL::ComputePipelineState* skyViewPipelineState = nullptr;
    MTL4::ArgumentTable*      skyViewArgumentTable = nullptr;
    RenderGraphResourceHandle skyViewLUTHandle;
    RenderGraphResourceHandle atmosphereUniformsHandle;
    RenderGraphResourceHandle directionalLightBufferHandle;
    RenderGraphResourceHandle frameUniformHandle;
};

void AtmosphereLUT::Setup(MetalContext &context) {
    m_context = &context;
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kAtmosphereLUTShaderLibrary, {
        "transmittance_main", "skyView_main"
    });

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();

    // ==================== TransmittanceLUT ====================
    {
        MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
        pipelineDescriptor->setLabel(NS::String::string("AtmosphereTransmittanceLUT", NS::UTF8StringEncoding));
        pipelineDescriptor->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transmittance_main"));
        m_transmittanceLUTPipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
        LOG_ERROR_IF(!m_transmittanceLUTPipelineState, "Failed to create atmosphere transmittanceLUT compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

        MTL::TextureDescriptor* texDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat::PixelFormatRGBA16Float, TRANS_WIDTH, TRANS_HEIGHT, false);
        texDesc->setStorageMode(MTL::StorageModePrivate);
        texDesc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
        m_transmittanceLUT = std::make_unique<Texture>(device, texDesc);
        m_transmittanceLUT->GetNative()->setLabel(NS::String::string("Atmosphere TransmittanceLUT", NS::UTF8StringEncoding));

        const TransmittanceArgumentData transmittanceData { m_transmittanceLUT->GetNative()->gpuResourceID() };
        m_transmittanceParamsBuffer = std::make_unique<Buffer>(device, sizeof(TransmittanceArgumentData), MTL::ResourceStorageModeShared);
        m_transmittanceParamsBuffer->Update(&transmittanceData, sizeof(TransmittanceArgumentData));
        m_transmittanceParamsBuffer->GetNative()->setLabel(NS::String::string("AtmosphereTransmittanceLUT Argument Buffer", NS::UTF8StringEncoding));

        MTL4::ArgumentTableDescriptor* argTableDesc = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
        argTableDesc->setLabel(NS::String::string("AtmosphereTransmittanceLUT", NS::UTF8StringEncoding));
        argTableDesc->setInitializeBindings(true);
        argTableDesc->setMaxBufferBindCount(static_cast<NS::UInteger>(TransmittanceBufferIndex::MaxBufferBindCount));
        m_transmittanceArgumentTable = device->newArgumentTable(argTableDesc, &error);
        LOG_ERROR_IF(!m_transmittanceArgumentTable, "Failed to create transmittance argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");
        m_transmittanceArgumentTable->setAddress(m_transmittanceParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TransmittanceBufferIndex::KernelArguments));
    }

    // ==================== SkyViewLUT ====================
    {
        MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
        pipelineDescriptor->setLabel(NS::String::string("AtmosphereSkyViewLUT", NS::UTF8StringEncoding));
        pipelineDescriptor->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "skyView_main"));
        m_skyViewLUTPipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
        LOG_ERROR_IF(!m_skyViewLUTPipelineState, "Failed to create atmosphere skyViewLUT compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

        const uint32_t frameSlotCount = context.GetFrameSlotCount();
        m_skyViewLUTs.resize(frameSlotCount);
        m_skyViewParamsBuffers.resize(frameSlotCount);
        m_skyViewAtmosphereUniformsBuffers.resize(frameSlotCount);
        for (uint32_t frameSlot = 0; frameSlot < frameSlotCount; ++frameSlot) {
            MTL::TextureDescriptor* texDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat::PixelFormatRGBA16Float, SKYVIEW_WIDTH, SKYVIEW_HEIGHT, false);
            texDesc->setStorageMode(MTL::StorageModePrivate);
            texDesc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
            m_skyViewLUTs[frameSlot] = std::make_unique<Texture>(device, texDesc);
            m_skyViewLUTs[frameSlot]->GetNative()->setLabel(NS::String::string("Atmosphere SkyViewLUT", NS::UTF8StringEncoding));

            m_skyViewParamsBuffers[frameSlot] = std::make_unique<Buffer>(device, sizeof(SkyViewArgumentData), MTL::ResourceStorageModeShared);
            m_skyViewParamsBuffers[frameSlot]->GetNative()->setLabel(NS::String::string("AtmosphereSkyViewLUT Argument Buffer", NS::UTF8StringEncoding));

            m_skyViewAtmosphereUniformsBuffers[frameSlot] = std::make_unique<Buffer>(device, sizeof(AtmosphereUniforms), MTL::ResourceStorageModeShared);
            m_skyViewAtmosphereUniformsBuffers[frameSlot]->GetNative()->setLabel(NS::String::string("AtmosphereUniforms Buffer", NS::UTF8StringEncoding));
        }

        MTL4::ArgumentTableDescriptor* argTableDesc = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
        argTableDesc->setLabel(NS::String::string("AtmosphereSkyViewLUT", NS::UTF8StringEncoding));
        argTableDesc->setInitializeBindings(true);
        argTableDesc->setMaxBufferBindCount(static_cast<NS::UInteger>(SkyViewBufferIndex::MaxBufferBindCount));
        m_skyViewArgumentTable = device->newArgumentTable(argTableDesc, &error);
        LOG_ERROR_IF(!m_skyViewArgumentTable, "Failed to create skyView argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");
    }

    compiler->release();
}

void AtmosphereLUT::AddToGraph(RenderGraph &graph) {
    RenderGraphResourceHandle transmittanceLUTHandle = graph.RegisterTexture("AtmosphereTransmittanceLUT", *m_transmittanceLUT);
    for (uint32_t frameSlot = 0; frameSlot < m_skyViewLUTs.size(); ++frameSlot) {
        graph.RegisterFrameLocalTexture("AtmosphereSkyViewLUT", frameSlot, *m_skyViewLUTs[frameSlot]);
        graph.RegisterFrameLocalBuffer("AtmosphereUniformsBuffer", frameSlot, *m_skyViewAtmosphereUniformsBuffers[frameSlot]);
    }
    RenderGraphResourceHandle skyViewLUTHandle = graph.DeclareTexture("AtmosphereSkyViewLUT");
    RenderGraphResourceHandle atmosphereUniformsHandle = graph.DeclareBuffer("AtmosphereUniformsBuffer");
    RenderGraphResourceHandle directionalLightBufferHandle = graph.DeclareBuffer("DirectionalLightBuffer");
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");

    graph.AddPass<PassData>(
        "AtmosphereLUT",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, PassData& data, RenderGraphResources&) {
            data.transmittancePiplineState   = m_transmittanceLUTPipelineState;
            data.transmittanceArgumentTable  = m_transmittanceArgumentTable;
            data.transmittanceArgumentBuffer = m_transmittanceParamsBuffer->GetNative();
            data.transmittanceLUTHandle      = transmittanceLUTHandle;

            data.skyViewPipelineState             = m_skyViewLUTPipelineState;
            data.skyViewArgumentTable             = m_skyViewArgumentTable;
            data.skyViewLUTHandle                 = skyViewLUTHandle;
            data.atmosphereUniformsHandle         = atmosphereUniformsHandle;
            data.directionalLightBufferHandle     = directionalLightBufferHandle;
            data.frameUniformHandle               = frameUniformHandle;

            builder.WriteTexture(transmittanceLUTHandle);
            builder.WriteTexture(skyViewLUTHandle);
            builder.WriteBuffer(atmosphereUniformsHandle);
            builder.WriteBuffer(frameUniformHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
        },
        [this](const PassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Texture* transmittanceLUT = resources.GetTexture(data.transmittanceLUTHandle);
            MTL::Texture* skyViewLUT = resources.GetTexture(data.skyViewLUTHandle);
            MTL::Buffer* atmosphereUniformsBuffer = resources.GetBuffer(data.atmosphereUniformsHandle);
            MTL::Buffer* directionalLightBuffer = resources.GetBuffer(data.directionalLightBufferHandle);
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            MTL::Buffer* skyViewArgumentBuffer = m_skyViewParamsBuffers[m_context->GetCurrentFrameSlot()]->GetNative();

            constexpr float kRground = 6360.0f; // km
            const glm::vec3 planetCenter = glm::vec3(0.0f, -kRground, 0.0f);

            const auto* fu = static_cast<const FrameUniform*>(frameUniformBuffer->contents());
            const glm::vec3 cameraWorldPosKm = glm::vec3(fu->cameraPosition) * 0.001f;
            float cameraAltitude = std::max(0.001f, glm::length(cameraWorldPosKm - planetCenter) - kRground);

            const AtmosphereUniforms atmoUniforms {
                .planetCenter = planetCenter,
                .cameraAltitude = cameraAltitude,
                .groundAlbedo = glm::vec3(0.3f, 0.3f, 0.3f),
            };
            memcpy(atmosphereUniformsBuffer->contents(), &atmoUniforms, sizeof(AtmosphereUniforms));

            const SkyViewArgumentData skyViewArgs {
                .skyViewLUT = skyViewLUT->gpuResourceID(),
                .transmittanceLUT = transmittanceLUT->gpuResourceID(),
                .directionalLights = directionalLightBuffer->gpuAddress(),
                .param = atmosphereUniformsBuffer->gpuAddress(),
                .frameUniform = frameUniformBuffer->gpuAddress(),
            };
            memcpy(skyViewArgumentBuffer->contents(), &skyViewArgs, sizeof(SkyViewArgumentData));
            data.skyViewArgumentTable->setAddress(skyViewArgumentBuffer->gpuAddress(), static_cast<NS::UInteger>(SkyViewBufferIndex::KernelArguments));

            if (!hasInit) {
                cmd.AddResource(data.transmittanceArgumentBuffer);
            }
            cmd.AddResource(skyViewArgumentBuffer);
            cmd.AddResource(atmosphereUniformsBuffer);
            cmd.AddResource(skyViewLUT);
            cmd.AddResource(transmittanceLUT);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(frameUniformBuffer);

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();

            // ---- Transmittance LUT (compute once) ----
            if (!hasInit) {
                computeEncoder->setComputePipelineState(data.transmittancePiplineState);
                computeEncoder->setArgumentTable(data.transmittanceArgumentTable);
                NS::UInteger tew = data.transmittancePiplineState->threadExecutionWidth();
                NS::UInteger teh = data.transmittancePiplineState->maxTotalThreadsPerThreadgroup() / tew;
                computeEncoder->dispatchThreads(MTL::Size(TRANS_WIDTH, TRANS_HEIGHT, 1), MTL::Size(tew, teh, 1));
                computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
                hasInit = true;
            }

            // ---- SkyView LUT (every frame) ----
            computeEncoder->setComputePipelineState(data.skyViewPipelineState);
            computeEncoder->setArgumentTable(data.skyViewArgumentTable);
            NS::UInteger tew = data.skyViewPipelineState->threadExecutionWidth();
            NS::UInteger teh = data.skyViewPipelineState->maxTotalThreadsPerThreadgroup() / tew;
            computeEncoder->dispatchThreads(MTL::Size(SKYVIEW_WIDTH, SKYVIEW_HEIGHT, 1), MTL::Size(tew, teh, 1));
            computeEncoder->endEncoding();
        }
    );
}
