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

#define MULTISCATTERING_WIDTH 32
#define MULTISCATTERING_HEIGHT 32

#define SKYVIEW_WIDTH 200
#define SKYVIEW_HEIGHT 100

constexpr const char* kAtmosphereLUTShaderLibrary = STONE_SHADER_DIR "/AtmosphereLUT.metallib";

struct TransmittanceArgumentData {
    MTL::ResourceID transmittanceLUT;
};

struct MultiScatteringArgumentData {
    MTL::ResourceID multiScatteringLUT;
    MTL::ResourceID transmittanceLUT;
    MTL::GPUAddress param;
};

struct SkyViewArgumentData {
    MTL::ResourceID skyViewLUT;
    MTL::ResourceID transmittanceLUT;
    MTL::GPUAddress directionalLights;
    MTL::GPUAddress param;
    MTL::GPUAddress frameUniform;
    MTL::ResourceID multiScatteringLUT;
};

struct PassData {
    MTL::ComputePipelineState* transmittancePiplineState = nullptr;
    MTL4::ArgumentTable* transmittanceArgumentTable = nullptr;
    MTL::Buffer* transmittanceArgumentBuffer = nullptr;
    RenderGraphResourceHandle transmittanceLUTHandle;

    MTL::ComputePipelineState* multiScatteringPipelineState = nullptr;
    MTL4::ArgumentTable* multiScatteringArgumentTable = nullptr;
    MTL::Buffer* multiScatteringArgumentBuffer = nullptr;
    RenderGraphResourceHandle multiScatteringLUTHandle;

    MTL::ComputePipelineState* skyViewPipelineState = nullptr;
    MTL4::ArgumentTable* skyViewArgumentTable = nullptr;
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
        "transmittance_main", "multiScattering_main", "skyView_main"
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

    // ==================== MultiScatteringLUT ====================
    {
        MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
        pipelineDescriptor->setLabel(NS::String::string("AtmosphereMultiScatteringLUT", NS::UTF8StringEncoding));
        pipelineDescriptor->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "multiScattering_main"));
        m_multiScatteringLUTPipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
        LOG_ERROR_IF(!m_multiScatteringLUTPipelineState, "Failed to create atmosphere multiScatteringLUT compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

        MTL::TextureDescriptor* texDesc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat::PixelFormatRGBA16Float, MULTISCATTERING_WIDTH, MULTISCATTERING_HEIGHT, false);
        texDesc->setStorageMode(MTL::StorageModePrivate);
        texDesc->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
        m_multiScatteringLUT = std::make_unique<Texture>(device, texDesc);
        m_multiScatteringLUT->GetNative()->setLabel(NS::String::string("Atmosphere MultiScatteringLUT", NS::UTF8StringEncoding));

        m_multiScatteringParamsBuffer = std::make_unique<Buffer>(device, sizeof(MultiScatteringArgumentData), MTL::ResourceStorageModeShared);
        m_multiScatteringParamsBuffer->GetNative()->setLabel(NS::String::string("AtmosphereMultiScatteringLUT Argument Buffer", NS::UTF8StringEncoding));

        MTL4::ArgumentTableDescriptor* argTableDesc = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
        argTableDesc->setLabel(NS::String::string("AtmosphereMultiScatteringLUT", NS::UTF8StringEncoding));
        argTableDesc->setInitializeBindings(true);
        argTableDesc->setMaxBufferBindCount(static_cast<NS::UInteger>(MultiScatteringBufferIndex::MaxBufferBindCount));
        m_multiScatteringArgumentTable = device->newArgumentTable(argTableDesc, &error);
        LOG_ERROR_IF(!m_multiScatteringArgumentTable, "Failed to create multiScattering argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");
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
    RenderGraphResourceHandle multiScatteringLUTHandle = graph.RegisterTexture("AtmosphereMultiScatteringLUT", *m_multiScatteringLUT);
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

            data.multiScatteringPipelineState    = m_multiScatteringLUTPipelineState;
            data.multiScatteringArgumentTable    = m_multiScatteringArgumentTable;
            data.multiScatteringArgumentBuffer   = m_multiScatteringParamsBuffer->GetNative();
            data.multiScatteringLUTHandle        = multiScatteringLUTHandle;

            data.skyViewPipelineState             = m_skyViewLUTPipelineState;
            data.skyViewArgumentTable             = m_skyViewArgumentTable;
            data.skyViewLUTHandle                 = skyViewLUTHandle;
            data.atmosphereUniformsHandle         = atmosphereUniformsHandle;
            data.directionalLightBufferHandle     = directionalLightBufferHandle;
            data.frameUniformHandle               = frameUniformHandle;

            builder.WriteTexture(transmittanceLUTHandle);
            builder.WriteTexture(multiScatteringLUTHandle);
            builder.WriteTexture(skyViewLUTHandle);
            builder.WriteBuffer(atmosphereUniformsHandle);
            builder.WriteBuffer(frameUniformHandle);
            builder.ReadBuffer(directionalLightBufferHandle);
        },
        [this](const PassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Texture* transmittanceLUT = resources.GetTexture(data.transmittanceLUTHandle);
            MTL::Texture* multiScatteringLUT = resources.GetTexture(data.multiScatteringLUTHandle);
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
                .multiScatteringLUT = multiScatteringLUT->gpuResourceID(),
            };
            memcpy(skyViewArgumentBuffer->contents(), &skyViewArgs, sizeof(SkyViewArgumentData));
            data.skyViewArgumentTable->setAddress(skyViewArgumentBuffer->gpuAddress(), static_cast<NS::UInteger>(SkyViewBufferIndex::KernelArguments));

            if (!hasInit) {
                cmd.AddResource(data.transmittanceArgumentBuffer);
                cmd.AddResource(data.multiScatteringArgumentBuffer);
            }
            cmd.AddResource(skyViewArgumentBuffer);
            cmd.AddResource(atmosphereUniformsBuffer);
            cmd.AddResource(skyViewLUT);
            cmd.AddResource(transmittanceLUT);
            cmd.AddResource(multiScatteringLUT);
            cmd.AddResource(directionalLightBuffer);
            cmd.AddResource(frameUniformBuffer);

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();

            // ---- Transmittance LUT & MultiScattering LUT (compute once) ----
            if (!hasInit) {
                computeEncoder->setComputePipelineState(data.transmittancePiplineState);
                computeEncoder->setArgumentTable(data.transmittanceArgumentTable);
                NS::UInteger tew = data.transmittancePiplineState->threadExecutionWidth();
                NS::UInteger teh = data.transmittancePiplineState->maxTotalThreadsPerThreadgroup() / tew;
                computeEncoder->dispatchThreads(MTL::Size(TRANS_WIDTH, TRANS_HEIGHT, 1), MTL::Size(tew, teh, 1));
                computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

                const MultiScatteringArgumentData multiScatteringArgs {
                    .multiScatteringLUT = multiScatteringLUT->gpuResourceID(),
                    .transmittanceLUT = transmittanceLUT->gpuResourceID(),
                    .param = atmosphereUniformsBuffer->gpuAddress(),
                };
                m_multiScatteringParamsBuffer->Update(&multiScatteringArgs, sizeof(MultiScatteringArgumentData));
                data.multiScatteringArgumentTable->setAddress(m_multiScatteringParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(MultiScatteringBufferIndex::KernelArguments));

                computeEncoder->setComputePipelineState(data.multiScatteringPipelineState);
                computeEncoder->setArgumentTable(data.multiScatteringArgumentTable);
                NS::UInteger tewMS = data.multiScatteringPipelineState->threadExecutionWidth();
                NS::UInteger tehMS = data.multiScatteringPipelineState->maxTotalThreadsPerThreadgroup() / tewMS;
                computeEncoder->dispatchThreads(MTL::Size(MULTISCATTERING_WIDTH, MULTISCATTERING_HEIGHT, 1), MTL::Size(tewMS, tehMS, 1));
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
