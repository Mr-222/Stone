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

constexpr const char* kAtmosphereLUTShaderLibrary = STONE_SHADER_DIR "/AtmosphereScattering.metallib";

struct TransmittanceArgumentData {
    MTL::ResourceID transmittanceLUT;
    glm::uvec2 texSize;
};

struct PassData {
    MTL::ComputePipelineState* transmittancePiplineState = nullptr;
    MTL4::ArgumentTable*      transmittanceArgumentTable = nullptr;
    MTL::Buffer*             transmittanceArgumentBuffer = nullptr;
    RenderGraphResourceHandle transmittanceLUTHandle;
};

void AtmosphereLUT::Setup(MetalContext &context) {
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    // TransmittanceLUT pipeline
    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kAtmosphereLUTShaderLibrary, {
        "transmittance_main"
    });

    MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setLabel(NS::String::string("AtmosphereTransmittanceLUT", NS::UTF8StringEncoding));
    pipelineDescriptor->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "transmittance_main"));

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_transmittanceLUTPipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_transmittanceLUTPipelineState, "Failed to create atmosphere transmittanceLUT compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    // transmittance LUT
    MTL::TextureDescriptor* transmittanceLUTDescriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormat::PixelFormatRGBA16Float, TRANS_WIDTH, TRANS_HEIGHT, false);
    transmittanceLUTDescriptor->setStorageMode(MTL::StorageModePrivate);
    transmittanceLUTDescriptor->setUsage(MTL::TextureUsageShaderWrite | MTL::TextureUsageShaderRead);
    m_transmittanceLUT = std::make_unique<Texture>(device, transmittanceLUTDescriptor);
    m_transmittanceLUT->GetNative()->setLabel(NS::String::string("Atmosphere TransmittanceLUT", NS::UTF8StringEncoding));

    const TransmittanceArgumentData transmittanceData
    {
        m_transmittanceLUT->GetNative()->gpuResourceID(),
        glm::uvec2 { m_transmittanceLUT->GetWidth(), m_transmittanceLUT->GetHeight() },
    };
    m_transmittanceParamsBuffer = std::make_unique<Buffer>(device, sizeof(TransmittanceArgumentData), MTL::ResourceStorageModeShared);
    m_transmittanceParamsBuffer->Update(&transmittanceData, sizeof(TransmittanceArgumentData));
    m_transmittanceParamsBuffer->GetNative()->setLabel(NS::String::string("AtmosphereTransmittanceLUT Argument Buffer", NS::UTF8StringEncoding));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("AtmosphereTransmittanceLUT", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(TransmittanceBufferIndex::MaxBufferBindCount));
    m_transmittanceArgumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_transmittanceArgumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_transmittanceArgumentTable->setAddress(m_transmittanceParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TransmittanceBufferIndex::KernelArguments));

    // TODO: SkyViewLUT Pass

    compiler->release();
}

void AtmosphereLUT::AddToGraph(RenderGraph &graph) {
    RenderGraphResourceHandle transmittanceLUTHandle = graph.RegisterTexture("AtmosphereTransmittanceLUT", *m_transmittanceLUT);

    graph.AddPass<PassData>(
        "AtmosphereLUT",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, PassData& data, RenderGraphResources&) {
            data.transmittancePiplineState           = m_transmittanceLUTPipelineState;
            data.transmittanceArgumentTable          = m_transmittanceArgumentTable;
            data.transmittanceArgumentBuffer         = m_transmittanceParamsBuffer->GetNative();
            data.transmittanceLUTHandle = transmittanceLUTHandle;

            builder.WriteTexture(transmittanceLUTHandle);
        },
        [this](const PassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            if (!hasInit) {
                MTL::Texture* transmittanceLUT = resources.GetTexture(data.transmittanceLUTHandle);

                cmd.AddResource(data.transmittanceArgumentBuffer);
                cmd.AddResource(transmittanceLUT);

                MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();
                computeEncoder->setComputePipelineState(data.transmittancePiplineState);
                computeEncoder->setArgumentTable(data.transmittanceArgumentTable);
                NS::UInteger width = data.transmittancePiplineState->threadExecutionWidth();
                NS::UInteger height = data.transmittancePiplineState->maxTotalThreadsPerThreadgroup() / width;
                MTL::Size gridSize = MTL::Size(TRANS_WIDTH, TRANS_HEIGHT, 1);
                MTL::Size threadGroupSize = MTL::Size(width, height, 1);
                computeEncoder->dispatchThreads(gridSize, threadGroupSize);
                computeEncoder->endEncoding();

                hasInit = true;
            }
            else {
                auto encoder = cmd.BeginComputePass(); // dummy encoder
                encoder->endEncoding();
            }
        }
    );
}
