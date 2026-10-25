#include "FFTBloomPass.h"

#include <algorithm>

#include <Core/Buffer.h>
#include <Core/MetalContext.h>
#include <Core/RenderGraph.h>
#include <Core/RenderGraphResources.h>
#include <Core/Texture.h>
#include <Shader/ShaderTypes.h>
#include <Utility/Logger.h>
#include <Utility/ShaderLibrary.h>

constexpr const char* kFFTBloomShaderLibrary = STONE_SHADER_DIR "/FFTBloom.metallib";
constexpr uint32_t kFFTSize = kFFTBloomSize;

struct FFTBloomPrefilterArgumentData {
    MTL::ResourceID sceneColorTexture;
    MTL::ResourceID outRealTexture;
    MTL::ResourceID outImagTexture;
    float threshold;
    float knee;
    float maxLuma;
    float padding = 0.0f;
};
static_assert(sizeof(FFTBloomPrefilterArgumentData) == 40);

struct FFTBloomPSFArgumentData {
    MTL::ResourceID outRealTexture;
    MTL::ResourceID outImagTexture;
    uint32_t diffractionSpikeCount;
    float diffractionIntensity;
    float pixelAspect;
    float padding = 0.0f;
};
static_assert(sizeof(FFTBloomPSFArgumentData) == 32);

struct FFTBloom1DArgumentData {
    MTL::ResourceID inRealTexture;
    MTL::ResourceID inImagTexture;
    MTL::ResourceID outRealTexture;
    MTL::ResourceID outImagTexture;
    uint32_t inverse;
    uint32_t padding = 0;
};
static_assert(sizeof(FFTBloom1DArgumentData) == 40);

struct FFTBloomSpectralArgumentData {
    MTL::ResourceID sceneRealTexture;
    MTL::ResourceID sceneImagTexture;
    MTL::ResourceID kernelRealTexture;
    MTL::ResourceID kernelImagTexture;
    MTL::ResourceID outRealTexture;
    MTL::ResourceID outImagTexture;
};
static_assert(sizeof(FFTBloomSpectralArgumentData) == 48);

struct FFTBloomPassData {
    RenderGraphResourceHandle sceneColorHandle;
    RenderGraphResourceHandle bloomHandle;
};

FFTBloomPass::~FFTBloomPass() {
    for (auto* table : m_argumentTables) {
        if (table) table->release();
    }
    if (m_prefilterPipelineState) m_prefilterPipelineState->release();
    if (m_psfGeneratePipelineState) m_psfGeneratePipelineState->release();
    if (m_fft1DHorizontalPipelineState) m_fft1DHorizontalPipelineState->release();
    if (m_fft1DVerticalPipelineState) m_fft1DVerticalPipelineState->release();
    if (m_spectralMultiplyPipelineState) m_spectralMultiplyPipelineState->release();
}

void FFTBloomPass::Setup(MetalContext& context) {
    m_context = &context;
    MTL::Device* device = context.GetDevice();
    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kFFTBloomShaderLibrary, {
        "fft_bloom_prefilter", "fft_psf_generate", "fft_1d_horizontal", "fft_1d_vertical", "fft_spectral_multiply"
    });

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "FFTBloom: Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();

    auto createComputePipeline = [&](const char* functionName, const char* label) -> MTL::ComputePipelineState* {
        MTL4::ComputePipelineDescriptor* desc = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
        desc->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
        desc->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), functionName));
        MTL::ComputePipelineState* pipeline = compiler->newComputePipelineState(desc, taskOptions, &error);
        LOG_ERROR_IF(!pipeline, "FFTBloom: Failed to create pipeline {}: {}", label, error ? error->localizedDescription()->utf8String() : "unknown error");
        return pipeline;
    };

    m_prefilterPipelineState = createComputePipeline("fft_bloom_prefilter", "FFTBloomPrefilter");
    m_psfGeneratePipelineState = createComputePipeline("fft_psf_generate", "FFTBloomPSFGenerate");
    m_fft1DHorizontalPipelineState = createComputePipeline("fft_1d_horizontal", "FFTBloom1DHorizontal");
    m_fft1DVerticalPipelineState = createComputePipeline("fft_1d_vertical", "FFTBloom1DVertical");
    m_spectralMultiplyPipelineState = createComputePipeline("fft_spectral_multiply", "FFTBloomSpectralMultiply");

    compiler->release();

    MTL::TextureDescriptor* desc32 = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, kFFTSize, kFFTSize, false);
    desc32->setStorageMode(MTL::StorageModePrivate);
    desc32->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    m_spatialRealTexture = std::make_unique<Texture>(device, desc32);
    m_spatialImagTexture = std::make_unique<Texture>(device, desc32);
    m_tempRealTexture = std::make_unique<Texture>(device, desc32);
    m_tempImagTexture = std::make_unique<Texture>(device, desc32);
    m_kernelRealTexture = std::make_unique<Texture>(device, desc32);
    m_kernelImagTexture = std::make_unique<Texture>(device, desc32);

    MTL::TextureDescriptor* desc16 = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kFFTSize, kFFTSize, false);
    desc16->setStorageMode(MTL::StorageModePrivate);
    desc16->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    const uint32_t frameSlotCount = context.GetFrameSlotCount();
    m_bloomOutputTextures.resize(frameSlotCount);
    for (uint32_t i = 0; i < frameSlotCount; ++i) {
        m_bloomOutputTextures[i] = std::make_unique<Texture>(device, desc16);
        m_bloomOutputTextures[i]->GetNative()->setLabel(NS::String::string("Bloom Output", NS::UTF8StringEncoding));
    }

    m_argumentBuffers.resize(frameSlotCount);
    for (uint32_t i = 0; i < frameSlotCount; ++i) {
        m_argumentBuffers[i] = std::make_unique<Buffer>(device, 1024, MTL::ResourceStorageModeShared);
    }

    MTL4::ArgumentTableDescriptor* tableDesc = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    tableDesc->setLabel(NS::String::string("FFT Bloom Argument Table", NS::UTF8StringEncoding));
    tableDesc->setInitializeBindings(true);
    tableDesc->setMaxBufferBindCount(static_cast<NS::UInteger>(FFTBloomBufferIndex::MaxBufferBindCount));

    for (size_t i = 0; i < static_cast<size_t>(FFTBloomStep::Count); ++i) {
        m_argumentTables[i] = device->newArgumentTable(tableDesc, &error);
        LOG_ERROR_IF(!m_argumentTables[i], "FFTBloom: Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");
    }
}

void FFTBloomPass::AddToGraph(RenderGraph& graph) {
    const uint32_t frameSlotCount = m_context->GetFrameSlotCount();
    for (uint32_t frameSlot = 0; frameSlot < frameSlotCount; ++frameSlot) {
        graph.RegisterFrameLocalTexture(kBloomImageName, frameSlot, *m_bloomOutputTextures[frameSlot]);
    }

    RenderGraphResourceHandle sceneColorHandle = graph.DeclareTexture(kSceneColorImageName);
    RenderGraphResourceHandle bloomHandle = graph.DeclareTexture(kBloomImageName);

    graph.AddPass<FFTBloomPassData>(
        "FFTBloom",
        IsCompute,
        [=](RenderGraphBuilder& builder, FFTBloomPassData& data, RenderGraphResources&) {
            data.sceneColorHandle = sceneColorHandle;
            data.bloomHandle = bloomHandle;
            builder.ReadTexture(sceneColorHandle);
            builder.WriteTexture(bloomHandle);
        },
        [this](const FFTBloomPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            const uint32_t frameSlot = m_context->GetCurrentFrameSlot();
            MTL::Texture* sceneColorTexture = resources.GetTexture(data.sceneColorHandle);
            MTL::Texture* bloomTexture = resources.GetTexture(data.bloomHandle);

            LOG_ERROR_IF(!sceneColorTexture, "FFTBloom: No scene color texture.");
            LOG_ERROR_IF(!bloomTexture, "FFTBloom: No bloom output texture.");

            uint8_t* bufferPtr = static_cast<uint8_t*>(m_argumentBuffers[frameSlot]->GetNative()->contents());
            uint64_t baseGpuAddr = m_argumentBuffers[frameSlot]->GetGPUAddress();

            // Track resources
            cmd.AddResource(sceneColorTexture);
            cmd.AddResource(bloomTexture);
            cmd.AddResource(m_spatialRealTexture->GetNative());
            cmd.AddResource(m_spatialImagTexture->GetNative());
            cmd.AddResource(m_tempRealTexture->GetNative());
            cmd.AddResource(m_tempImagTexture->GetNative());
            cmd.AddResource(m_kernelRealTexture->GetNative());
            cmd.AddResource(m_kernelImagTexture->GetNative());
            cmd.AddResource(m_argumentBuffers[frameSlot]->GetNative());

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();

            // 1. Optical PSF generation and forward 2D FFT (regenerated when the screen aspect changes)
            const float pixelAspect = static_cast<float>(sceneColorTexture->width()) / static_cast<float>(std::max<NS::UInteger>(sceneColorTexture->height(), 1));
            if (!m_kernelInitialized || std::abs(pixelAspect - m_kernelPixelAspect) > 1e-3f) {
                const FFTBloomPSFArgumentData psfArgs {
                    .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                    .diffractionSpikeCount = 6,
                    .diffractionIntensity = 0.08f,
                    .pixelAspect = pixelAspect,
                };
                memcpy(bufferPtr + 0, &psfArgs, sizeof(FFTBloomPSFArgumentData));
                auto* psfTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::PSFGenerate)];
                psfTable->setAddress(baseGpuAddr + 0, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
                computeEncoder->setComputePipelineState(m_psfGeneratePipelineState);
                computeEncoder->setArgumentTable(psfTable);
                computeEncoder->dispatchThreads(MTL::Size(kFFTSize, kFFTSize, 1), MTL::Size(16, 16, 1));
                computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

                // PSF Horizontal FFT: temp -> spatial
                const FFTBloom1DArgumentData psfHArgs {
                    .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                    .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                    .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                    .inverse = 0,
                    .padding = 0,
                };
                memcpy(bufferPtr + 64, &psfHArgs, sizeof(FFTBloom1DArgumentData));
                auto* psfHTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::PSFFFTHorizontal)];
                psfHTable->setAddress(baseGpuAddr + 64, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
                computeEncoder->setComputePipelineState(m_fft1DHorizontalPipelineState);
                computeEncoder->setArgumentTable(psfHTable);
                computeEncoder->dispatchThreads(MTL::Size(256, kFFTSize, 1), MTL::Size(256, 1, 1));
                computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

                // PSF Vertical FFT: spatial -> kernel
                const FFTBloom1DArgumentData psfVArgs {
                    .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                    .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                    .outRealTexture = m_kernelRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_kernelImagTexture->GetNative()->gpuResourceID(),
                    .inverse = 0,
                    .padding = 0,
                };
                memcpy(bufferPtr + 128, &psfVArgs, sizeof(FFTBloom1DArgumentData));
                auto* psfVTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::PSFFFTVertical)];
                psfVTable->setAddress(baseGpuAddr + 128, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
                computeEncoder->setComputePipelineState(m_fft1DVerticalPipelineState);
                computeEncoder->setArgumentTable(psfVTable);
                computeEncoder->dispatchThreads(MTL::Size(kFFTSize, 256, 1), MTL::Size(1, 256, 1));
                computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

                m_kernelInitialized = true;
                m_kernelPixelAspect = pixelAspect;
            }

            // 2. Prefilter & Thresholding: sceneColor -> spatial
            const FFTBloomPrefilterArgumentData prefilterArgs {
                .sceneColorTexture = sceneColorTexture->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .threshold = 1.0f,
                .knee = 0.4f,
                .maxLuma = 280.0f,
            };
            memcpy(bufferPtr + 192, &prefilterArgs, sizeof(FFTBloomPrefilterArgumentData));
            auto* prefilterTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::Prefilter)];
            prefilterTable->setAddress(baseGpuAddr + 192, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_prefilterPipelineState);
            computeEncoder->setArgumentTable(prefilterTable);
            computeEncoder->dispatchThreads(MTL::Size(kFFTSize, kFFTSize, 1), MTL::Size(16, 16, 1));
            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            // 3. Scene Forward Horizontal FFT: spatial -> temp
            const FFTBloom1DArgumentData sceneHArgs {
                .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .inverse = 0,
                .padding = 0,
            };
            memcpy(bufferPtr + 256, &sceneHArgs, sizeof(FFTBloom1DArgumentData));
            auto* sceneHTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::SceneFFTHorizontal)];
            sceneHTable->setAddress(baseGpuAddr + 256, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_fft1DHorizontalPipelineState);
            computeEncoder->setArgumentTable(sceneHTable);
            computeEncoder->dispatchThreads(MTL::Size(256, kFFTSize, 1), MTL::Size(256, 1, 1));
            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            // 4. Scene Forward Vertical FFT: temp -> spatial
            const FFTBloom1DArgumentData sceneVArgs {
                .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .inverse = 0,
                .padding = 0,
            };
            memcpy(bufferPtr + 320, &sceneVArgs, sizeof(FFTBloom1DArgumentData));
            auto* sceneVTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::SceneFFTVertical)];
            sceneVTable->setAddress(baseGpuAddr + 320, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_fft1DVerticalPipelineState);
            computeEncoder->setArgumentTable(sceneVTable);
            computeEncoder->dispatchThreads(MTL::Size(kFFTSize, 256, 1), MTL::Size(1, 256, 1));
            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            // 5. Spectral Multiply: spatial * kernel -> temp
            const FFTBloomSpectralArgumentData spectralArgs {
                .sceneRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .sceneImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .kernelRealTexture = m_kernelRealTexture->GetNative()->gpuResourceID(),
                .kernelImagTexture = m_kernelImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
            };
            memcpy(bufferPtr + 384, &spectralArgs, sizeof(FFTBloomSpectralArgumentData));
            auto* spectralTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::SpectralMultiply)];
            spectralTable->setAddress(baseGpuAddr + 384, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_spectralMultiplyPipelineState);
            computeEncoder->setArgumentTable(spectralTable);
            computeEncoder->dispatchThreads(MTL::Size(kFFTSize, kFFTSize, 1), MTL::Size(16, 16, 1));
            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            // 6. Inverse Horizontal FFT: temp -> spatial
            const FFTBloom1DArgumentData ifftHArgs {
                .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .inverse = 1,
                .padding = 0,
            };
            memcpy(bufferPtr + 480, &ifftHArgs, sizeof(FFTBloom1DArgumentData));
            auto* ifftHTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::SceneIFFTHorizontal)];
            ifftHTable->setAddress(baseGpuAddr + 480, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_fft1DHorizontalPipelineState);
            computeEncoder->setArgumentTable(ifftHTable);
            computeEncoder->dispatchThreads(MTL::Size(256, kFFTSize, 1), MTL::Size(256, 1, 1));
            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            // 7. Inverse Vertical FFT: spatial -> bloomTexture (output)
            const FFTBloom1DArgumentData ifftVArgs {
                .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = bloomTexture->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .inverse = 1,
                .padding = 0,
            };
            memcpy(bufferPtr + 544, &ifftVArgs, sizeof(FFTBloom1DArgumentData));
            auto* ifftVTable = m_argumentTables[static_cast<size_t>(FFTBloomStep::SceneIFFTVertical)];
            ifftVTable->setAddress(baseGpuAddr + 544, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
            computeEncoder->setComputePipelineState(m_fft1DVerticalPipelineState);
            computeEncoder->setArgumentTable(ifftVTable);
            computeEncoder->dispatchThreads(MTL::Size(kFFTSize, 256, 1), MTL::Size(1, 256, 1));

            computeEncoder->endEncoding();
        }
    );
}
