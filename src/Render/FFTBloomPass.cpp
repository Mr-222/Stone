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
constexpr uint32_t kFFTHalfSize = kFFTBloomSize / 2;
constexpr size_t kArgumentStride = 64;
constexpr size_t kArgumentBufferSize = static_cast<size_t>(FFTBloomStep::Count) * kArgumentStride;

struct FFTBloomDownsampleArgumentData {
    MTL::ResourceID sceneColorTexture;
    MTL::ResourceID outTexture;
    float threshold;
    float knee;
    float maxLuma;
    float padding = 0.0f;
};
static_assert(sizeof(FFTBloomDownsampleArgumentData) == 32);

struct FFTBloomPrefilterArgumentData {
    MTL::ResourceID inTexture;
    MTL::ResourceID outRealTexture;
    MTL::ResourceID outImagTexture;
};
static_assert(sizeof(FFTBloomPrefilterArgumentData) == 24);

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
    if (m_downsamplePipelineState) m_downsamplePipelineState->release();
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
        "fft_bloom_downsample", "fft_bloom_prefilter", "fft_psf_generate", "fft_1d_horizontal", "fft_1d_vertical", "fft_spectral_multiply"
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

    m_downsamplePipelineState = createComputePipeline("fft_bloom_downsample", "FFTBloomDownsample");
    m_prefilterPipelineState = createComputePipeline("fft_bloom_prefilter", "FFTBloomPrefilter");
    m_psfGeneratePipelineState = createComputePipeline("fft_psf_generate", "FFTBloomPSFGenerate");
    m_fft1DHorizontalPipelineState = createComputePipeline("fft_1d_horizontal", "FFTBloom1DHorizontal");
    m_fft1DVerticalPipelineState = createComputePipeline("fft_1d_vertical", "FFTBloom1DVertical");
    m_spectralMultiplyPipelineState = createComputePipeline("fft_spectral_multiply", "FFTBloomSpectralMultiply");

    compiler->release();

    MTL::TextureDescriptor* desc32 = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, kFFTBloomSize, kFFTBloomSize, false);
    desc32->setStorageMode(MTL::StorageModePrivate);
    desc32->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    m_spatialRealTexture = std::make_unique<Texture>(device, desc32);
    m_spatialImagTexture = std::make_unique<Texture>(device, desc32);
    m_tempRealTexture = std::make_unique<Texture>(device, desc32);
    m_tempImagTexture = std::make_unique<Texture>(device, desc32);
    m_kernelRealTexture = std::make_unique<Texture>(device, desc32);
    m_kernelImagTexture = std::make_unique<Texture>(device, desc32);

    MTL::TextureDescriptor* desc16 = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, kFFTBloomSize, kFFTBloomSize, false);
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
        m_argumentBuffers[i] = std::make_unique<Buffer>(device, kArgumentBufferSize, MTL::ResourceStorageModeShared);
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

            auto dispatchStep = [&]<typename Args>(FFTBloomStep step, MTL::ComputePipelineState* pipeline, const Args& args, MTL::Size gridSize, MTL::Size groupSize, bool barrier = true) {
                static_assert(sizeof(Args) <= kArgumentStride);
                const size_t stepIdx = static_cast<size_t>(step);
                const size_t offset = stepIdx * kArgumentStride;
                memcpy(bufferPtr + offset, &args, sizeof(Args));
                MTL4::ArgumentTable* table = m_argumentTables[stepIdx];
                table->setAddress(baseGpuAddr + offset, static_cast<NS::UInteger>(FFTBloomBufferIndex::KernelArguments));
                computeEncoder->setComputePipelineState(pipeline);
                computeEncoder->setArgumentTable(table);
                computeEncoder->dispatchThreads(gridSize, groupSize);
                if (barrier)
                    computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
            };

            const MTL::Size grid2D(kFFTBloomSize, kFFTBloomSize, 1);
            const MTL::Size group2D(16, 16, 1);
            const MTL::Size gridH(kFFTHalfSize, kFFTBloomSize, 1);
            const MTL::Size groupH(kFFTHalfSize, 1, 1);
            const MTL::Size gridV(kFFTBloomSize, kFFTHalfSize, 1);
            const MTL::Size groupV(1, kFFTHalfSize, 1);

            // 1. Optical PSF generation and forward 2D FFT (regenerated when the screen aspect changes)
            const float pixelAspect = static_cast<float>(sceneColorTexture->width()) / static_cast<float>(std::max<NS::UInteger>(sceneColorTexture->height(), 1));
            if (!m_kernelInitialized || std::abs(pixelAspect - m_kernelPixelAspect) > 1e-3f) {
                dispatchStep(FFTBloomStep::PSFGenerate, m_psfGeneratePipelineState, FFTBloomPSFArgumentData{
                    .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                    .diffractionSpikeCount = 6,
                    .diffractionIntensity = 0.08f,
                    .pixelAspect = pixelAspect,
                }, grid2D, group2D);

                // PSF Horizontal FFT: temp -> spatial
                dispatchStep(FFTBloomStep::PSFFFTHorizontal, m_fft1DHorizontalPipelineState, FFTBloom1DArgumentData{
                    .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                    .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                    .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                    .inverse = 0,
                }, gridH, groupH);

                // PSF Vertical FFT: spatial -> kernel
                dispatchStep(FFTBloomStep::PSFFFTVertical, m_fft1DVerticalPipelineState, FFTBloom1DArgumentData{
                    .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                    .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                    .outRealTexture = m_kernelRealTexture->GetNative()->gpuResourceID(),
                    .outImagTexture = m_kernelImagTexture->GetNative()->gpuResourceID(),
                    .inverse = 0,
                }, gridV, groupV);

                m_kernelInitialized = true;
                m_kernelPixelAspect = pixelAspect;
            }

            // 2. Bright-pass downsample & tent prefilter: sceneColor -> temp -> spatial
            dispatchStep(FFTBloomStep::PrefilterDownsample, m_downsamplePipelineState, FFTBloomDownsampleArgumentData{
                .sceneColorTexture = sceneColorTexture->gpuResourceID(),
                .outTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .threshold = 1.0f,
                .knee = 0.4f,
                .maxLuma = 280.0f,
            }, grid2D, group2D);

            dispatchStep(FFTBloomStep::Prefilter, m_prefilterPipelineState, FFTBloomPrefilterArgumentData{
                .inTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
            }, grid2D, group2D);

            // 3. Scene Forward Horizontal FFT: spatial -> temp
            dispatchStep(FFTBloomStep::SceneFFTHorizontal, m_fft1DHorizontalPipelineState, FFTBloom1DArgumentData{
                .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .inverse = 0,
            }, gridH, groupH);

            // 4. Scene Forward Vertical FFT: temp -> spatial
            dispatchStep(FFTBloomStep::SceneFFTVertical, m_fft1DVerticalPipelineState, FFTBloom1DArgumentData{
                .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .inverse = 0,
            }, gridV, groupV);

            // 5. Spectral Multiply: spatial * kernel -> temp
            dispatchStep(FFTBloomStep::SpectralMultiply, m_spectralMultiplyPipelineState, FFTBloomSpectralArgumentData{
                .sceneRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .sceneImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .kernelRealTexture = m_kernelRealTexture->GetNative()->gpuResourceID(),
                .kernelImagTexture = m_kernelImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
            }, grid2D, group2D);

            // 6. Inverse Horizontal FFT: temp -> spatial
            dispatchStep(FFTBloomStep::SceneIFFTHorizontal, m_fft1DHorizontalPipelineState, FFTBloom1DArgumentData{
                .inRealTexture = m_tempRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .outImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .inverse = 1,
            }, gridH, groupH);

            // 7. Inverse Vertical FFT: spatial -> bloomTexture (output)
            dispatchStep(FFTBloomStep::SceneIFFTVertical, m_fft1DVerticalPipelineState, FFTBloom1DArgumentData{
                .inRealTexture = m_spatialRealTexture->GetNative()->gpuResourceID(),
                .inImagTexture = m_spatialImagTexture->GetNative()->gpuResourceID(),
                .outRealTexture = bloomTexture->gpuResourceID(),
                .outImagTexture = m_tempImagTexture->GetNative()->gpuResourceID(),
                .inverse = 1,
            }, gridV, groupV, false);

            computeEncoder->endEncoding();
        }
    );
}
