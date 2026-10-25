#pragma once

#include <array>
#include <memory>
#include <vector>
#include <Metal/Metal.hpp>
#include "Pass.h"

class Buffer;
class MetalContext;
class Texture;

enum class FFTBloomStep : uint32_t {
    Prefilter = 0,
    SceneFFTHorizontal,
    SceneFFTVertical,
    SpectralMultiply,
    SceneIFFTHorizontal,
    SceneIFFTVertical,
    PSFGenerate,
    PSFFFTHorizontal,
    PSFFFTVertical,
    Count
};

class FFTBloomPass final : Pass {
public:
    FFTBloomPass() = default;
    ~FFTBloomPass();

    void Setup(MetalContext& context);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = true;

private:
    MetalContext* m_context = nullptr;

    MTL::ComputePipelineState* m_prefilterPipelineState = nullptr;
    MTL::ComputePipelineState* m_psfGeneratePipelineState = nullptr;
    MTL::ComputePipelineState* m_fft1DHorizontalPipelineState = nullptr;
    MTL::ComputePipelineState* m_fft1DVerticalPipelineState = nullptr;
    MTL::ComputePipelineState* m_spectralMultiplyPipelineState = nullptr;

    std::array<MTL4::ArgumentTable*, static_cast<size_t>(FFTBloomStep::Count)> m_argumentTables;
    std::vector<std::unique_ptr<Buffer>> m_argumentBuffers;

    std::unique_ptr<Texture> m_spatialRealTexture;
    std::unique_ptr<Texture> m_spatialImagTexture;
    std::unique_ptr<Texture> m_tempRealTexture;
    std::unique_ptr<Texture> m_tempImagTexture;
    std::unique_ptr<Texture> m_kernelRealTexture;
    std::unique_ptr<Texture> m_kernelImagTexture;
    std::vector<std::unique_ptr<Texture>> m_bloomOutputTextures;

    bool m_kernelInitialized = false;
    float m_kernelPixelAspect = 0.0f;
};
