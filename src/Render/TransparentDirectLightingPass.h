#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Mesh.h"
#include "Pass.h"

class Buffer;
class MetalContext;
class Texture;

class TransparentDirectLightingPass final : Pass {
public:
    TransparentDirectLightingPass() = default;
    ~TransparentDirectLightingPass();

    void Setup(MetalContext& context, int numPrimitives, const std::vector<Texture>& textures);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL::DepthStencilState* m_depthStencilState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;

    std::unique_ptr<Buffer> m_vertexArgumentBuffer;
    std::unique_ptr<Buffer> m_fragmentArgumentBuffer;
    std::vector<MTL::Texture*> m_textures;
    std::vector<std::unique_ptr<Texture>> m_oitAccumTextures;
    std::vector<std::unique_ptr<Texture>> m_oitRevealTextures;

    int m_numPrimitives;
};
