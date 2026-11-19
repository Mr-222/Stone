#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Mesh.h"
#include "Pass.h"

class Buffer;
class MetalContext;
class Texture;

class OpaqueDirectLightingPass final : Pass {
public:
    OpaqueDirectLightingPass() = default;
    ~OpaqueDirectLightingPass();

    void Setup(MetalContext& context, int numPrimitives, int maskedPrimitiveStart, const std::vector<Texture>& textures, MTL::AccelerationStructure* tlas, MTL::ResidencySet* accelResidencySet);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL::RenderPipelineState* m_maskedPipelineState = nullptr;
    MTL::RenderPipelineState* m_tileCullingPipelineState = nullptr;
    MTL::RenderPipelineState* m_deferredLightingPipelineState = nullptr;
    MTL::DepthStencilState* m_depthStencilState = nullptr;
    MTL::DepthStencilState* m_deferredLightingDepthStencilState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;

    std::unique_ptr<Buffer> m_vertexArgumentBuffer;
    std::unique_ptr<Buffer> m_fragmentArgumentBuffer;
    std::vector<MTL::Texture*> m_textures;
    std::vector<std::unique_ptr<Texture>> m_gbufferAlbedoMetallicTextures;
    std::vector<std::unique_ptr<Texture>> m_gbufferNormalTextures;
    std::vector<std::unique_ptr<Texture>> m_gbufferDepthRoughnessTextures;
    std::vector<std::unique_ptr<Buffer>> m_transparentTileLightListBuffers; // written here, read by TransparentDirectLightingPass
    MTL::AccelerationStructure* m_tlas = nullptr;
    MTL::ResidencySet* m_accelResidencySet = nullptr;

    int m_numPrimitives = 0;
    int m_maskedPrimitiveStart = 0;
};
