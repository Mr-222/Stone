#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Pass.h"

class Buffer;
class MetalContext;

class AtmosphereScatteringPass final : Pass {
public:
    AtmosphereScatteringPass() = default;
    ~AtmosphereScatteringPass();

    void Setup(MetalContext& context);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MetalContext* m_context = nullptr;
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL::DepthStencilState* m_depthStencilState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;

    // Per frame slot: contents depend on per-slot resources and are rewritten every frame
    std::vector<std::unique_ptr<Buffer>> m_fragmentArgumentBuffers;
};
