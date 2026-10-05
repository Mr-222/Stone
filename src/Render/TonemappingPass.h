#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Pass.h"

class Buffer;
class MetalContext;
class Window;

class TonemappingPass final : Pass {
public:
    TonemappingPass() = default;
    ~TonemappingPass();

    void Setup(MetalContext& context, const Window& window);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MetalContext* m_context = nullptr;
    const Window* m_window = nullptr;
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;
    std::vector<std::unique_ptr<Buffer>> m_fragmentArgumentBuffers;
};
