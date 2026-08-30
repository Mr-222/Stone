#pragma once

#include <memory>
#include <Metal/Metal.hpp>

#include "Pass.h"

class Buffer;
class MetalContext;

class TransparentCompositePass final : Pass {
public:
    TransparentCompositePass() = default;
    ~TransparentCompositePass();

    void Setup(MetalContext& context);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;
    std::unique_ptr<Buffer> m_fragmentArgumentBuffer;
};
