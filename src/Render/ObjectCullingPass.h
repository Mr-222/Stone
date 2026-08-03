#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Mesh.h"
#include "Pass.h"

class Buffer;
class CommandBufferPool;
class Heap;
class MetalContext;
class RenderGraph;
class IndirectCommandBuffer;

class ObjectCullingPass final : Pass {
public:
    ObjectCullingPass();
    ~ObjectCullingPass();

    void Setup(MetalContext& context, int numPrimitives);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = true;

private:
    MTL::ComputePipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;
    std::unique_ptr<IndirectCommandBuffer> m_indirectCB;

    std::unique_ptr<Buffer> m_visibilityBuffer;
    std::unique_ptr<Buffer> m_cullingParamsBuffer;
    std::unique_ptr<Buffer> m_ICBExecutionRangeBuffer;
    std::unique_ptr<Buffer> m_icbArgumentBuffer;

    int m_numPrimitives;
};
