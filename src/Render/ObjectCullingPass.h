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
    struct FrameResources {
        std::unique_ptr<IndirectCommandBuffer> indirectCB;
        std::unique_ptr<Buffer> visibilityBuffer;
        std::unique_ptr<Buffer> executionRangeBuffer;
        std::unique_ptr<Buffer> icbArgumentBuffer;
    };

    MTL::ComputePipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;

    std::unique_ptr<Buffer> m_cullingParamsBuffer;
    std::vector<FrameResources> m_frameResources;

    int m_numPrimitives;
};
