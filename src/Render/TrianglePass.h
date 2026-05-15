#pragma once

#include <memory>
#include <Metal/Metal.hpp>

#include "Core/RenderGraphResources.h"

class Buffer;
class CommandBufferPool;
class Heap;
class MetalContext;
class RenderGraph;

class TrianglePass {
public:
    TrianglePass();
    ~TrianglePass();

    void Setup(MetalContext& context, CommandBufferPool& commandBufferPool);
    void AddToGraph(RenderGraph& graph);

private:
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;
    MTL::ResidencySet* m_residencySet = nullptr;

    std::unique_ptr<Heap> m_privateHeap;
    std::unique_ptr<Heap> m_sharedHeap;
    std::unique_ptr<Buffer> m_argumentBuffer;
    std::unique_ptr<Buffer> m_positionBuffer;
    std::unique_ptr<Buffer> m_colorBuffer;
};
