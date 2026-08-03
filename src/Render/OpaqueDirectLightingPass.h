#pragma once

#include <memory>
#include <Metal/Metal.hpp>

#include "Mesh.h"
#include "Pass.h"

class Buffer;
class MetalContext;

class OpaqueDirectLightingPass final : Pass {
public:
    OpaqueDirectLightingPass();
    ~OpaqueDirectLightingPass();

    void Setup(MetalContext& context, int numPrimitives);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = false;

private:
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL4::ArgumentTable* m_argumentTable = nullptr;
    MTL::ArgumentEncoder* m_argumentEncoder = nullptr;

    std::unique_ptr<Buffer> m_argumentBuffer;

    int m_numPrimitives;
};
