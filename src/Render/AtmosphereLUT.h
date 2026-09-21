#pragma once

#include <memory>
#include <Metal/Metal.hpp>

#include "Pass.h"

class Buffer;
class Texture;
class MetalContext;

class AtmosphereLUT final : Pass {
public:
    AtmosphereLUT(): hasInit(false) {};
    ~AtmosphereLUT() = default;

    void Setup(MetalContext& context);
    void AddToGraph(RenderGraph& graph) override;

    static constexpr bool IsCompute = true;

private:
    MTL::ComputePipelineState* m_transmittanceLUTPipelineState = nullptr;
    MTL4::ArgumentTable* m_transmittanceArgumentTable = nullptr;
    std::unique_ptr<Buffer> m_transmittanceParamsBuffer;
    std::unique_ptr<Texture> m_transmittanceLUT;

    MTL::ComputePipelineState* m_skyViewLUTPipelineState = nullptr;

    bool hasInit;
};