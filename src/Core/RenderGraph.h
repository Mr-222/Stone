#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Metal/Metal.hpp>

#include "CommandBuffer.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphResources.h"
#include "RenderPassNode.h"

class RenderGraph {
public:
    RenderGraph(CommandBuffer&& cmd);

    template<typename PassData>
    void AddPass(
        const std::string& name,
        std::function<void(RenderGraphBuilder&, PassData&)> setupFn,
        std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn
    );

    void Compile();

    void Execute(MTL4::CommandQueue*);

private:
    CommandBuffer m_cmd;
    // TODO: add dependency map
    std::vector<std::unique_ptr<RenderPassNodeBase>> m_passes;
    RenderGraphBuilder m_builder;
    RenderGraphResources m_resources;
};

template<typename PassData>
void RenderGraph::AddPass(
    const std::string& name,
    std::function<void(RenderGraphBuilder&, PassData&)> setupFn,
    std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn)
{
    // TODO: Add name as key to dependency map

    PassData data;
    setupFn(m_builder, data);

    auto node = std::make_unique<RenderPassNode<PassData>>(data, std::move(executeFn));
    m_passes.emplace_back(std::move(node));
}
