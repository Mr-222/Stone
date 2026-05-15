#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <Metal/Metal.hpp>

#include "CommandBuffer.h"
#include "CommandBufferPool.h"
#include "MetalContext.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphResources.h"
#include "RenderPassNode.h"

class RenderGraph {
public:
    RenderGraph(std::shared_ptr<MetalContext> metalContext, std::shared_ptr<CommandBufferPool> commandBufferPool);

    RenderGraphTextureHandle DeclareTexture(const std::string& name);
    RenderGraphTextureHandle RegisterTexture(const std::string& name, const Texture& texture);

    template<typename PassData>
    void AddPass(
        const std::string& name,
        std::function<void(RenderGraphBuilder&, PassData&)> setupFn,
        std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn
    );

    void Compile();

    void Execute(MTL4::CommandQueue*);

private:
    // TODO: add dependency map
    std::vector<std::unique_ptr<RenderPassNodeBase>> m_passes;
    RenderGraphResources m_resources;
    std::shared_ptr<MetalContext> m_metalContext;
    std::shared_ptr<CommandBufferPool> m_commandBufferPool;
};

template<typename PassData>
void RenderGraph::AddPass(
    const std::string& name,
    std::function<void(RenderGraphBuilder&, PassData&)> setupFn,
    std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn)
{
    // TODO: Add name as key to dependency map

    PassData data;
    RenderGraphBuilder builder;
    setupFn(builder, data);

    auto node = std::make_unique<RenderPassNode<PassData>>(
        name,
        builder.GetTextureAccesses(),
        std::move(m_commandBufferPool->Acquire()),
        data,
        std::move(executeFn));
    m_passes.emplace_back(std::move(node));
}
