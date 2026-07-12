#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
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

    RenderGraphResourceHandle DeclareTexture(const std::string& name);
    RenderGraphResourceHandle RegisterTexture(const std::string& name, const Texture& texture);

    RenderGraphResourceHandle DeclareBuffer(const std::string& name);
    RenderGraphResourceHandle RegisterBuffer(const std::string& name, const Buffer& buffer);

    RenderGraphResourceHandle DeclareIndirectCommandBuffer(const std::string& name);
    RenderGraphResourceHandle RegisterIndirectCommandBuffer(const std::string& name, const IndirectCommandBuffer& indirectCB);

    template<typename PassData>
    void AddPass(
        const std::string& name,
        std::function<void(RenderGraphBuilder&, PassData&, RenderGraphResources&)> setupFn,
        std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn
    );

    template<typename PassType>
    void AddPassNode(const std::string& name, auto&&... args);

    void Compile();

    void Execute(MTL4::CommandQueue*);

private:
    struct PassHolderBase {
        virtual ~PassHolderBase() = default;
    };

    template<typename T>
    struct PassHolder : PassHolderBase {
        std::unique_ptr<T> pass;
        PassHolder(std::unique_ptr<T> p) : pass(std::move(p)) {}
    };

    std::vector<std::unique_ptr<PassHolderBase>> m_passObjects;
    // TODO: add dependency map
    std::unordered_map<std::string, std::unique_ptr<RenderPassNodeBase>> m_passes;
    RenderGraphResources m_resources;
    std::shared_ptr<MetalContext> m_metalContext;
    std::shared_ptr<CommandBufferPool> m_commandBufferPool;
};

template<typename PassData>
void RenderGraph::AddPass(
    const std::string& name,
    std::function<void(RenderGraphBuilder&, PassData&, RenderGraphResources&)> setupFn,
    std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)> executeFn)
{
    // TODO: Add name as key to dependency map

    PassData data;
    RenderGraphBuilder builder;
    setupFn(builder, data, m_resources);

    auto node = std::make_unique<RenderPassNode<PassData>>(
        name,
        builder.GetResourceAccesses(),
        data,
        std::move(executeFn));
    assert(!m_passes.contains(name));
    m_passes[name] = std::move(node);
}

template<typename PassType>
void RenderGraph::AddPassNode(const std::string& name, auto&&... args) {
    auto pass = std::make_unique<PassType>();
    pass->Setup(std::forward<decltype(args)>(args)...);
    pass->AddToGraph(*this);
    
    // Type-erase the unique_ptr and store it to keep the pass alive
    m_passObjects.push_back(std::make_unique<PassHolder<PassType>>(std::move(pass)));
}