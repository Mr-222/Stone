#include "RenderGraph.h"

#include <queue>

#include "Utility/Logger.h"

RenderGraph::RenderGraph(std::shared_ptr<MetalContext> metalContext, std::shared_ptr<CommandBufferPool> commandBufferPool)
    : m_metalContext(std::move(metalContext)), m_commandBufferPool(std::move(commandBufferPool))
    {}

RenderGraphResourceHandle RenderGraph::DeclareTexture(const std::string& name) {
    return m_resources.DeclareTexture(name);
}

RenderGraphResourceHandle RenderGraph::RegisterTexture(const std::string& name, const Texture& texture) {
    return m_resources.RegisterTexture(name, texture);
}

RenderGraphResourceHandle RenderGraph::DeclareBuffer(const std::string &name) {
    return m_resources.DeclareBuffer(name);
}

RenderGraphResourceHandle RenderGraph::RegisterBuffer(const std::string &name, const Buffer& buffer) {
    return m_resources.RegisterBuffer(name, buffer);
}

RenderGraphResourceHandle RenderGraph::DeclareIndirectCommandBuffer(const std::string& name) {
    return m_resources.DeclareIndirectCommandBuffer(name);
}

RenderGraphResourceHandle RenderGraph::RegisterIndirectCommandBuffer(const std::string& name, const IndirectCommandBuffer& indirectCB) {
    return m_resources.RegisterIndirectCommandBuffer(name, indirectCB);
}

void RenderGraph::SetDependencyGraph(const std::unordered_map<std::string, std::vector<std::string>>& dependencies) {
    m_dependencyGraph = dependencies;
}

void RenderGraph::Compile() {
    for (const auto& [name, pass] : m_passes) {
        for (const RenderGraphResourceAccess& access : pass->GetResourceAccesses()) {
            switch (access.resourceType) {
            case RenderGraphResourceType::Texture:
                LOG_ERROR_IF(
                    !m_resources.IsTextureHandleDeclared(access.resource),
                    "Render pass '{}' declared an invalid texture dependency.",
                    pass->GetName());
                break;
            case RenderGraphResourceType::Buffer:
                LOG_ERROR_IF(
                    !m_resources.IsBufferHandleDeclared(access.resource),
                    "Render pass '{} declared an invalid buffer dependency.'",
                    pass->GetName());
                break;
            case RenderGraphResourceType::IndirectCommandBuffer:
                LOG_ERROR_IF(
                    !m_resources.IsIndirectCommandBufferHandleDeclared(access.resource),
                    "Render pass '{}' declared an invalid indirect command buffer dependency.",
                    pass->GetName());
                break;
            }
        }

        if (auto it = m_dependencyGraph.find(name); it != m_dependencyGraph.end()) {
            for (const std::string& dependencyName : it->second) {
                MTL::Event* event = m_passes.at(dependencyName)->Event;
                pass->WaitEventList.emplace_back(event);
            }
        }
    }

    // Topological sorting to determine execution order
    std::unordered_map<std::string, std::vector<std::string>> adjacency; // dependency -> dependents
    std::unordered_map<std::string, size_t> inDegree;

    for (const auto& [name, pass] : m_passes) {
        inDegree[name] = 0;
    }

    for (const auto& [name, dependencies] : m_dependencyGraph) {
        if (!m_passes.contains(name)) continue;
        inDegree[name] += dependencies.size();
        for (const std::string& dep : dependencies) {
            adjacency[dep].push_back(name);
        }
    }

    std::queue<std::string> q;
    for (const auto& [name, degree] : inDegree) {
        if (degree == 0) {
            q.push(name);
        }
    }

    m_executionOrder.clear();
    while (!q.empty()) {
        std::string current = q.front();
        q.pop();
        m_executionOrder.push_back(current);

        for (const std::string& dependent : adjacency[current]) {
            if (--inDegree[dependent] == 0) {
                q.push(dependent);
            }
        }
    }

    LOG_ERROR_IF(m_executionOrder.size() != m_passes.size(),
        "Render graph has a cycle! Only {} of {} passes were sorted.",
        m_executionOrder.size(), m_passes.size());
}

void RenderGraph::Execute(MTL4::CommandQueue* queue) {
    LOG_ERROR_IF(!queue, "RenderGraph execute requires a valid command queue.");

    const uint64_t frameIndex = m_metalContext->GetCurrentFrameIndex();

    for (const auto& name : m_executionOrder) {
        auto& pass = m_passes.at(name);

        for (MTL::Event* waitEvent : pass->WaitEventList) {
            queue->wait(waitEvent, frameIndex);
        }

        auto commandBuffer = m_commandBufferPool->Acquire();
        pass->Execute(m_resources, commandBuffer);
        commandBuffer.SubmitTo(queue);

        queue->signalEvent(pass->Event, frameIndex);
    }
}
