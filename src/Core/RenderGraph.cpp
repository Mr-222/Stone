#include "RenderGraph.h"

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

void RenderGraph::Compile() {
    for (const auto& pass : m_passes) {
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
            }
        }
    }
}

void RenderGraph::Execute(MTL4::CommandQueue* queue) {
    LOG_ERROR_IF(!queue, "RenderGraph execute requires a valid command queue.");

    for (auto& pass : m_passes) {
        auto commandBuffer = m_commandBufferPool->Acquire();
        pass->Execute(m_resources, commandBuffer);
        commandBuffer.SubmitTo(queue);
    }
}
