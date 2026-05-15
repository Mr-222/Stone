#include "RenderGraph.h"

#include "Utility/Logger.h"

RenderGraph::RenderGraph(std::shared_ptr<MetalContext> metalContext, std::shared_ptr<CommandBufferPool> commandBufferPool)
    : m_metalContext(std::move(metalContext)), m_commandBufferPool(std::move(commandBufferPool))
    {}

RenderGraphTextureHandle RenderGraph::ImportTexture(const std::string& name, MTL::Texture* texture) {
    return m_resources.ImportTexture(name, texture);
}

RenderGraphTextureHandle RenderGraph::ImportTexture(const std::string& name, const Texture& texture) {
    return m_resources.ImportTexture(name, texture);
}

void RenderGraph::Compile() {
    for (const auto& pass : m_passes) {
        for (const RenderGraphTextureAccess& access : pass->GetTextureAccesses()) {
            LOG_ERROR_IF(
                !m_resources.IsTextureHandleValid(access.texture),
                "Render pass '{}' declared an invalid texture dependency.",
                pass->GetName());
        }
    }
}

void RenderGraph::Execute(MTL4::CommandQueue* queue) {
    LOG_ERROR_IF(!queue, "RenderGraph execute requires a valid command queue.");

    for (auto& pass : m_passes)
        pass->Execute(queue, m_resources);
}
