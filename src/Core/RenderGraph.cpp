#include "RenderGraph.h"

RenderGraph::RenderGraph(CommandBuffer&& cmd) : m_cmd(std::move(cmd)) {}

void RenderGraph::Compile() {
    // TODO
}

void RenderGraph::Execute(MTL4::CommandQueue* queue) {
    for (auto& pass : m_passes)
        pass->Execute(m_cmd, m_resources);

    m_cmd.SubmitTo(queue);
}
