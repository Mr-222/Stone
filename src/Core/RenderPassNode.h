#pragma once

#include <functional>
#include <utility>

#include "CommandBuffer.h"

class RenderGraphResources;

class RenderPassNodeBase {
public:
    virtual ~RenderPassNodeBase() = default;

    // Graph calls this during rg.Execute()
    virtual void Execute(CommandBuffer& cmd, RenderGraphResources& resources) = 0;
};

template<typename PassData>
class RenderPassNode : public RenderPassNodeBase {
public:
    using ExecuteCallback = std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)>;

    RenderPassNode(const PassData& data, ExecuteCallback executeFn)
        : m_data(data), m_executeFn(std::move(executeFn)) {}

    void Execute(CommandBuffer& cmd, RenderGraphResources& resources) override {
        m_executeFn(m_data, resources, cmd);
    }

private:
    PassData m_data;
    ExecuteCallback m_executeFn;
};
