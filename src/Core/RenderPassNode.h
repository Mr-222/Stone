#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <Metal/Metal.hpp>

#include "CommandBuffer.h"
#include "RenderGraphBuilder.h"

class RenderPassNodeBase {
public:
    RenderPassNodeBase(std::string name, std::vector<RenderGraphTextureAccess> textureAccesses, CommandBuffer&& cmd)
        : m_name(std::move(name))
        , m_textureAccesses(std::move(textureAccesses))
        , m_cmd(std::move(cmd))
    {
    }

    virtual ~RenderPassNodeBase() = default;

    // Graph calls this during rg.Execute()
    virtual void Execute(MTL4::CommandQueue*, RenderGraphResources& resources) = 0;

    const std::string& GetName() const { return m_name; }
    const std::vector<RenderGraphTextureAccess>& GetTextureAccesses() const { return m_textureAccesses; }

    CommandBuffer m_cmd;

private:
    std::string m_name;
    std::vector<RenderGraphTextureAccess> m_textureAccesses;
};

template<typename PassData>
class RenderPassNode : public RenderPassNodeBase {
public:
    using ExecuteCallback = std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)>;

    RenderPassNode(
        std::string name,
        std::vector<RenderGraphTextureAccess> textureAccesses,
        CommandBuffer&& cmd,
        const PassData& data,
        ExecuteCallback executeFn)
        : RenderPassNodeBase(std::move(name), std::move(textureAccesses), std::move(cmd))
        , m_data(data)
        , m_executeFn(std::move(executeFn))
    {
    }

    void Execute(MTL4::CommandQueue* queue, RenderGraphResources& resources) override {
        m_executeFn(m_data, resources, m_cmd);
        m_cmd.SubmitTo(queue);
    }

private:
    PassData m_data;
    ExecuteCallback m_executeFn;
};
