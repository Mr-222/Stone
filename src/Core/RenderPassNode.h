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
    RenderPassNodeBase(MTL::Device* device ,std::string name, std::vector<RenderGraphResourceAccess> resourceAccesses)
        : m_name(std::move(name))
        , m_resourceAccesses(std::move(resourceAccesses))
        , Event(device->newEvent())
    {
    }

    virtual ~RenderPassNodeBase() { Event->release(); }

    // Graph calls this during rg.Execute()
    virtual void Execute(RenderGraphResources&, CommandBuffer&) = 0;

    const std::string& GetName() const { return m_name; }
    const std::vector<RenderGraphResourceAccess>& GetResourceAccesses() const { return m_resourceAccesses; }

    // For synchronizing between passes
    MTL::Event* Event;
    std::vector<MTL::Event*> WaitEventList;

private:
    std::string m_name;
    std::vector<RenderGraphResourceAccess> m_resourceAccesses;
};

template<typename PassData>
class RenderPassNode : public RenderPassNodeBase {
public:
    using ExecuteCallback = std::function<void(const PassData&, RenderGraphResources&, CommandBuffer&)>;

    RenderPassNode(
        MTL::Device* device,
        std::string name,
        std::vector<RenderGraphResourceAccess> resourceAccesses,
        const PassData& data,
        ExecuteCallback executeFn)
        : RenderPassNodeBase(device, std::move(name), std::move(resourceAccesses))
        , m_data(data)
        , m_executeFn(std::move(executeFn))
    {
    }

    void Execute(RenderGraphResources& resources, CommandBuffer& cmd) override {
        m_executeFn(m_data, resources, cmd);
    }

private:
    PassData m_data;
    ExecuteCallback m_executeFn;
};
