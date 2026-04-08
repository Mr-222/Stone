#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

class MetalContext {
public:
    MetalContext() = default;
    ~MetalContext() = default;

    void Init(CA::MetalLayer* metalLayer);
    void BeginFrame();
    void EndFrame();

    MTL::Device* GetDevice() const { return m_device.get(); }
    MTL::CommandQueue* GetCommandQueue() const { return m_commandQueue.get(); }

private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL::CommandQueue> m_commandQueue;

    NS::SharedPtr<CA::MetalLayer> m_swapchain;
    CA::MetalDrawable* m_currentDrawable = nullptr;
};
