#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <dispatch/dispatch.h>

class MetalContext {
public:
    MetalContext() = default;
    ~MetalContext() = default;

    void Init(CA::MetalLayer* metalLayer);
    void BeginFrame();
    void EndFrame();

    MTL::Device* GetDevice() const { return m_device.get(); }
    MTL4::CommandQueue* GetCommandQueue() const { return m_commandQueue.get(); }
    CA::MetalDrawable* GetCurrentDrawable() const { return m_currentDrawable; }

private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL4::CommandQueue> m_commandQueue;

    // Synchronization primitives
    dispatch_semaphore_t m_frameBoundarySemaphore;
    uint64_t m_currentFrameIndex;
    NS::SharedPtr<MTL::SharedEvent> m_frameEvent;
    NS::SharedPtr<MTL::SharedEventListener> m_eventListener;

    static constexpr uint MAX_FRAMES_IN_FLIGHT = 3;
    NS::SharedPtr<MTL4::CommandAllocator> m_commandAllocators[MAX_FRAMES_IN_FLIGHT];
    NS::SharedPtr<MTL4::CommandBuffer> m_commandBuffer;

    NS::SharedPtr<CA::MetalLayer> m_swapchain;
    CA::MetalDrawable* m_currentDrawable = nullptr;
};
