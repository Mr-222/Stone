#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <dispatch/dispatch.h>
#include <vector>

#include "Core/Window.h"

class MetalContext {
public:
    MetalContext(CA::MetalLayer* metalLayer);
    ~MetalContext() = default;

    void Init(CA::MetalLayer* metalLayer);
    void BeginFrame();
    void EndFrame(const std::vector<MTL4::CommandBuffer*>& buffers);

    MTL::Device* GetDevice() const { return m_device.get(); }
    CA::MetalDrawable* GetCurrentDrawable() const { return m_currentDrawable; }
    MTL4::CommandAllocator* GetCurrentAllocator() const { return m_commandAllocators[m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT].get(); }

private:
    NS::SharedPtr<MTL::Device> m_device;
    NS::SharedPtr<MTL4::CommandQueue> m_queue;

    // Synchronization primitives
    dispatch_semaphore_t m_frameBoundarySemaphore = nullptr;
    uint64_t m_currentFrameIndex = 0;
    NS::SharedPtr<MTL::SharedEvent> m_frameEvent;
    NS::SharedPtr<MTL::SharedEventListener> m_eventListener;

    NS::SharedPtr<MTL4::CommandAllocator> m_commandAllocators[MAX_FRAMES_IN_FLIGHT];

    NS::SharedPtr<CA::MetalLayer> m_swapchain;
    CA::MetalDrawable* m_currentDrawable;
};
