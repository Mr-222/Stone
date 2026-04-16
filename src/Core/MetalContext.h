#pragma once

#include <Metal/Metal.hpp>
#include <semaphore>
#include <vector>

#include "Core/Window.h"

class MetalContext {
public:
    MetalContext(CA::MetalLayer* metalLayer);
    ~MetalContext() = default;

    void Init(CA::MetalLayer* metalLayer);
    void BeginFrame();
    void EndFrame();

    MTL::Device* GetDevice() const { return m_device; }
    CA::MetalDrawable* GetCurrentDrawable() const { return m_currentDrawable; }
    MTL4::CommandAllocator* GetCurrentAllocator() const { return m_commandAllocators[m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT]; }
    MTL4::CommandQueue* GetCommandQueue() const { return m_queue; }

private:
    MTL::Device* m_device;
    MTL4::CommandQueue* m_queue;

    // Synchronization primitives
    std::counting_semaphore<MAX_FRAMES_IN_FLIGHT> m_frameBoundarySemaphore;
    uint64_t m_currentFrameIndex;
    MTL::SharedEvent* m_frameEvent;
    MTL::SharedEventListener* m_eventListener;

    MTL4::CommandAllocator* m_commandAllocators[MAX_FRAMES_IN_FLIGHT];

    CA::MetalLayer* m_swapchain;
    CA::MetalDrawable* m_currentDrawable;
};
