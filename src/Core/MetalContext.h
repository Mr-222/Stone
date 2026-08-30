#pragma once

#include <Metal/Metal.hpp>
#include <array>
#include <semaphore>
#include <vector>

#include "Window.h"

class MetalContext {
public:
    MetalContext(CA::MetalLayer* metalLayer);
    ~MetalContext() = default;

    void Init(CA::MetalLayer* metalLayer);
    void BeginFrame();
    void EndFrame();

    MTL::Device* GetDevice() const { return m_device; }
    CA::MetalDrawable* GetCurrentDrawable() const { return m_currentDrawable; }
    MTL4::CommandAllocator* GetCurrentAllocator() const { return m_commandAllocators[GetCurrentFrameSlot()]; }
    MTL::ResidencySet* GetCurrentFrameResidencySet() const { return m_frameResidencySets[GetCurrentFrameSlot()].get(); }
    MTL4::CommandQueue* GetRenderCommandQueue() const { return m_renderQueue; }
    MTL4::CommandQueue* GetComputeCommandQueue() const { return m_computeQueue; }
    uint64_t GetCurrentFrameIndex() const { return m_currentFrameIndex; }
    uint32_t GetCurrentFrameSlot() const { return static_cast<uint32_t>(m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT); }
    uint32_t GetFrameSlotCount() const { return MAX_FRAMES_IN_FLIGHT; }
    MTL::PixelFormat GetSwapchainPixelFormat() const { return kSwapchainPixelFormat; }
    uint32_t GetDrawableWidth() const { return static_cast<uint32_t>(m_swapchain->drawableSize().width); }
    uint32_t GetDrawableHeight() const { return static_cast<uint32_t>(m_swapchain->drawableSize().height); }
    CA::MetalLayer* GetSwapchain() const { return m_swapchain; }

private:
    static constexpr MTL::PixelFormat kSwapchainPixelFormat = MTL::PixelFormatRGBA16Float;

    MTL::Device* m_device;
    MTL4::CommandQueue* m_renderQueue;
    MTL4::CommandQueue* m_computeQueue;

    // Synchronization primitives
    std::counting_semaphore<MAX_FRAMES_IN_FLIGHT> m_frameBoundarySemaphore;
    uint64_t m_currentFrameIndex;
    MTL::SharedEvent* m_frameEvent;
    MTL::SharedEventListener* m_eventListener;

    MTL4::CommandAllocator* m_commandAllocators[MAX_FRAMES_IN_FLIGHT];
    std::array<NS::SharedPtr<MTL::ResidencySet>, MAX_FRAMES_IN_FLIGHT> m_frameResidencySets;

    CA::MetalLayer* m_swapchain;
    CA::MetalDrawable* m_currentDrawable;
};
