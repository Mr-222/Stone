#include "MetalContext.h"

#include "Utility/Logger.h"
#include "Core/CommandBufferPool.h"

MetalContext::MetalContext(CA::MetalLayer* metalLayer): m_currentFrameIndex(0), m_frameBoundarySemaphore(MAX_FRAMES_IN_FLIGHT) {
    m_device = MTL::CreateSystemDefaultDevice();
    m_queue = m_device->newMTL4CommandQueue();

    m_frameEvent = m_device->newSharedEvent();
    m_eventListener = MTL::SharedEventListener::alloc()->init();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_commandAllocators[i] = m_device->newCommandAllocator();
    }

    m_swapchain = metalLayer;
    m_swapchain->setDevice(m_device);
    m_swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    m_swapchain->setFramebufferOnly(true);
    m_swapchain->setMaximumDrawableCount(MAX_FRAMES_IN_FLIGHT);
    m_queue->addResidencySet(m_swapchain->residencySet());
}

void MetalContext::BeginFrame() {
    // Take a token. Block the CPU if GPU is 3 frames behind.
    m_frameBoundarySemaphore.acquire();

    m_currentDrawable = m_swapchain->nextDrawable();
    LOG_ERROR_IF(!m_currentDrawable, "No more drawables available!");

    const uint32_t bufferIndex = static_cast<uint32_t>(m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT);
    m_commandAllocators[bufferIndex]->reset();
}

void MetalContext::EndFrame() {
    // Tell the GPU to return the token when it's done
    uint64_t signalValue = m_currentFrameIndex + 1;
    auto* sema = &m_frameBoundarySemaphore;
    m_frameEvent->notifyListener(m_eventListener, signalValue, ^(MTL::SharedEvent* ev, uint64_t val) {
        sema->release();
    });

    m_queue->signalEvent(m_frameEvent, signalValue);

    m_queue->wait(m_currentDrawable);
    m_queue->signalDrawable(m_currentDrawable);
    m_currentDrawable->present();

    m_currentFrameIndex++;
}
