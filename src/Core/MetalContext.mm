#include "MetalContext.h"

#include "Utility/Logger.h"
#include "Core/CommandBufferPool.h"

MetalContext::MetalContext(CA::MetalLayer* metalLayer) {
    m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    m_queue = NS::TransferPtr(m_device->newMTL4CommandQueue());

    m_frameBoundarySemaphore = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);
    m_frameEvent = NS::TransferPtr(m_device->newSharedEvent());
    m_eventListener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_commandAllocators[i] = NS::TransferPtr(m_device->newCommandAllocator());
    }

    m_swapchain = NS::RetainPtr(metalLayer);
    m_swapchain->setDevice(m_device.get());
    m_swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    m_swapchain->setFramebufferOnly(true);
    m_swapchain->setMaximumDrawableCount(MAX_FRAMES_IN_FLIGHT);
    m_queue->addResidencySet(m_swapchain->residencySet());
}

void MetalContext::BeginFrame() {
    // Take a token. Block the CPU if GPU is 3 frames behind.
    dispatch_semaphore_wait(m_frameBoundarySemaphore, DISPATCH_TIME_FOREVER);

    m_currentDrawable = m_swapchain->nextDrawable();
    LOG_ERROR_IF(!m_currentDrawable, "No more drawables available!");

    const uint32_t bufferIndex = static_cast<uint32_t>(m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT);
    m_commandAllocators[bufferIndex]->reset();
}

void MetalContext::EndFrame(const std::vector<MTL4::CommandBuffer*>& buffers) {
    if (!buffers.empty())
        m_queue->commit(buffers.data(), buffers.size());

    // Tell the GPU to return the token when it's done
    uint64_t signalValue = m_currentFrameIndex + 1;
    dispatch_semaphore_t blockSema = m_frameBoundarySemaphore;
    m_frameEvent->notifyListener(m_eventListener.get(), signalValue, ^(MTL::SharedEvent* ev, uint64_t val) {
        dispatch_semaphore_signal(blockSema);
    });

    m_queue->signalEvent(m_frameEvent.get(), signalValue);

    m_queue->wait(m_currentDrawable);
    m_queue->signalDrawable(m_currentDrawable);
    m_currentDrawable->present();

    m_currentFrameIndex++;
}
