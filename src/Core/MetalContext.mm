#include "Core/MetalContext.h"
#include "Utility/Logger.h"

void MetalContext::Init(CA::MetalLayer* metalLayer) {
    m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    m_commandQueue = NS::TransferPtr(m_device->newMTL4CommandQueue());

    m_frameBoundarySemaphore = dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT);
    m_frameEvent = NS::TransferPtr(m_device->newSharedEvent());
    m_eventListener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
    m_currentFrameIndex = 0;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_commandAllocators[i] = NS::TransferPtr(m_device->newCommandAllocator());
    }

    m_commandBuffer = NS::TransferPtr(m_device->newCommandBuffer());

    m_swapchain = NS::RetainPtr(metalLayer);
    m_swapchain->setDevice(m_device.get());
    m_swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    m_swapchain->setFramebufferOnly(true);
}

void MetalContext::BeginFrame() {
    // Take a token. Block the CPU if GPU is 3 frames behind.
    dispatch_semaphore_wait(m_frameBoundarySemaphore, DISPATCH_TIME_FOREVER);

    m_currentDrawable = m_swapchain->nextDrawable();
    LOG_ERROR_IF(!m_currentDrawable, "No more drawables available!");

    int bufferIndex = m_currentFrameIndex % MAX_FRAMES_IN_FLIGHT;
    m_commandAllocators[bufferIndex]->reset();
    m_commandBuffer->beginCommandBuffer(m_commandAllocators[bufferIndex].get());
}

void MetalContext::EndFrame() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
    MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);

    colorAttachment->setTexture(m_currentDrawable->texture());
    colorAttachment->setLoadAction(MTL::LoadActionClear);
    colorAttachment->setClearColor(MTL::ClearColor::Make(0.1, 0.2, 0.3, 1.0));
    colorAttachment->setStoreAction(MTL::StoreActionStore);

    MTL4::RenderCommandEncoder* commandEncoder = m_commandBuffer->renderCommandEncoder(passDescriptor);
    commandEncoder->endEncoding();

    m_commandBuffer->endCommandBuffer();

    MTL4::CommandBuffer* buffersToSubmit[] = { m_commandBuffer.get() };
    m_commandQueue->commit(buffersToSubmit, 1);

    // Tell the GPU to return the token when it's done
    uint64_t signalValue = m_currentFrameIndex + 1;
    dispatch_semaphore_t blockSema = m_frameBoundarySemaphore;
    m_frameEvent->notifyListener(m_eventListener.get(), signalValue, ^(MTL::SharedEvent* ev, uint64_t val) {
        dispatch_semaphore_signal(blockSema);
    });

    m_commandQueue->signalEvent(m_frameEvent.get(), signalValue);

    m_commandQueue->signalDrawable(m_currentDrawable);
    m_currentDrawable->present();

    m_currentFrameIndex++;

    pool->release();
}
