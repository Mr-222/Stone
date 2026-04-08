#include "Core/MetalContext.h"

#include "Utility/Logger.h"

void MetalContext::Init(CA::MetalLayer* metalLayer) {
    m_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    m_commandQueue = NS::TransferPtr(m_device->newCommandQueue());

    m_swapchain = NS::RetainPtr(metalLayer);

    m_swapchain->setDevice(m_device.get());
    m_swapchain->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    m_swapchain->setFramebufferOnly(true);
}

void MetalContext::BeginFrame() {
    m_currentDrawable = m_swapchain->nextDrawable();
    LOG_ERROR_IF(!m_currentDrawable, "No more drawables available!");
}

void MetalContext::EndFrame() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
    MTL::RenderPassDescriptor* passDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
    MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);

    colorAttachment->setTexture(m_currentDrawable->texture());
    colorAttachment->setLoadAction(MTL::LoadActionClear);
    colorAttachment->setClearColor(MTL::ClearColor::Make(0.1, 0.2, 0.3, 1.0));
    colorAttachment->setStoreAction(MTL::StoreActionStore);

    MTL::RenderCommandEncoder* commandEncoder = commandBuffer->renderCommandEncoder(passDescriptor);
    commandEncoder->endEncoding();

    commandBuffer->presentDrawable(m_currentDrawable);
    commandBuffer->commit();

    pool->release();
}
