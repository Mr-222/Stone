#include "CommandBuffer.h"

#include <semaphore>

#include "Core/CommandBufferPool.h"

CommandBuffer::CommandBuffer(MTL4::CommandBuffer* cmd, MTL4::CommandAllocator* allocator, CommandBufferPool* pool, bool flush = false)
    : m_commandBuffer(cmd), m_allocator(allocator), m_pool(pool), m_flushGPU(flush) {}

CommandBuffer::~CommandBuffer() {
    if (m_pool && m_commandBuffer)
        m_pool->Release(m_commandBuffer);
}

MTL4::RenderCommandEncoder* CommandBuffer::BeginRenderPass(MTL4::RenderPassDescriptor* desc, MTL::ResidencySet* set) const {
    m_commandBuffer->beginCommandBuffer(m_allocator);
    m_commandBuffer->useResidencySet(set);
    return m_commandBuffer->renderCommandEncoder(desc);
}

void CommandBuffer::SubmitTo(MTL4::CommandQueue* submitQueue) const {
    m_commandBuffer->endCommandBuffer();
    MTL4::CommandBuffer* bufferToSubmit[] { m_commandBuffer };

    if (m_flushGPU) {
        std::binary_semaphore gpuDone{0};

        MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init();

        auto feedbackBlock = [&gpuDone](MTL4::CommitFeedback*) {
            gpuDone.release(); // Unblock thread
        };

        options->addFeedbackHandler(feedbackBlock);

        submitQueue->commit(bufferToSubmit, 1, options);
        options->release();

        // Halt thread here until the queue fires the feedback block
        gpuDone.acquire();
    } else {
        // Standard non-blocking commit
        submitQueue->commit(bufferToSubmit, 1);
    }
}
