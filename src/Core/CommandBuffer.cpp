#include "CommandBuffer.h"

#include <semaphore>
#include <utility>

#include "CommandBufferPool.h"
#include "Utility/Logger.h"

CommandBuffer::CommandBuffer(MTL4::CommandBuffer* cmd,
                             MTL4::CommandAllocator* allocator,
                             MTL::Device* device,
                             MTL::ResidencySet* frameResidencySet,
                             CommandBufferPool* pool,
                             bool flush)
    : m_flushGPU(flush)
    , m_hasBegun(false)
    , m_commandBuffer(cmd)
    , m_allocator(allocator)
    , m_residencySet(frameResidencySet)
    , m_pool(pool)
{
    if (!m_flushGPU) {
        LOG_ERROR_IF(!m_residencySet, "Frame command buffer requires a frame residency set.");
    } else {
        MTL::ResidencySetDescriptor* descriptor = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
        MTL::ResidencySet* residencySet = device->newResidencySet(descriptor, nullptr);

        LOG_ERROR_IF(!residencySet, "Failed to create residency set for synchronous command buffer.");

        residencySet->requestResidency();
        m_ownedResidencySet = NS::TransferPtr(residencySet);
        m_residencySet = m_ownedResidencySet.get();
    }
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : m_flushGPU(other.m_flushGPU)
    , m_hasBegun(other.m_hasBegun)
    , m_commandBuffer(other.m_commandBuffer)
    , m_allocator(other.m_allocator)
    , m_residencySet(other.m_residencySet)
    , m_ownedResidencySet(std::move(other.m_ownedResidencySet))
    , m_pool(other.m_pool)
{
    other.m_flushGPU = false;
    other.m_hasBegun = false;
    other.m_commandBuffer = nullptr;
    other.m_allocator = nullptr;
    other.m_residencySet = nullptr;
    other.m_pool = nullptr;
}

CommandBuffer::~CommandBuffer() {
    if (m_pool && m_commandBuffer)
        m_pool->Release(m_commandBuffer);
}

void CommandBuffer::AddResource(const MTL::Allocation* allocation) {
    m_residencySet->addAllocation(allocation);
}

MTL4::RenderCommandEncoder* CommandBuffer::BeginRenderPass(MTL4::RenderPassDescriptor* desc) {
    if (!m_hasBegun) {
        m_hasBegun = true;
        m_commandBuffer->beginCommandBuffer(m_allocator);
    }
    return m_commandBuffer->renderCommandEncoder(desc);
}

MTL4::ComputeCommandEncoder* CommandBuffer::BeginComputePass() {
    if (!m_hasBegun) {
        m_hasBegun = true;
        m_commandBuffer->beginCommandBuffer(m_allocator);
    }
    return m_commandBuffer->computeCommandEncoder();
}

void CommandBuffer::SubmitTo(MTL4::CommandQueue* submitQueue) {
    LOG_WARN_IF(!m_hasBegun, "Command Buffer has not begun but you submit it.");

    if (m_residencySet->allocationCount() > 0) {
        m_residencySet->commit();
        m_commandBuffer->useResidencySet(m_residencySet);
    }

    m_commandBuffer->endCommandBuffer();
    MTL4::CommandBuffer* bufferToSubmit[] { m_commandBuffer };

    if (m_flushGPU) {
        std::binary_semaphore gpuDone{0};

        MTL4::CommitOptions* options = MTL4::CommitOptions::alloc()->init()->autorelease();

        auto feedbackBlock = [&gpuDone](MTL4::CommitFeedback*) {
            gpuDone.release(); // Unblock thread
        };

        options->addFeedbackHandler(feedbackBlock);

        submitQueue->commit(bufferToSubmit, 1, options);

        // Halt thread here until the queue fires the feedback block
        gpuDone.acquire();
    } else {
        // Standard non-blocking commit
        submitQueue->commit(bufferToSubmit, 1);
    }

    if (m_flushGPU && m_residencySet->allocationCount() > 0) {
        m_residencySet->removeAllAllocations();
        m_residencySet->commit();
    }

    m_hasBegun = false;
}
