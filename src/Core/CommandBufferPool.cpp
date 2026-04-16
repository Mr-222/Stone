#include "CommandBufferPool.h"

#include "Utility/Logger.h"

CommandBufferPool::CommandBufferPool(size_t cap, const MetalContext &context)
    : m_capacity(cap), m_count(cap), m_context(context)
{
    m_pool.resize(cap);
    MTL::Device* device = context.GetDevice();
    for (size_t i = 0; i < cap; i++)
        m_pool[i] = device->newCommandBuffer();
}

CommandBuffer CommandBufferPool::Acquire() {
    std::unique_lock lock(m_mtx);

    m_cv.wait(lock, [this] { return m_count > 0; });

    MTL4::CommandBuffer* buffer = m_pool[m_head];

    m_head = (m_head + 1) % m_capacity;
    m_count--;

    return CommandBuffer(buffer, m_context.GetCurrentAllocator(), m_context.GetDevice(), this, false);
}

CommandBuffer CommandBufferPool::AcquireFlushGPU() {
    std::unique_lock lock(m_mtx);

    m_cv.wait(lock, [this] { return m_count > 0; });

    MTL4::CommandBuffer* buffer = m_pool[m_head];

    m_head = (m_head + 1) % m_capacity;
    m_count--;

    return CommandBuffer(buffer, m_context.GetCurrentAllocator(), m_context.GetDevice(), this, true);
}

void CommandBufferPool::Release(MTL4::CommandBuffer* buffer) {
    std::unique_lock lock(m_mtx);

    LOG_ERROR_IF(m_count == m_capacity, "Command buffer pool is already full! Cannot release buffer.");

    m_pool[m_tail] = buffer;

    m_tail = (m_tail + 1) % m_capacity;
    m_count++;

    lock.unlock();
    m_cv.notify_one();
}
