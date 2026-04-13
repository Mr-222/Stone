#include "CommandBufferPool.h"

#include "Utility/logger.h"

CommandBufferPool::CommandBufferPool(size_t cap, const MetalContext &context)
    : capacity(cap), count(cap), context(context)
{
    pool.resize(cap);
    MTL::Device* device = context.GetDevice();
    for (size_t i = 0; i < cap; i++)
        pool[i] = device->newCommandBuffer();
}

MTL4::CommandBuffer *CommandBufferPool::Acquire() {
    std::unique_lock lock(mtx);

    cv.wait(lock, [this] { return count > 0; });

    MTL4::CommandBuffer* buffer = pool[head];

    head = (head + 1) % capacity;
    count--;

    return buffer;
}

void CommandBufferPool::Release(MTL4::CommandBuffer* buffer) {
    std::unique_lock lock(mtx);

    LOG_ERROR_IF(count == capacity, "Command buffer pool is already full! Cannot release buffer.");

    pool[tail] = buffer;

    tail = (tail + 1) % capacity;
    count++;

    lock.unlock();
    cv.notify_one();
}