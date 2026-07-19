#include "Buffer.h"

#include "Utility/Logger.h"

namespace {
void AddBufferResidency(CommandBuffer& cmd, MTL::Buffer* buffer) {
    if (!buffer)
        return;

    MTL::Heap* heap = buffer->heap();
    if (heap)
        cmd.AddResource(heap);
    else
        cmd.AddResource(buffer);
}
}

Buffer::Buffer(MTL::Device* device, size_t size, MTL::ResourceOptions options) : m_size(size)
{
    m_buffer = device->newBuffer(size, options);
}

Buffer::Buffer(MTL::Device *device, const void *data, size_t size, MTL::ResourceOptions options) : m_size(size)
{
    LOG_ERROR_IF(options != MTL::ResourceStorageModeShared, "Unable to create a GPU private buffer with initialization data. Please use a blit encoder to transfer data.");
    m_buffer = device->newBuffer(data, size, options);
}

Buffer::Buffer(const Heap &heap, size_t size, MTL::ResourceOptions options) : m_size(size) {
    LOG_ERROR_IF(heap.GetUsedSize() + size > heap.GetSize(), "Heap has ran out of space.");
    m_buffer = heap.GetNative()->newBuffer(size, options);
}

Buffer::~Buffer() {
    if (m_buffer)
        m_buffer->release();
}

void Buffer::UploadFrom(const Buffer& src, CommandBuffer& cmd) const {
    AddBufferResidency(cmd, src.GetNative());
    AddBufferResidency(cmd, m_buffer);

    MTL4::ComputeCommandEncoder* encoder = cmd.BeginComputePass();
    encoder->copyFromBuffer(src.GetNative(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
}

void Buffer::UploadFromFlush(const Buffer& src, CommandBufferPool& pool, MTL4::CommandQueue* queue) const {
    CommandBuffer temp = pool.AcquireFlushGPU();
    AddBufferResidency(temp, src.GetNative());
    AddBufferResidency(temp, m_buffer);

    MTL4::ComputeCommandEncoder* encoder = temp.BeginComputePass();
    encoder->copyFromBuffer(src.GetNative(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
    temp.SubmitTo(queue);
}

void Buffer::Update(const void *data, size_t size, size_t offset) {
    LOG_ERROR_IF(m_buffer->storageMode() == MTL::StorageModePrivate, "Cannot directly update a GPU Private buffer from the CPU. Use UpdateStaged instead.");
    LOG_ERROR_IF(offset + size > m_size, "Buffer update out of bounds.");

    uint8_t* destPtr = static_cast<uint8_t*>(m_buffer->contents()) + offset;
    std::memcpy(destPtr, data, size);

    // If using
    if (m_buffer->storageMode() == MTL::StorageModeManaged)
        m_buffer->didModifyRange(NS::Range::Make(offset, size));
}

void Buffer::UpdateStaged(const void *data, size_t size, size_t offset, CommandBufferPool &pool, MTL4::CommandQueue *queue) {
    LOG_ERROR_IF(offset + size > m_size, "Buffer update out of bounds.");

    // 1. Create a temporary shared buffer (staging buffer)
    Buffer stagingBuffer(m_buffer->device(), data, size, MTL::ResourceStorageModeShared);

    // 2. Acquire a command buffer
    CommandBuffer temp = pool.AcquireFlushGPU();
    AddBufferResidency(temp, stagingBuffer.GetNative());
    AddBufferResidency(temp, m_buffer);

    // 3. Issue the copy command
    MTL4::ComputeCommandEncoder* encoder = temp.BeginComputePass();
    encoder->copyFromBuffer(stagingBuffer.GetNative(), 0, m_buffer, offset, size);
    encoder->endEncoding();

    temp.SubmitTo(queue);
}
