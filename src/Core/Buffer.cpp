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

    MTL4::ComputeCommandEncoder* encoder = cmd.BeginBlitPass();
    encoder->copyFromBuffer(src.GetNative(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
}

void Buffer::UploadFromFlush(const Buffer& src, CommandBufferPool& pool, MTL4::CommandQueue* queue) const {
    CommandBuffer temp = pool.AcquireFlushGPU();
    AddBufferResidency(temp, src.GetNative());
    AddBufferResidency(temp, m_buffer);

    MTL4::ComputeCommandEncoder* encoder = temp.BeginBlitPass();
    encoder->copyFromBuffer(src.GetNative(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
    temp.SubmitTo(queue);
}
