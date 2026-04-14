#include "Buffer.h"

Buffer::Buffer(MTL::Device* device, size_t size, MTL::ResourceOptions options) : m_size(size)
{
    m_buffer = device->newBuffer(size, options);
}

Buffer::Buffer(MTL::Device *device, const void *data, size_t size, MTL::ResourceOptions options) : m_size(size)
{
    assert(options == MTL::ResourceStorageModeShared);
    m_buffer = device->newBuffer(data, size, options);
}

Buffer::~Buffer() {
    if (m_buffer)
        m_buffer->release();
}

void Buffer::UploadFrom(const Buffer& src, CommandBuffer& cmd) const {
    MTL4::ComputeCommandEncoder* encoder = cmd.BeginBlitPass();
    encoder->copyFromBuffer(src.GetNativeBuffer(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
}

void Buffer::UploadFromFlush(const Buffer& src, CommandBufferPool& pool, MTL4::CommandQueue* queue) const {
    CommandBuffer temp = pool.AcquireFlushGPU();
    MTL4::ComputeCommandEncoder* encoder = temp.BeginBlitPass();
    encoder->copyFromBuffer(src.GetNativeBuffer(), 0, m_buffer, 0, m_size);
    encoder->endEncoding();
    temp.SubmitTo(queue);
}
