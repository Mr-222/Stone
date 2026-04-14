#pragma once

#include <Metal/Metal.hpp>

#include "Core/CommandBufferPool.h"

class Buffer {
public:
    Buffer(MTL::Device* device, size_t size, MTL::ResourceOptions options);
    Buffer(MTL::Device* device, const void* data, size_t size, MTL::ResourceOptions options);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void UploadFrom(const Buffer& src, CommandBuffer& cmd) const;
    void UploadFromFlush(const Buffer& src, CommandBufferPool& pool, MTL4::CommandQueue* queue) const;

    MTL::Buffer* GetNativeBuffer() const { return m_buffer; }
    size_t       GetSize()         const { return m_size; }
    uint64_t     GetGPUAddress()   const { return m_buffer->gpuAddress(); }

private:
    MTL::Buffer* m_buffer;
    size_t       m_size;
};
