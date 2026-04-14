#pragma once

#include <Metal/Metal.hpp>

class CommandBufferPool;

class CommandBuffer {
public:
    CommandBuffer() = delete;
    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&) = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    MTL4::RenderCommandEncoder*   BeginRenderPass(MTL4::RenderPassDescriptor* desc, MTL::ResidencySet* set = nullptr);
    MTL4::ComputeCommandEncoder*  BeginBlitPass();
    void SubmitTo(MTL4::CommandQueue* submitQueue) const;

private:
    friend class CommandBufferPool;
    CommandBuffer(MTL4::CommandBuffer* cmd, MTL4::CommandAllocator* allocator, CommandBufferPool* pool, bool flushGPU);

    bool m_flushGPU;
    bool m_hasBegun;

    MTL4::CommandBuffer*    m_commandBuffer;
    MTL4::CommandAllocator* m_allocator;
    CommandBufferPool*      m_pool;
};
