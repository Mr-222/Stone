#pragma once

#include <Metal/Metal.hpp>

class IndirectCommandBuffer {
public:
    IndirectCommandBuffer(MTL::Device* device, MTL::IndirectCommandBufferDescriptor* descriptor, size_t maxCommandCount, MTL::ResourceOptions options);
    ~IndirectCommandBuffer();

    IndirectCommandBuffer(const IndirectCommandBuffer&) = delete;
    IndirectCommandBuffer& operator=(const IndirectCommandBuffer&) = delete;

    MTL::IndirectCommandBuffer* GetNative() const { return m_indirectCB; }

private:
    MTL::IndirectCommandBuffer* m_indirectCB;
};
