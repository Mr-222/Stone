#include "IndirectCommandBuffer.h"

IndirectCommandBuffer::IndirectCommandBuffer(MTL::Device* device, MTL::IndirectCommandBufferDescriptor* descriptor, size_t maxCommandCount, MTL::ResourceOptions options) {
    m_indirectCB = device->newIndirectCommandBuffer(descriptor, maxCommandCount, options);
}

IndirectCommandBuffer::~IndirectCommandBuffer() {
    if (m_indirectCB) {
        m_indirectCB->release();
    }
}
