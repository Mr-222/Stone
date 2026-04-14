#pragma once

#include <Metal/Metal.hpp>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <vector>

#include <Core/MetalContext.h>
#include <Core/CommandBuffer.h>

class CommandBufferPool {
public:
    CommandBufferPool(size_t cap, const MetalContext& context);

    CommandBuffer Acquire();

private:
    friend class CommandBuffer;
    void Release(MTL4::CommandBuffer* buffer);

    const MetalContext& m_context;

    std::vector<MTL4::CommandBuffer*> m_pool;
    size_t m_capacity;

    size_t m_head = 0;
    size_t m_tail = 0;
    size_t m_count;

    std::mutex m_mtx;
    std::condition_variable m_cv;
};

