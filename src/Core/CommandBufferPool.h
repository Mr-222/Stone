#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <vector>

#include <Core/MetalContext.h>

class CommandBufferPool {
public:
    CommandBufferPool(size_t cap, const MetalContext& context);

    MTL4::CommandBuffer* Acquire();
    void Release(MTL4::CommandBuffer* buffer);

private:
    const MetalContext& context;

    std::vector<MTL4::CommandBuffer*> pool;
    size_t capacity;

    size_t head = 0;
    size_t tail = 0;
    size_t count;

    std::mutex mtx;
    std::condition_variable cv;
};

