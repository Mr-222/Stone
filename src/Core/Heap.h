#pragma once

#include <Metal/Metal.hpp>

class Heap {
public:
    Heap(MTL::Device* device, size_t size, MTL::StorageMode mode);
    ~Heap();

    Heap(const Heap&) = delete;
    Heap& operator=(const Heap&) = delete;

    MTL::Heap* GetNative() const { return m_heap; }
    size_t GetUsedSize() const { return m_heap->usedSize(); }
    size_t GetSize() const { return m_heap->size(); }

private:
    MTL::Heap* m_heap;
};

