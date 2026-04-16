#include "Heap.h"

Heap::Heap(MTL::Device* device, size_t size, MTL::StorageMode mode) {
    MTL::HeapDescriptor* heapDesc = MTL::HeapDescriptor::alloc()->init();
    heapDesc->setSize(size);
    heapDesc->setStorageMode(mode);
    heapDesc->setType(MTL::HeapTypeAutomatic);

    m_heap = device->newHeap(heapDesc);
    heapDesc->release();
}

Heap::~Heap() {
    if (m_heap)
        m_heap->release();
}