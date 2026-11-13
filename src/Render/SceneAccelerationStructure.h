#pragma once

#include <memory>
#include <vector>
#include <Metal/Metal.hpp>

#include "Core/Buffer.h"
#include "Core/CommandBufferPool.h"

class Scene;

class SceneAccelerationStructure {
public:
    SceneAccelerationStructure() = default;
    ~SceneAccelerationStructure() = default;

    void Build(MTL::Device* device, const Scene& scene, const Buffer& opaqueVertexBuffer, const Buffer& opaqueIndexBuffer, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue);

    MTL::AccelerationStructure* GetTLAS() const { return m_tlas.get(); }
    MTL::ResidencySet* GetResidencySet() const { return m_residencySet.get(); }

private:
    std::vector<NS::SharedPtr<MTL::AccelerationStructure>> m_blas;
    NS::SharedPtr<MTL::AccelerationStructure> m_tlas;
    std::unique_ptr<Buffer> m_instanceDescriptorBuffer;
    NS::SharedPtr<MTL::ResidencySet> m_residencySet;
};
