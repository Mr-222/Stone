#include "SceneAccelerationStructure.h"

#include <algorithm>
#include <Metal/MTL4AccelerationStructure.hpp>

#include "Core/CommandBuffer.h"
#include "Scene.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"

namespace {
size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
}

void SceneAccelerationStructure::Build(MTL::Device* device, const Scene& scene, const Buffer& opaqueVertexBuffer, const Buffer& opaqueIndexBuffer, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue) {
    LOG_ERROR_IF(!device, "Failed to build acceleration structure: device is null.");
    LOG_ERROR_IF(!queue, "Failed to build acceleration structure: command queue is null.");
    LOG_ERROR_IF(!device->supportsRaytracing(), "Device does not support hardware raytracing.");
    LOG_ERROR_IF(!device->supportsRaytracingFromRender(), "Device does not support raytracing from render shaders.");

    m_blas.clear();
    m_tlas.reset();
    m_instanceDescriptorBuffer.reset();
    m_residencySet.reset();

    // Build one BLAS per mesh (containing all opaque/masked submeshes of that mesh as geometryDescriptors)
    const size_t meshCount = scene.opaqueMeshes.size();
    m_blas.resize(meshCount);

    std::vector<MTL::AccelerationStructure*> builtBlasList;
    std::vector<MTL4::PrimitiveAccelerationStructureDescriptor*> blasDescs;
    std::vector<size_t> scratchOffsets;
    std::vector<size_t> scratchSizes;
    builtBlasList.reserve(meshCount);
    blasDescs.reserve(meshCount);
    scratchOffsets.reserve(meshCount);
    scratchSizes.reserve(meshCount);

    constexpr size_t kScratchAlignment = 256;
    size_t totalBlasScratchSize = 0;

    const uint64_t vertexBaseAddr = opaqueVertexBuffer.GetGPUAddress();
    const uint64_t indexBaseAddr = opaqueIndexBuffer.GetGPUAddress();
    const size_t totalOpaqueVertexBytes = scene.opaqueVertices.size() * sizeof(Vertex);

    for (size_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
        const SubMeshRange& meshRange = scene.opaqueMeshes[meshIndex];
        if (meshRange.submeshCount == 0)
            continue;

        std::vector<const NS::Object*> geomObjs;
        geomObjs.reserve(meshRange.submeshCount);
        for (uint32_t g = 0; g < meshRange.submeshCount; ++g) {
            const SubMesh& submesh = scene.opaqueSubmeshes[meshRange.firstSubmesh + g];
            const SceneMaterial& material = scene.materials[submesh.materialIndex];
            const bool isOpaque = (material.flags & kMaterialFlagAlphaMask) == 0;

            auto* geomDesc = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init()->autorelease();
            const uint64_t vertexOffsetBytes = static_cast<uint64_t>(submesh.vertexOffset) * sizeof(Vertex);
            const uint64_t vertexLengthBytes = totalOpaqueVertexBytes - vertexOffsetBytes;
            geomDesc->setVertexBuffer(MTL4::BufferRange::Make(vertexBaseAddr + vertexOffsetBytes, vertexLengthBytes));
            geomDesc->setVertexFormat(MTL::AttributeFormatFloat3);
            geomDesc->setVertexStride(sizeof(Vertex));

            const uint64_t indexOffsetBytes = static_cast<uint64_t>(submesh.firstIndex) * sizeof(uint32_t);
            const uint64_t indexLengthBytes = static_cast<uint64_t>(submesh.indexCount) * sizeof(uint32_t);
            geomDesc->setIndexBuffer(MTL4::BufferRange::Make(indexBaseAddr + indexOffsetBytes, indexLengthBytes));
            geomDesc->setIndexType(MTL::IndexTypeUInt32);
            geomDesc->setTriangleCount(submesh.indexCount / 3);
            geomDesc->setOpaque(isOpaque);
            geomDesc->setAllowDuplicateIntersectionFunctionInvocation(true);
            geomObjs.push_back(geomDesc);
        }

        auto* blasDesc = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init()->autorelease();
        blasDesc->setGeometryDescriptors(NS::Array::array(geomObjs.data(), geomObjs.size()));

        const MTL::AccelerationStructureSizes sizes = device->accelerationStructureSizes(blasDesc);

        MTL::AccelerationStructure* blas = device->newAccelerationStructure(sizes.accelerationStructureSize);
        LOG_ERROR_IF(!blas, "Failed to allocate BLAS for mesh {}.", meshIndex);
        blas->setLabel(NS::String::string(("BLAS_Mesh_" + std::to_string(meshIndex)).c_str(), NS::UTF8StringEncoding));

        m_blas[meshIndex] = NS::TransferPtr(blas);
        builtBlasList.push_back(blas);
        blasDescs.push_back(blasDesc);

        totalBlasScratchSize = AlignUp(totalBlasScratchSize, kScratchAlignment);
        scratchOffsets.push_back(totalBlasScratchSize);
        scratchSizes.push_back(sizes.buildScratchBufferSize);
        totalBlasScratchSize += sizes.buildScratchBufferSize;
    }

    if (!builtBlasList.empty()) {
        Buffer blasScratchBuffer(device, std::max<size_t>(totalBlasScratchSize, kScratchAlignment), MTL::ResourceStorageModePrivate);
        blasScratchBuffer.GetNative()->setLabel(NS::String::string("BLAS Scratch Buffer", NS::UTF8StringEncoding));

        CommandBuffer cmd = commandBufferPool.AcquireFlushGPU();
        cmd.AddResource(opaqueVertexBuffer.GetNative());
        cmd.AddResource(opaqueIndexBuffer.GetNative());
        cmd.AddResource(blasScratchBuffer.GetNative());
        for (MTL::AccelerationStructure* blas : builtBlasList)
            cmd.AddResource(blas);

        MTL4::ComputeCommandEncoder* encoder = cmd.BeginComputePass();
        for (size_t i = 0; i < builtBlasList.size(); ++i) {
            const MTL4::BufferRange scratchRange = MTL4::BufferRange::Make(blasScratchBuffer.GetGPUAddress() + scratchOffsets[i], scratchSizes[i]);
            encoder->buildAccelerationStructure(builtBlasList[i], blasDescs[i], scratchRange);
        }
        encoder->endEncoding();
        cmd.SubmitTo(queue);
    }

    // Build TLAS with one instance per RenderObject (userID = firstSubmesh of the object's mesh)
    std::vector<MTL::IndirectAccelerationStructureInstanceDescriptor> instances;
    instances.reserve(scene.objects.size());
    for (const RenderObject& object : scene.objects) {
        const SubMeshRange& meshRange = scene.opaqueMeshes[object.meshIndex];
        if (meshRange.submeshCount == 0 || !m_blas[object.meshIndex])
            continue;

        const glm::mat4& m = object.transform;
        MTL::PackedFloat4x3 transform;
        for (int col = 0; col < 4; ++col)
            transform.columns[col] = MTL::PackedFloat3(m[col][0], m[col][1], m[col][2]);

        MTL::IndirectAccelerationStructureInstanceDescriptor instance{};
        instance.transformationMatrix = transform;
        instance.options = MTL::AccelerationStructureInstanceOptionNone;
        instance.mask = 0xFF;
        instance.intersectionFunctionTableOffset = 0;
        instance.userID = meshRange.firstSubmesh;
        instance.accelerationStructureID = m_blas[object.meshIndex]->gpuResourceID();
        instances.push_back(instance);
    }

    const size_t instanceBufferSize = std::max(instances.size(), size_t(1)) * sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor);
    m_instanceDescriptorBuffer = std::make_unique<Buffer>(device, instanceBufferSize, MTL::ResourceStorageModePrivate);
    m_instanceDescriptorBuffer->GetNative()->setLabel(NS::String::string("TLAS Instance Descriptor Buffer", NS::UTF8StringEncoding));
    if (!instances.empty())
        m_instanceDescriptorBuffer->UpdateStaged(instances.data(), instances.size() * sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor), 0, commandBufferPool, queue);

    auto* tlasDesc = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init()->autorelease();
    tlasDesc->setInstanceCount(instances.size());
    tlasDesc->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    tlasDesc->setInstanceDescriptorStride(sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
    tlasDesc->setInstanceDescriptorBuffer(MTL4::BufferRange::Make(m_instanceDescriptorBuffer->GetGPUAddress(), m_instanceDescriptorBuffer->GetSize()));

    const MTL::AccelerationStructureSizes tlasSizes = device->accelerationStructureSizes(tlasDesc);
    MTL::AccelerationStructure* tlas = device->newAccelerationStructure(tlasSizes.accelerationStructureSize);
    LOG_ERROR_IF(!tlas, "Failed to allocate TLAS.");
    tlas->setLabel(NS::String::string("Scene TLAS", NS::UTF8StringEncoding));
    m_tlas = NS::TransferPtr(tlas);

    Buffer tlasScratchBuffer(device, std::max<size_t>(tlasSizes.buildScratchBufferSize, kScratchAlignment), MTL::ResourceStorageModePrivate);
    tlasScratchBuffer.GetNative()->setLabel(NS::String::string("TLAS Scratch Buffer", NS::UTF8StringEncoding));

    CommandBuffer tlasCmd = commandBufferPool.AcquireFlushGPU();
    tlasCmd.AddResource(m_tlas.get());
    tlasCmd.AddResource(m_instanceDescriptorBuffer->GetNative());
    tlasCmd.AddResource(tlasScratchBuffer.GetNative());
    for (MTL::AccelerationStructure* blas : builtBlasList)
        tlasCmd.AddResource(blas);

    MTL4::ComputeCommandEncoder* tlasEncoder = tlasCmd.BeginComputePass();
    tlasEncoder->buildAccelerationStructure(m_tlas.get(), tlasDesc, MTL4::BufferRange::Make(tlasScratchBuffer.GetGPUAddress(), tlasSizes.buildScratchBufferSize));
    tlasEncoder->endEncoding();
    tlasCmd.SubmitTo(queue);

    // Build persistent residency set for TLAS, instance buffer, and all mesh BLASes
    auto* resDesc = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
    resDesc->setInitialCapacity(static_cast<NS::UInteger>(builtBlasList.size() + 2));
    NS::Error* error = nullptr;
    MTL::ResidencySet* residencySet = device->newResidencySet(resDesc, &error);
    LOG_ERROR_IF(!residencySet, "Failed to create acceleration structure residency set: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    residencySet->addAllocation(m_tlas.get());
    residencySet->addAllocation(m_instanceDescriptorBuffer->GetNative()); // for rebuild / refit during runtime

    for (MTL::AccelerationStructure* blas : builtBlasList)
        residencySet->addAllocation(blas);
    residencySet->commit();
    residencySet->requestResidency();
    m_residencySet = NS::TransferPtr(residencySet);

    LOG_INFO("Built scene acceleration structure: {} mesh BLASes, {} object TLAS instances.", builtBlasList.size(), instances.size());
}
