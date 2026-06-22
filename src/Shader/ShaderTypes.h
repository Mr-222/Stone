#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

struct FrameUniform {
    float4x4 viewProjection;
};

struct GPURenderObject {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t pad0;
    float4x4 worldMat;
};
#else
#include <cstdint>
#include <glm/glm.hpp>

struct FrameUniform {
    glm::mat4 viewProjection;
};

struct GPURenderObject {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t pad0;
    glm::mat4 worldMat;
};
#endif

struct IndirectCommandBufferExecutionRange {
    uint32_t location;
#ifdef __METAL_VERSION__
    atomic_uint length;
#else
    uint32_t length;
#endif
};

struct IndexBufferInfo {
    uint64_t addr;
};

struct Visibility {
    uint32_t objID;
};

enum class ObjectCullingBufferIndex {
    ExecutionRange,
    IndexBufferInfo,
    RenderObjects,
    Visibilities,
    ICBContainer,
    MaxBufferBindCount,
};

enum class ObjectCullingICBArgumentID {
    CommandBuffer,
    MaxArgumentID,
};

enum class TriangleBufferIndex {
    BindlessArguments,
    FrameUniform,
    MaxBufferBindCount,
};

enum class TriangleBindlessArgumentID {
    Positions,
    Colors,
    MaxArgumentID,
};

#ifdef __METAL_VERSION__
struct ObjectCullingICBContainer {
    command_buffer commandBuffer [[id(ObjectCullingICBArgumentID::CommandBuffer)]];
};

struct TriangleBindlessArguments {
    const device float4* positions [[id(TriangleBindlessArgumentID::Positions)]];
    const device float4* colors [[id(TriangleBindlessArgumentID::Colors)]];
};
#endif
