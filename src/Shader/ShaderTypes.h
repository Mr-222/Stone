#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

struct FrameUniform {
    float4x4 viewProjection;
};

struct GPURenderPrimitive {
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

struct GPURenderPrimitive {
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

struct ObjectCullingParams {
    uint32_t primitiveCount;
};

struct IndexBufferInfo {
    uint64_t addr;
};

struct Visibility {
    uint32_t primitiveID;
};

enum class ObjectCullingBufferIndex {
    ExecutionRange,
    CullingParams,
    IndexBufferInfo,
    RenderPrimitives,
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

enum class OpaqueDirectLightingBufferIndex {
    BindlessArguments,
    FrameUniform,
    MaxBufferBindCount,
};

enum class OpaqueDirectLightingBindlessArgumentID {
    Vertices,
    MaxArgumentID,
};

#ifdef __METAL_VERSION__
struct GPUVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
    packed_float3 tangent;
};
#else
struct GPUVertex {
    float position[3];
    float normal[3];
    float uv[2];
    float tangent[3];
};
#endif

#ifdef __METAL_VERSION__
struct ObjectCullingICBContainer {
    command_buffer commandBuffer [[id(ObjectCullingICBArgumentID::CommandBuffer)]];
};

struct TriangleBindlessArguments {
    const device float4* positions [[id(TriangleBindlessArgumentID::Positions)]];
    const device float4* colors [[id(TriangleBindlessArgumentID::Colors)]];
};

struct OpaqueDirectLightingBindlessArguments {
    const device GPUVertex* vertices [[id(OpaqueDirectLightingBindlessArgumentID::Vertices)]];
};
#endif
