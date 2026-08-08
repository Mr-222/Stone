#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

constant constexpr uint32_t kMaxBindlessTextureCount = 1024;

struct FrameUniform {
    float4x4 viewProjection;
};

struct GPURenderPrimitive {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialIndex;
    float4x4 worldMat;
};

struct GPUMaterial {
    float4 baseColorFactor;
    uint32_t baseColorTextureIndex;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};
#else
#include <cstdint>
#include <glm/glm.hpp>

constexpr uint32_t kMaxBindlessTextureCount = 1024;

struct FrameUniform {
    glm::mat4 viewProjection;
};

struct GPURenderPrimitive {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialIndex;
    glm::mat4 worldMat;
};

struct GPUMaterial {
    glm::vec4 baseColorFactor;
    uint32_t baseColorTextureIndex;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
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
    VertexArguments,
    FrameUniform,
    FragmentArguments,
    MaxBufferBindCount,
};

enum class OpaqueDirectLightingVertexArgumentID {
    Vertices,
    RenderPrimitives,
    MaxArgumentID,
};

enum class OpaqueDirectLightingFragmentArgumentID {
    Materials,
    Textures,
    MaxArgumentID = 1 + kMaxBindlessTextureCount,
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

struct OpaqueDirectLightingVertexArguments {
    const device GPUVertex* vertices;
    const device GPURenderPrimitive* renderPrimitives;
};

struct OpaqueDirectLightingFragmentArguments {
    const device GPUMaterial* materials [[id(OpaqueDirectLightingFragmentArgumentID::Materials)]];
    array<texture2d<float>, kMaxBindlessTextureCount> textures [[id(OpaqueDirectLightingFragmentArgumentID::Textures)]];
};

#endif
