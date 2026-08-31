#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

constant constexpr uint32_t kMaxBindlessTextureCount = 1024;

struct FrameUniform {
    float4x4 viewProjection;
    float4 cameraPosition;
};

struct GPULightListInfo {
    uint32_t directionalLightCount;
    uint32_t pointLightCount;
    uint32_t spotLightCount;
    uint32_t pad0;
    float4 ambientColorAndIntensity;
};

struct GPUDirectionalLight {
    float4 direction;
    float4 colorAndIlluminance;
};

struct GPUPointLight {
    float4 positionAndRange;     // xyz = world position, w = range
    float4 colorAndIntensity;    // rgb = color, w = luminous intensity (candela)
};

struct GPUSpotLight {
    float4 positionAndRange;     // xyz = world position, w = range
    float4 direction;            // xyz = direction (normalized), w = unused
    float4 colorAndIntensity;    // rgb = color, w = luminous intensity (candela)
    float4 scaleOffset;          // x = angleScale, y = angleOffset, zw = unused
};

struct GPURenderPrimitive {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialIndex;
    float4x4 worldMat;
    float4x4 worldNormalMat;
};

struct GPUMaterial {
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    uint32_t baseColorTextureIndex;
    uint32_t metallicRoughnessTextureIndex;
};
#else
#include <cstdint>
#include <glm/glm.hpp>

constexpr uint32_t kMaxBindlessTextureCount = 1024;

struct FrameUniform {
    glm::mat4 viewProjection;
    glm::vec4 cameraPosition;
};

struct GPULightListInfo {
    uint32_t directionalLightCount;
    uint32_t pointLightCount;
    uint32_t spotLightCount;
    uint32_t pad0;
    glm::vec4 ambientColorAndIntensity;
};

struct GPUDirectionalLight {
    glm::vec4 direction;
    glm::vec4 colorAndIlluminance;
};

struct GPUPointLight {
    glm::vec4 positionAndRange;     // xyz = world position, w = range
    glm::vec4 colorAndIntensity;    // rgb = color, w = luminous intensity (candela)
};

struct GPUSpotLight {
    glm::vec4 positionAndRange;     // xyz = world position, w = range
    glm::vec4 direction;            // xyz = direction (normalized), w = unused
    glm::vec4 colorAndIntensity;    // rgb = color, w = luminous intensity (candela)
    glm::vec4 scaleOffset;          // x = angleScale, y = angleOffset, zw = unused
};

struct GPURenderPrimitive {
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t materialIndex;
    glm::mat4 worldMat;
    glm::mat4 worldNormalMat;
};

struct GPUMaterial {
    glm::vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    uint32_t baseColorTextureIndex;
    uint32_t metallicRoughnessTextureIndex;
};

static_assert(sizeof(FrameUniform) == 80);
static_assert(sizeof(GPULightListInfo) == 32);
static_assert(sizeof(GPUDirectionalLight) == 32);
static_assert(sizeof(GPUPointLight) == 32);
static_assert(sizeof(GPUSpotLight) == 64);
static_assert(sizeof(GPURenderPrimitive) == 144);
static_assert(sizeof(GPUMaterial) == 32);
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
    LightListInfo,
    DirectionalLights,
    PointLights,
    SpotLights,
    Textures,
    MaxArgumentID = 5 + kMaxBindlessTextureCount,
};

enum class TransparentDirectLightingBufferIndex {
    VertexArguments,
    FrameUniform,
    FragmentArguments,
    MaxBufferBindCount,
};

enum class TransparentDirectLightingVertexArgumentID {
    Vertices,
    RenderPrimitives,
    MaxArgumentID,
};

enum class TransparentDirectLightingFragmentArgumentID {
    Materials,
    LightListInfo,
    DirectionalLights,
    PointLights,
    SpotLights,
    Textures,
    MaxArgumentID = 5 + kMaxBindlessTextureCount,
};

enum class TransparentCompositeBufferIndex {
    FragmentArguments,
    MaxBufferBindCount,
};

enum class TransparentCompositeFragmentArgumentID {
    AccumTexture,
    RevealTexture,
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

struct OpaqueDirectLightingVertexArguments {
    const device GPUVertex* vertices;
    const device GPURenderPrimitive* renderPrimitives;
};

struct OpaqueDirectLightingFragmentArguments {
    const device GPUMaterial* materials [[id(OpaqueDirectLightingFragmentArgumentID::Materials)]];
    const device GPULightListInfo& lightListInfo [[id(OpaqueDirectLightingFragmentArgumentID::LightListInfo)]];
    const device GPUDirectionalLight* directionalLights [[id(OpaqueDirectLightingFragmentArgumentID::DirectionalLights)]];
    const device GPUPointLight* pointLights [[id(OpaqueDirectLightingFragmentArgumentID::PointLights)]];
    const device GPUSpotLight* spotLights [[id(OpaqueDirectLightingFragmentArgumentID::SpotLights)]];
    const array<texture2d<float>, kMaxBindlessTextureCount> textures [[id(OpaqueDirectLightingFragmentArgumentID::Textures)]];
};


struct TransparentDirectLightingVertexArguments {
    const device GPUVertex* vertices;
    const device GPURenderPrimitive* renderPrimitives;
};

struct TransparentDirectLightingFragmentArguments {
    const device GPUMaterial* materials [[id(TransparentDirectLightingFragmentArgumentID::Materials)]];
    const device GPULightListInfo& lightListInfo [[id(TransparentDirectLightingFragmentArgumentID::LightListInfo)]];
    const device GPUDirectionalLight* directionalLights [[id(TransparentDirectLightingFragmentArgumentID::DirectionalLights)]];
    const device GPUPointLight* pointLights [[id(TransparentDirectLightingFragmentArgumentID::PointLights)]];
    const device GPUSpotLight* spotLights [[id(TransparentDirectLightingFragmentArgumentID::SpotLights)]];
    const array<texture2d<float>, kMaxBindlessTextureCount> textures [[id(TransparentDirectLightingFragmentArgumentID::Textures)]];
};

struct TransparentCompositeFragmentArguments {
    texture2d<float> accumTexture [[id(TransparentCompositeFragmentArgumentID::AccumTexture)]];
    texture2d<float> revealTexture [[id(TransparentCompositeFragmentArgumentID::RevealTexture)]];
};


#endif
