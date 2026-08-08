#include "ShaderTypes.h"

struct OpaqueVertexOut {
    float4 position [[position]];
    float2 uv;
    uint materialIndex [[flat]];
};

vertex OpaqueVertexOut opaqueDirect_vertex(
    uint vertexID [[vertex_id]],
    uint primitiveID [[instance_id]],
    constant OpaqueDirectLightingVertexArguments& args [[buffer(OpaqueDirectLightingBufferIndex::VertexArguments)]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]])
{
    OpaqueVertexOut out;
    const device GPUVertex& gpuVertex = args.vertices[vertexID];
    const device GPURenderPrimitive& primitive = args.renderPrimitives[primitiveID];
    out.position = frame.viewProjection * primitive.worldMat * float4(gpuVertex.position, 1.0);
    out.uv = gpuVertex.uv;
    out.materialIndex = primitive.materialIndex;
    return out;
}

fragment float4 opaqueDirect_fragment(
    OpaqueVertexOut in [[stage_in]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    constexpr sampler baseColorSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        mip_filter::linear);

    const device GPUMaterial& material = args.materials[in.materialIndex];
    return args.textures[material.baseColorTextureIndex].sample(baseColorSampler, in.uv) * material.baseColorFactor;
}