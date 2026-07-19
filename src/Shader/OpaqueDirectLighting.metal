#include "ShaderTypes.h"

struct OpaqueVertexOut {
    float4 position [[position]];
};

vertex OpaqueVertexOut opaqueDirect_vertex(
    uint vertexID [[vertex_id]],
    constant OpaqueDirectLightingBindlessArguments& args [[buffer(OpaqueDirectLightingBufferIndex::BindlessArguments)]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]])
{
    OpaqueVertexOut out;
    float3 pos = args.vertices[vertexID].position;
    out.position = frame.viewProjection * float4(pos, 1.0);
    return out;
}

fragment float4 opaqueDirect_fragment(OpaqueVertexOut in [[stage_in]])
{
    return float4(1.0, 1.0, 1.0, 1.0);
}