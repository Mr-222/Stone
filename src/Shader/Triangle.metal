#include "ShaderTypes.h"

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(
    uint vertexID [[vertex_id]],
    constant TriangleBindlessArguments& args [[buffer(TriangleBufferIndex::BindlessArguments)]],
    constant FrameUniform& frame [[buffer(TriangleBufferIndex::FrameUniform)]])
{
    VertexOut out;
    out.position = frame.viewProjection * args.positions[vertexID];
    out.color = args.colors[vertexID];

    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]])
{
    return in.color;
}
