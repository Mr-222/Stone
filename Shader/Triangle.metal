#include <metal_stdlib>
using namespace metal;

struct BindlessArguments {
    const device float4* positions [[id(0)]];
    const device float4* colors    [[id(1)]];
};

struct FrameUniform {
    float4x4 viewProjection;
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertex_main(uint vertexID [[vertex_id]], constant BindlessArguments& args [[buffer(0)]], constant FrameUniform& frame [[buffer(1)]]) {
    VertexOut out;
    out.position = frame.viewProjection * args.positions[vertexID];
    out.color = args.colors[vertexID];

    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
