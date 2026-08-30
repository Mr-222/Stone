#include "ShaderTypes.h"

struct CompositeVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex CompositeVertexOut transparentComposite_vertex(uint vertexID [[vertex_id]]) {
    CompositeVertexOut out;
    // Fullscreen triangle: 3 vertices cover the entire screen
    out.uv = float2((vertexID << 1) & 2, vertexID & 2);
    out.position = float4(out.uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    return out;
}

fragment float4 transparentComposite_fragment(
    CompositeVertexOut in [[stage_in]],
    device TransparentCompositeFragmentArguments& args [[buffer(TransparentCompositeBufferIndex::FragmentArguments)]])
{
    constexpr sampler s(coord::normalized, filter::nearest);
    
    float4 accum = args.accumTexture.sample(s, in.uv);
    float reveal = args.revealTexture.sample(s, in.uv).r;

    // No transparent fragments contributed to this pixel
    if (accum.a <= 1e-5f)
        discard_fragment();

    float3 averageColor = accum.rgb / max(accum.a, 1e-5f);

    // (1 - reveal) gives the total accumulated opacity
    float transparency = 1.0f - reveal;

    return float4(averageColor, transparency);
}
