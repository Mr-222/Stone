#include "ShaderTypes.h"

float3 ACES_EDR(float3 colorLinear, float edrHeadroom) {
    constexpr float3x3 ACESInputMat = transpose(float3x3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777
    ));
    constexpr float3x3 ACESOutputMat = transpose(float3x3(
         1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602
    ));

    float3 x = max(colorLinear, 0.0f) / edrHeadroom; // normalize based on Headroom
    float3 v = ACESInputMat * x;
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;

    // rescaling to [0, edrHeadroom]
    float3 tonemapped = (ACESOutputMat * (a / b)) * edrHeadroom;
    return max(tonemapped, 0.0f);
}

struct TonemappingVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex TonemappingVertexOut tonemapping_vertex(uint vertexID [[vertex_id]]) {
    TonemappingVertexOut out;
    // Fullscreen triangle: 3 vertices cover the entire screen
    constexpr float2 positions[3] = {
        float2(-1.0f,  1.0f), // Top-left
        float2( 3.0f,  1.0f), // Far top-right
        float2(-1.0f, -3.0f)  // Far bottom-left
    };
    constexpr float2 uvs[3] = {
        float2(0.0f, 0.0f),
        float2(2.0f, 0.0f),
        float2(0.0f, 2.0f)
    };
    out.position = float4(positions[vertexID], 0.0f, 1.0f);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 tonemapping_fragment(
    TonemappingVertexOut in [[stage_in]],
    device TonemappingFragmentArguments& args [[buffer(TonemappingBufferIndex::FragmentArguments)]])
{
    constexpr sampler s(coord::normalized, filter::nearest);
    float4 sceneColor = args.sceneColorTexture.sample(s, in.uv);
    float headroom = max(args.edrHeadroom, 1.0f);
    float3 tonemapped = ACES_EDR(sceneColor.rgb, headroom);
    return float4(tonemapped, 1.0f);
}
