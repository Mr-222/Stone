#pragma once

#include <metal_stdlib>
using namespace metal;

constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float InvPI = 1.0f / PI;
constant float SUN_ANGULAR_RADIUS = 0.00465f; // Solar angular radius (0.267 degree -> radian)

inline float2 OctWrap(float2 v) {
    return (1.0f - abs(v.yx)) * select(float2(-1.0f), float2(1.0f), v >= 0.0f);
}

inline float2 EncodeOctahedralNormal(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0f ? n.xy : OctWrap(n.xy);
    return clamp(n.xy, -1.0f, 1.0f);
}

inline float3 DecodeOctahedralNormal(float2 e) {
    float3 n = float3(e.xy, 1.0f - abs(e.x) - abs(e.y));
    const float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0f);
    return normalize(n);
}

// Reconstructs world position from linear view-space Z (left-handed view, w = z); precise enough for ray-traced shadow origins.
inline float3 ReconstructWorldPositionFromViewZ(float2 pixelPos, uint2 screenSize, float viewZ, float4x4 view, float4x4 proj) {
    const float2 uv = pixelPos / float2(screenSize);
    const float2 ndcXY = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    const float3 viewPos = float3(ndcXY.x * viewZ / proj[0][0], ndcXY.y * viewZ / proj[1][1], viewZ);
    const float3x3 viewRotation = float3x3(view[0].xyz, view[1].xyz, view[2].xyz);
    return transpose(viewRotation) * (viewPos - view[3].xyz);
}

inline float ClipDepthToViewZ(float clipDepth, float4x4 proj) {
    return proj[3][2] / (clipDepth - proj[2][2]);
}

constexpr float ConstexprSqrt(float x) {
    float curr = x >= 1.0f ? x : 1.0f, prev = 0.0f;
    while (curr != prev) { prev = curr; curr = 0.5f * (curr + x / curr); }
    return curr;
}

float RaySphereIntersect(float3 rayOrigin, float3 rayDir, float radius)
{
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    if (c >= 0.0f)
    {
        if (b >= 0.0f) return -1.0f;
        float d = b * b - c;
        if (d < 0.0f) return -1.0f;
        float t = -b - sqrt(d);
        return t > 0.0f ? t : -1.0f;
    }
    float d = b * b - c;
    return -b + sqrt(max(0.0f, d));
}

float3 GetUniformSphereDirection(uint sampleIdx, uint totalSamples)
{
    float phi = float(sampleIdx) * 2.39996323f; // golden angle
    float z   = 1.0f - (2.0f * float(sampleIdx) + 1.0f) / float(totalSamples);
    float sinTheta = sqrt(max(0.0f, 1.0f - z * z));
    return float3(sinTheta * cos(phi), z, sinTheta * sin(phi));
}

// Atmospheric scattering coefficients
constant constexpr float R_ground = 6360; // km
constant constexpr float R_top    = 6460;
constant constexpr float H        = ConstexprSqrt(R_top * R_top - R_ground * R_ground); // Maximum Geometric Horizon Distance