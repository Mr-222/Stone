#pragma once

#include <metal_stdlib>
using namespace metal;

constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float InvPI = 1.0f / PI;

inline float D_GGX(float NoH, float roughness) {
    const float roughnessSquared = roughness * roughness;
    const float f = (NoH * roughnessSquared - NoH) * NoH + 1.0f;
    return roughnessSquared / max(PI * f * f, 1e-7f);
}

inline float3 F_Schlick(float u, float3 f0) {
    const float f = pow(1.0f - u, 5.0f);
    return f0 + (1.0f - f0) * f;
}

inline float V_SmithGGXCorrelatedFast(float NoV, float NoL, float roughness) {
    const float a2 = roughness * roughness;
    const float GGXV = NoL * sqrt(NoV * NoV * (1.0f - a2) + a2);
    const float GGXL = NoV * sqrt(NoL * NoL * (1.0f - a2) + a2);
    return 0.5f / (GGXV + GGXL);
}

inline float Fd_Lambert() {
    return InvPI;
}

inline float3 SafeNormalize(float3 value, float3 fallback) {
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-10f ? normalize(value) : fallback;
}

inline float3 EvaluateBRDF(
    float3 n,
    float3 v,
    float3 l,
    float3 diffuseColor,
    float3 f0,
    float perceptualRoughness,
    float NoL)
{
    const float3 h = SafeNormalize(v + l, n);
    const float NoV = abs(dot(n, v)) + 1e-5f;
    const float NoH = saturate(dot(n, h));
    const float LoH = saturate(dot(l, h));

    const float roughness = perceptualRoughness * perceptualRoughness;
    const float D = D_GGX(NoH, roughness);
    const float3 F = F_Schlick(LoH, f0);
    const float V = V_SmithGGXCorrelatedFast(NoV, NoL, roughness);

    const float3 Fr = (D * V) * F;
    const float3 Fd = diffuseColor * Fd_Lambert();
    return Fd + Fr;
}
