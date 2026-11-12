#pragma once

#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "BRDF.h"

struct Surface {
    float3 baseColor;
    float alpha;
    float3 n;
    float perceptualRoughness;
    float metallic;
};

inline Surface SampleSurface(const device GPUMaterial& material, const device array<texture2d<float>, kMaxBindlessTextureCount>& textures, sampler texSampler, float2 uv, float3 worldNormal, float4 worldTangent) {
    const float4 baseColorSample = textures[material.baseColorTextureIndex].sample(texSampler, uv) * material.baseColorFactor;
    const float4 metallicRoughnessSample = textures[material.metallicRoughnessTextureIndex].sample(texSampler, uv);

    float3 n = SafeNormalize(worldNormal, float3(0.0f, 1.0f, 0.0f));
    if (material.normalTextureIndex != 0) {
        const float3 t = SafeNormalize(worldTangent.xyz - n * dot(n, worldTangent.xyz), float3(0.0f));
        const float3 b = cross(n, t) * worldTangent.w;
        const float3x3 tbn = float3x3(t, b, n);
        const float3 nTS = (textures[material.normalTextureIndex].sample(texSampler, uv).rgb * 2.0f - 1.0f) * float3(material.normalScale, material.normalScale, 1.0f);
        n = SafeNormalize(tbn * nTS, n);
    }

    Surface surface;
    surface.baseColor = baseColorSample.rgb;
    surface.alpha = baseColorSample.a;
    surface.n = n;
    surface.perceptualRoughness = clamp(metallicRoughnessSample.g * material.roughnessFactor, 0.045f, 1.0f);
    surface.metallic = saturate(metallicRoughnessSample.b * material.metallicFactor);
    return surface;
}
