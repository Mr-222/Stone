#include "ShaderTypes.h"

struct OpaqueVertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 worldNormal;
    float2 uv;
    uint materialIndex [[flat]];
};

constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float InvPI = 1.0f / PI;

float D_GGX(float NoH, float roughness) {
    const float roughnessSquared = roughness * roughness;
    const float f = (NoH * roughnessSquared - NoH) * NoH + 1.0f;
    return roughnessSquared / max(PI * f * f, 1e-7f);
}

float3 F_Schlick(float u, float3 f0) {
    const float f = pow(1.0f - u, 5.0f);
    return f0 + (1.0f - f0) * f;
}

float V_SmithGGXCorrelatedFast(float NoV, float NoL, float roughness) {
    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

float Fd_Lambert() {
    return InvPI;
}

float3 SafeNormalize(float3 value, float3 fallback) {
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-10f ? normalize(value) : fallback;
}

float3 EvaluateBRDF(
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

    // Convert perceptually linear roughness to the GGX alpha parameter.
    const float roughness = perceptualRoughness * perceptualRoughness;
    const float D = D_GGX(NoH, roughness);
    const float3 F = F_Schlick(LoH, f0);
    const float V = V_SmithGGXCorrelatedFast(NoV, NoL, roughness);

    const float3 Fr = (D * V) * F;
    const float3 Fd = diffuseColor * Fd_Lambert();
    return Fd + Fr;
}

vertex OpaqueVertexOut opaqueDirect_vertex(
    uint vertexID [[vertex_id]],
    uint primitiveID [[instance_id]],
    constant OpaqueDirectLightingVertexArguments& args [[buffer(OpaqueDirectLightingBufferIndex::VertexArguments)]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]])
{
    OpaqueVertexOut out;
    const device GPUVertex& gpuVertex = args.vertices[vertexID];
    const device GPURenderPrimitive& primitive = args.renderPrimitives[primitiveID];
    const float4 worldPosition = primitive.worldMat * float4(gpuVertex.position, 1.0f);
    out.position = frame.viewProjection * worldPosition;
    out.worldPosition = worldPosition.xyz;
    out.worldNormal = (primitive.worldNormalMat * float4(gpuVertex.normal, 0.0f)).xyz;
    out.uv = gpuVertex.uv;
    out.materialIndex = primitive.materialIndex;
    return out;
}

fragment float4 opaqueDirect_fragment(
    OpaqueVertexOut in [[stage_in]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    constexpr sampler baseColorSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        mip_filter::linear);

    const device GPUMaterial& material = args.materials[in.materialIndex];

    const float4 baseColorSample = args.textures[material.baseColorTextureIndex].sample(baseColorSampler, in.uv) * material.baseColorFactor;
    const float4 metallicRoughnessSample = args.textures[material.metallicRoughnessTextureIndex].sample(baseColorSampler, in.uv);

    const float3 baseColor = baseColorSample.rgb;
    const float perceptualRoughness = clamp(metallicRoughnessSample.g * material.roughnessFactor, 0.045f, 1.0f);
    const float metallic = saturate(metallicRoughnessSample.b * material.metallicFactor);
    const float3 diffuseColor = baseColor * (1.0f - metallic);
    const float3 f0 = mix(float3(0.04f), baseColor, metallic);

    float3 n = SafeNormalize(in.worldNormal, float3(0.0f, 1.0f, 0.0f));

    const float3 v = SafeNormalize(frame.cameraPosition.xyz - in.worldPosition, n);
    const device GPULightListInfo& lightListInfo = args.lightListInfo;
    const float3 ambientColor = max(lightListInfo.ambientColorAndIntensity.rgb, 0.0f);
    const float ambientIntensity = max(lightListInfo.ambientColorAndIntensity.w, 0.0f);
    float3 luminance = baseColor * ambientColor * ambientIntensity;

    for (uint lightIndex = 0; lightIndex < lightListInfo.directionalLightCount; ++lightIndex) {
        const device GPUDirectionalLight& light = args.directionalLights[lightIndex];

        const float3 l = -normalize(light.direction.xyz);
        const float NoL = saturate(dot(n, l));
        if (NoL <= 0.0f)
            continue;

        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        const float3 lightColor = max(light.colorAndIlluminance.rgb, 0.0f);
        const float lightIlluminance = max(light.colorAndIlluminance.w, 0.0f);

        luminance += brdf * lightColor * (lightIlluminance * NoL);
    }

    return float4(luminance, 1.0f);
}