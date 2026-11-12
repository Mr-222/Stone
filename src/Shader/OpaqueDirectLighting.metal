#include "ShaderTypes.h"
#include "BRDF.h"
#include "Lighting.h"
#include "Surface.h"

struct OpaqueVertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 worldNormal;
    float4 worldTangent;
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
    const float4 worldPosition = primitive.worldMat * float4(gpuVertex.position, 1.0f);
    out.position = frame.viewProjection * worldPosition;
    out.worldPosition = worldPosition.xyz;
    out.worldNormal = (primitive.worldNormalMat * float4(gpuVertex.normal, 0.0f)).xyz;
    out.worldTangent = float4((primitive.worldMat * float4(gpuVertex.tangent.xyz, 0.0f)).xyz, gpuVertex.tangent.w);
    out.uv = gpuVertex.uv;
    out.materialIndex = primitive.materialIndex;
    return out;
}

template <bool kAlphaTest>
inline float4 ShadeOpaqueDirect(
    OpaqueVertexOut in,
    bool frontFacing,
    constant FrameUniform& frame,
    device OpaqueDirectLightingFragmentArguments& args)
{
    constexpr sampler baseColorSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear,
        mip_filter::linear);

    const device GPUMaterial& material = args.materials[in.materialIndex];
    const Surface surface = SampleSurface(material, args.textures, baseColorSampler, in.uv, in.worldNormal, in.worldTangent, frontFacing);
    if (kAlphaTest && (material.flags & kMaterialFlagAlphaMask) != 0 && surface.alpha < material.alphaCutoff)
        discard_fragment();

    const float3 baseColor = surface.baseColor;
    const float perceptualRoughness = surface.perceptualRoughness;
    const float metallic = surface.metallic;
    const float3 diffuseColor = baseColor * (1.0f - metallic);
    const float3 f0 = mix(float3(0.04f), baseColor, metallic);

    float3 n = surface.n;

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
        float3 lightColor = max(light.colorAndIlluminance.rgb, 0.0f);
        if (lightIndex == 0)
            lightColor *= frame.sunTransmittance.rgb;
        const float lightIlluminance = max(light.colorAndIlluminance.w, 0.0f);

        luminance += brdf * lightColor * (lightIlluminance * NoL);
    }

    for (uint lightIndex = 0; lightIndex < lightListInfo.pointLightCount; ++lightIndex) {
        const device GPUPointLight& light = args.pointLights[lightIndex];
        const float3 posToLight = light.positionAndRange.xyz - in.worldPosition;
        const float distSq = dot(posToLight, posToLight);
        const float3 l = SafeNormalize(posToLight, n);
        const float NoL = saturate(dot(n, l));
        if (NoL <= 0.0f) continue;
        const float attenuation = DistanceAttenuation(distSq, 1.0f / light.positionAndRange.w);
        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        const float3 lightColor = max(light.colorAndIntensity.rgb, 0.0f);
        const float lightIntensity = max(light.colorAndIntensity.w, 0.0f);
        luminance += brdf * lightColor * (lightIntensity * attenuation * NoL);
    }

    for (uint lightIndex = 0; lightIndex < lightListInfo.spotLightCount; ++lightIndex) {
        const device GPUSpotLight& light = args.spotLights[lightIndex];
        const float3 posToLight = light.positionAndRange.xyz - in.worldPosition;
        const float distSq = dot(posToLight, posToLight);
        const float3 l = SafeNormalize(posToLight, n);
        const float NoL = saturate(dot(n, l));
        if (NoL <= 0.0f) continue;
        const float distAtten = DistanceAttenuation(distSq, 1.0f / light.positionAndRange.w);
        const float spotAtten = SpotAngleAttenuation(l, normalize(light.direction.xyz), light.scaleOffset.x, light.scaleOffset.y);
        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        const float3 lightColor = max(light.colorAndIntensity.rgb, 0.0f);
        const float lightIntensity = max(light.colorAndIntensity.w, 0.0f);
        luminance += brdf * lightColor * (lightIntensity * distAtten * spotAtten * NoL);
    }

    return float4(luminance, 1.0f);
}

fragment float4 opaqueDirect_fragment(
    OpaqueVertexOut in [[stage_in]],
    bool frontFacing [[front_facing]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    return ShadeOpaqueDirect<false>(in, frontFacing, frame, args);
}

fragment float4 opaqueDirectMasked_fragment(
    OpaqueVertexOut in [[stage_in]],
    bool frontFacing [[front_facing]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    return ShadeOpaqueDirect<true>(in, frontFacing, frame, args);
}