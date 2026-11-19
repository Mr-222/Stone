#include "ShaderTypes.h"
#include "BRDF.h"
#include "Lighting.h"
#include "RayTracedShadow.h"
#include "Surface.h"
#include "Utility.h"

struct OpaqueVertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 worldNormal;
    float4 worldTangent;
    float2 uv;
    float  viewZ;
    uint   materialIndex [[flat]];
};

struct GBufferFragmentOut {
    half4 albedoMetallic [[color(1)]];
    float4 normalGeoNormal [[color(2)]];
    float2 depthRoughness [[color(3)]];
};

struct GBufferTileData {
    float2 depthRoughness [[color(3)]];
};

struct DeferredVertexOut {
    float4 position [[position]];
};

vertex OpaqueVertexOut gbuffer_vertex(
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
    out.viewZ = (frame.view * worldPosition).z;
    out.worldNormal = (primitive.worldNormalMat * float4(gpuVertex.normal, 0.0f)).xyz;
    out.worldTangent = float4((primitive.worldMat * float4(gpuVertex.tangent.xyz, 0.0f)).xyz, gpuVertex.tangent.w);
    out.uv = gpuVertex.uv;
    out.materialIndex = primitive.materialIndex;
    return out;
}

template <bool kAlphaTest>
inline GBufferFragmentOut EncodeGBuffer(OpaqueVertexOut in, bool frontFacing, device OpaqueDirectLightingFragmentArguments& args) {
    constexpr sampler baseColorSampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear, max_anisotropy(8));

    const float faceSign = frontFacing ? 1.0f : -1.0f;
    const float3 vertexNormal = SafeNormalize(in.worldNormal, float3(0.0f, 1.0f, 0.0f)) * faceSign;
    const float3 geometricNormal = ComputeGeometricNormal(in.worldPosition, vertexNormal);

    const device GPUMaterial& material = args.materials[in.materialIndex];
    const Surface surface = SampleSurface(material, args.textures, baseColorSampler, in.uv, in.worldNormal, in.worldTangent, frontFacing);
    if (kAlphaTest && (material.flags & kMaterialFlagAlphaMask) != 0 && surface.alpha < material.alphaCutoff)
        discard_fragment();

    GBufferFragmentOut out;
    out.albedoMetallic = half4(half3(surface.baseColor), half(surface.metallic));
    out.normalGeoNormal = float4(EncodeOctahedralNormal(surface.n), EncodeOctahedralNormal(geometricNormal));
    out.depthRoughness = float2(in.viewZ, surface.perceptualRoughness);
    return out;
}

fragment GBufferFragmentOut gbuffer_fragment(
    OpaqueVertexOut in [[stage_in]],
    bool frontFacing [[front_facing]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    return EncodeGBuffer<false>(in, frontFacing, args);
}

fragment GBufferFragmentOut gbufferMasked_fragment(
    OpaqueVertexOut in [[stage_in]],
    bool frontFacing [[front_facing]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    return EncodeGBuffer<true>(in, frontFacing, args);
}

struct LightCullingBounds {
    TileFrustumPlanes frustum;
    bool hasOpaque;
    float minOpaqueViewZ;
    float maxOpaqueViewZ;
    float nearViewZ;
    float maxTransViewZ;
};

// boundingSphere: xyz = world-space center, w = radius. Appends lightIndex to the opaque list (tile's opaque depth range) and/or transparent list (near plane to farthest opaque depth).
inline void CullLight(float4 boundingSphere, uint lightIndex, float4x4 view, LightCullingBounds bounds, threadgroup atomic_uint& opaqueCount, threadgroup uint16_t* opaqueIndices, threadgroup atomic_uint& transCount, threadgroup uint16_t* transIndices) {
    const float3 viewPos = (view * float4(boundingSphere.xyz, 1.0f)).xyz;
    const float radius = boundingSphere.w;
    if (!SphereIntersectsTileFrustumXY(viewPos, radius, bounds.frustum))
        return;
    if (bounds.hasOpaque && SphereIntersectsDepthRange(viewPos.z, radius, bounds.minOpaqueViewZ, bounds.maxOpaqueViewZ)) {
        const uint slot = atomic_fetch_add_explicit(&opaqueCount, 1u, memory_order_relaxed);
        if (slot < kMaxCulledLightsPerTile)
            opaqueIndices[slot] = uint16_t(lightIndex);
    }
    if (SphereIntersectsDepthRange(viewPos.z, radius, bounds.nearViewZ, bounds.maxTransViewZ)) {
        const uint slot = atomic_fetch_add_explicit(&transCount, 1u, memory_order_relaxed);
        if (slot < kMaxCulledLightsPerTile)
            transIndices[slot] = uint16_t(lightIndex);
    }
}

kernel void tileLightCulling_kernel(
    imageblock<GBufferTileData> imgBlk,
    ushort2 tid [[thread_position_in_threadgroup]],
    ushort threadIndex [[thread_index_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    threadgroup TileLightList& opaqueTileLights [[threadgroup(0)]],
    threadgroup TileCullingScratch& scratch [[threadgroup(1)]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]],
    device TileLightList* transparentTileLightLists [[buffer(OpaqueDirectLightingBufferIndex::TransparentTileLightLists)]])
{
    constexpr uint kTileThreads = kLightTileSize * kLightTileSize;

    if (threadIndex == 0) {
        atomic_store_explicit(&scratch.minOpaqueZ, as_type<uint>(FLT_MAX), memory_order_relaxed);
        atomic_store_explicit(&scratch.maxOpaqueZ, as_type<uint>(0.0f), memory_order_relaxed);
        atomic_store_explicit(&scratch.maxTransZ, as_type<uint>(0.0f), memory_order_relaxed);
        atomic_store_explicit(&scratch.opaquePointCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&scratch.opaqueSpotCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&scratch.transPointCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&scratch.transSpotCount, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool inBounds = gid.x < frame.screenSizeAndTileCount.x && gid.y < frame.screenSizeAndTileCount.y;
    // G-buffer stores linear view-space Z; 0 marks pixels with no opaque geometry (sky).
    const float farViewZ = ClipDepthToViewZ(1.0f, frame.proj);
    const float viewZ = inBounds ? imgBlk.read(tid).depthRoughness.x : 0.0f;
    const bool hasOpaquePixel = viewZ > 0.0f;
    const float pixelMinOpaqueZ = hasOpaquePixel ? viewZ : FLT_MAX;
    const float pixelMaxOpaqueZ = hasOpaquePixel ? viewZ : 0.0f;
    const float pixelMaxTransZ = inBounds ? (hasOpaquePixel ? viewZ : farViewZ) : 0.0f;

    const float sgMinOpaqueZ = simd_min(pixelMinOpaqueZ);
    const float sgMaxOpaqueZ = simd_max(pixelMaxOpaqueZ);
    const float sgMaxTransZ = simd_max(pixelMaxTransZ);
    if (simd_is_first()) {
        atomic_fetch_min_explicit(&scratch.minOpaqueZ, as_type<uint>(sgMinOpaqueZ), memory_order_relaxed);
        atomic_fetch_max_explicit(&scratch.maxOpaqueZ, as_type<uint>(sgMaxOpaqueZ), memory_order_relaxed);
        atomic_fetch_max_explicit(&scratch.maxTransZ, as_type<uint>(sgMaxTransZ), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float minOpaqueViewZ = as_type<float>(atomic_load_explicit(&scratch.minOpaqueZ, memory_order_relaxed));
    const float maxOpaqueViewZ = as_type<float>(atomic_load_explicit(&scratch.maxOpaqueZ, memory_order_relaxed));
    const float maxTransViewZ = as_type<float>(atomic_load_explicit(&scratch.maxTransZ, memory_order_relaxed));
    const bool hasOpaque = minOpaqueViewZ <= maxOpaqueViewZ;
    const float nearViewZ = ClipDepthToViewZ(0.0f, frame.proj);

    const uint2 tileCoord = gid / kLightTileSize;
    const TileFrustumPlanes frustum = ComputeTileFrustumPlanes(tileCoord, frame.screenSizeAndTileCount.xy, kLightTileSize, frame.proj);
    const device GPULightListInfo& lightListInfo = args.lightListInfo;

    const LightCullingBounds bounds { frustum, hasOpaque, minOpaqueViewZ, maxOpaqueViewZ, nearViewZ, maxTransViewZ };
    for (uint i = threadIndex; i < lightListInfo.pointLightCount; i += kTileThreads)
        CullLight(args.pointLights[i].positionAndRange, i, frame.view, bounds, scratch.opaquePointCount, opaqueTileLights.pointLightIndices, scratch.transPointCount, scratch.transPointIndices);
    for (uint i = threadIndex; i < lightListInfo.spotLightCount; i += kTileThreads) {
        const device GPUSpotLight& spot = args.spotLights[i];
        const float4 cullingSphere = float4(spot.positionAndRange.xyz + spot.direction.xyz * spot.direction.w, spot.scaleOffset.z);
        CullLight(cullingSphere, i, frame.view, bounds, scratch.opaqueSpotCount, opaqueTileLights.spotLightIndices, scratch.transSpotCount, scratch.transSpotIndices);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint transPointCount = min(atomic_load_explicit(&scratch.transPointCount, memory_order_relaxed), kMaxCulledLightsPerTile);
    const uint transSpotCount = min(atomic_load_explicit(&scratch.transSpotCount, memory_order_relaxed), kMaxCulledLightsPerTile);
    const uint tileIndex = tileCoord.y * frame.screenSizeAndTileCount.z + tileCoord.x;
    device TileLightList& outTransList = transparentTileLightLists[tileIndex];

    if (threadIndex == 0) {
        opaqueTileLights.pointLightCount = min(atomic_load_explicit(&scratch.opaquePointCount, memory_order_relaxed), kMaxCulledLightsPerTile);
        opaqueTileLights.spotLightCount = min(atomic_load_explicit(&scratch.opaqueSpotCount, memory_order_relaxed), kMaxCulledLightsPerTile);
        outTransList.pointLightCount = transPointCount;
        outTransList.spotLightCount = transSpotCount;
    }
    for (uint i = threadIndex; i < transPointCount; i += kTileThreads)
        outTransList.pointLightIndices[i] = scratch.transPointIndices[i];
    for (uint i = threadIndex; i < transSpotCount; i += kTileThreads)
        outTransList.spotLightIndices[i] = scratch.transSpotIndices[i];
}

vertex DeferredVertexOut deferredLighting_vertex(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[3] = { float2(-1.0f, -1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f) };
    DeferredVertexOut out;
    out.position = float4(positions[vertexID], 0.0f, 1.0f);
    return out;
}

fragment float4 deferredLighting_fragment(
    DeferredVertexOut in [[stage_in]],
    half4 albedoMetallic [[color(1)]],
    float4 normalGeoNormal [[color(2)]],
    float2 depthRoughness [[color(3)]],
    threadgroup const TileLightList& tileLights [[threadgroup(0)]],
    constant FrameUniform& frame [[buffer(OpaqueDirectLightingBufferIndex::FrameUniform)]],
    device OpaqueDirectLightingFragmentArguments& args [[buffer(OpaqueDirectLightingBufferIndex::FragmentArguments)]])
{
    const float viewZ = depthRoughness.x;
    if (viewZ <= 0.0f)
        return float4(0.0f, 0.0f, 0.0f, 1.0f);

    constexpr sampler alphaSampler(coord::normalized, address::repeat, filter::linear, mip_filter::linear);

    const float3 baseColor = float3(albedoMetallic.rgb);
    const float metallic = float(albedoMetallic.a);
    const float perceptualRoughness = depthRoughness.y;
    const float3 diffuseColor = baseColor * (1.0f - metallic);
    const float3 f0 = mix(float3(0.04f), baseColor, metallic);

    const float3 n = DecodeOctahedralNormal(normalGeoNormal.xy);
    const float3 geometricNormal = DecodeOctahedralNormal(normalGeoNormal.zw);
    const float3 worldPosition = ReconstructWorldPositionFromViewZ(in.position.xy, frame.screenSizeAndTileCount.xy, viewZ, frame.view, frame.proj);
    const float3 shadowRayOrigin = OffsetRayOrigin(worldPosition, geometricNormal);
    const float3 v = SafeNormalize(frame.cameraPosition.xyz - worldPosition, n);

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

        const float visibility = TraceShadowRay(args.accelerationStructure, args.opaqueVertices, args.opaqueIndices, args.opaqueSubmeshes, args.materials, args.textures, alphaSampler, shadowRayOrigin, l, INFINITY);
        if (visibility <= 0.0f)
            continue;

        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        float3 lightColor = max(light.colorAndIlluminance.rgb, 0.0f);
        if (lightIndex == 0)
            lightColor *= frame.sunTransmittance.rgb;
        const float lightIlluminance = max(light.colorAndIlluminance.w, 0.0f);
        luminance += brdf * lightColor * (lightIlluminance * NoL);
    }

    for (uint i = 0; i < tileLights.pointLightCount; ++i) {
        const uint lightIndex = tileLights.pointLightIndices[i];
        const device GPUPointLight& light = args.pointLights[lightIndex];
        const float3 posToLight = light.positionAndRange.xyz - worldPosition;
        const float distSq = dot(posToLight, posToLight);
        const float3 l = SafeNormalize(posToLight, n);
        const float NoL = saturate(dot(n, l));
        const float attenuation = DistanceAttenuation(distSq, 1.0f / light.positionAndRange.w);
        if (NoL <= 0.0f || attenuation <= 0.0f)
            continue;

        const float dist = sqrt(distSq);
        const float visibility = TraceShadowRay(args.accelerationStructure, args.opaqueVertices, args.opaqueIndices, args.opaqueSubmeshes, args.materials, args.textures, alphaSampler, shadowRayOrigin, l, max(dist - 1e-3f, 0.0f));
        if (visibility <= 0.0f)
            continue;

        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        const float3 lightColor = max(light.colorAndIntensity.rgb, 0.0f);
        const float lightIntensity = max(light.colorAndIntensity.w, 0.0f);
        luminance += brdf * lightColor * (lightIntensity * attenuation * NoL);
    }

    for (uint i = 0; i < tileLights.spotLightCount; ++i) {
        const uint lightIndex = tileLights.spotLightIndices[i];
        const device GPUSpotLight& light = args.spotLights[lightIndex];
        const float3 posToLight = light.positionAndRange.xyz - worldPosition;
        const float distSq = dot(posToLight, posToLight);
        const float3 l = SafeNormalize(posToLight, n);
        const float NoL = saturate(dot(n, l));
        const float distAtten = DistanceAttenuation(distSq, 1.0f / light.positionAndRange.w);
        const float spotAtten = SpotAngleAttenuation(l, normalize(light.direction.xyz), light.scaleOffset.x, light.scaleOffset.y);
        if (NoL <= 0.0f || distAtten <= 0.0f || spotAtten <= 0.0f)
            continue;

        const float dist = sqrt(distSq);
        const float visibility = TraceShadowRay(args.accelerationStructure, args.opaqueVertices, args.opaqueIndices, args.opaqueSubmeshes, args.materials, args.textures, alphaSampler, shadowRayOrigin, l, max(dist - 1e-3f, 0.0f));
        if (visibility <= 0.0f)
            continue;

        const float3 brdf = EvaluateBRDF(n, v, l, diffuseColor, f0, perceptualRoughness, NoL);
        const float3 lightColor = max(light.colorAndIntensity.rgb, 0.0f);
        const float lightIntensity = max(light.colorAndIntensity.w, 0.0f);
        luminance += brdf * lightColor * (lightIntensity * distAtten * spotAtten * NoL);
    }

    return float4(luminance, 1.0f);
}
