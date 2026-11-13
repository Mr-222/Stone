#pragma once

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

#include "ShaderTypes.h"
#include "BRDF.h"

inline float3 ComputeGeometricNormal(float3 worldPosition, float3 fallbackNormal) {
    const float3 ng = SafeNormalize(cross(dfdx(worldPosition), dfdy(worldPosition)), fallbackNormal);
    return dot(ng, fallbackNormal) < 0.0f ? -ng : ng;
}

// Wächter & Binder, Ray Tracing Gems Ch. 6 ("A Fast and Robust Method for Avoiding Self-Intersection")
inline float3 OffsetRayOrigin(float3 p, float3 n) {
    constexpr float kOrigin = 1.0f / 32.0f;
    constexpr float kFloatScale = 1.0f / 65536.0f;
    constexpr float kIntScale = 256.0f;
    const int3 of_i = int3(kIntScale * n.x, kIntScale * n.y, kIntScale * n.z);
    const float3 p_i = float3(
        as_type<float>(as_type<int>(p.x) + ((p.x < 0.0f) ? -of_i.x : of_i.x)),
        as_type<float>(as_type<int>(p.y) + ((p.y < 0.0f) ? -of_i.y : of_i.y)),
        as_type<float>(as_type<int>(p.z) + ((p.z < 0.0f) ? -of_i.z : of_i.z)));
    return float3(
        abs(p.x) < kOrigin ? p.x + kFloatScale * n.x : p_i.x,
        abs(p.y) < kOrigin ? p.y + kFloatScale * n.y : p_i.y,
        abs(p.z) < kOrigin ? p.z + kFloatScale * n.z : p_i.z) + n * 1e-4f;
}

inline float TraceShadowRay(
    instance_acceleration_structure accelerationStructure,
    const device GPUVertex* vertices,
    const device uint32_t* indices,
    const device GPUSubMesh* submeshes,
    const device GPUMaterial* materials,
    const device array<texture2d<float>, kMaxBindlessTextureCount>& textures,
    sampler alphaSampler,
    float3 rayOrigin,
    float3 rayDirection,
    float tMax)
{
    ray shadowRay(rayOrigin, rayDirection, 0.0f, tMax);
    intersection_params params;
    params.accept_any_intersection(true);
    params.assume_geometry_type(geometry_type::triangle);

    intersection_query<instancing, triangle_data> query(shadowRay, accelerationStructure, params);
    while (query.next()) {
        const uint submeshIndex = query.get_candidate_user_instance_id() + query.get_candidate_geometry_id();
        const uint primitiveID = query.get_candidate_primitive_id();
        const device GPUSubMesh& submesh = submeshes[submeshIndex];
        const uint indexBase = submesh.firstIndex + primitiveID * 3;
        const uint i0 = submesh.vertexOffset + indices[indexBase + 0];
        const uint i1 = submesh.vertexOffset + indices[indexBase + 1];
        const uint i2 = submesh.vertexOffset + indices[indexBase + 2];

        const float2 uv0 = float2(vertices[i0].uv);
        const float2 uv1 = float2(vertices[i1].uv);
        const float2 uv2 = float2(vertices[i2].uv);
        const float2 bary = query.get_candidate_triangle_barycentric_coord();
        const float2 uv = uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;

        const device GPUMaterial& material = materials[submesh.materialIndex];
        const float alpha = textures[material.baseColorTextureIndex].sample(alphaSampler, uv, level(0.0f)).a * material.baseColorFactor.a;
        if (alpha >= material.alphaCutoff)
            query.commit_triangle_intersection();
    }

    return query.get_committed_intersection_type() == intersection_type::none ? 1.0f : 0.0f;
}
