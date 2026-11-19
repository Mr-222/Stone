#pragma once

#include <metal_stdlib>
using namespace metal;

// Smooth distance attenuation with inverse-square falloff and a windowed range.
// Adapted from Filament / Epic's UE4 approach.
// distanceSquared: squared distance from shading point to light
// rangeInverse: 1.0 / light range
inline float DistanceAttenuation(float distanceSquared, float rangeInverse) {
    const float factor = distanceSquared * rangeInverse * rangeInverse;
    const float smoothFactor = saturate(1.0f - factor * factor);
    return (smoothFactor * smoothFactor) / max(distanceSquared, 1e-4f);
}

// Spot cone angular attenuation (Filament / UE4 parameterization).
// l: normalized direction from shading point toward the light
// spotDirection: normalized direction the spot light is pointing
// angleScale: 1.0 / max(cos(innerAngle) - cos(outerAngle), 1e-5)
// angleOffset: -cos(outerAngle) * angleScale
inline float SpotAngleAttenuation(float3 l, float3 spotDirection, float angleScale, float angleOffset) {
    const float cd = dot(-l, spotDirection);
    const float attenuation = saturate(cd * angleScale + angleOffset);
    return attenuation * attenuation;
}

struct TileFrustumPlanes {
    float3 left;
    float3 right;
    float3 bottom;
    float3 top;
};

inline TileFrustumPlanes ComputeTileFrustumPlanes(uint2 tileCoord, uint2 screenSize, uint tileSize, float4x4 proj) {
    const float2 minPx = float2(tileCoord * tileSize);
    const float2 maxPx = min(float2((tileCoord + 1u) * tileSize), float2(screenSize));
    const float xLeft = minPx.x / float(screenSize.x) * 2.0f - 1.0f;
    const float xRight = maxPx.x / float(screenSize.x) * 2.0f - 1.0f;
    const float yTop = 1.0f - (minPx.y / float(screenSize.y)) * 2.0f;
    const float yBottom = 1.0f - (maxPx.y / float(screenSize.y)) * 2.0f;
    const float p00 = proj[0][0];
    const float p11 = proj[1][1];

    TileFrustumPlanes planes;
    planes.left = normalize(float3(p00, 0.0f, -xLeft));
    planes.right = normalize(float3(-p00, 0.0f, xRight));
    planes.bottom = normalize(float3(0.0f, p11, -yBottom));
    planes.top = normalize(float3(0.0f, -p11, yTop));
    return planes;
}

inline bool SphereIntersectsTileFrustumXY(float3 viewCenter, float radius, TileFrustumPlanes planes) {
    return dot(planes.left, viewCenter) >= -radius &&
           dot(planes.right, viewCenter) >= -radius &&
           dot(planes.bottom, viewCenter) >= -radius &&
           dot(planes.top, viewCenter) >= -radius;
}

inline bool SphereIntersectsDepthRange(float viewZ, float radius, float minViewZ, float maxViewZ) {
    return viewZ + radius >= minViewZ && viewZ - radius <= maxViewZ;
}
