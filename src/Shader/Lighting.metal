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
