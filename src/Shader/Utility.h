#include <metal_stdlib>
using namespace metal;

constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float InvPI = 1.0f / PI;

constexpr float ConstexprSqrt(float x) {
    float curr = x >= 1.0f ? x : 1.0f, prev = 0.0f;
    while (curr != prev) { prev = curr; curr = 0.5f * (curr + x / curr); }
    return curr;
}

float RaySphereIntersect(float3 rayOrigin, float3 rayDir, float radius)
{
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float d = b * b - c;
    if (d < 0.0f) return -1.0f;

    float sqrtD = sqrt(d);
    float t1 = -b - sqrtD;
    float t2 = -b + sqrtD;

    if (t1 > 0.0f) return t1;
    if (t2 > 0.0f) return t2;
    return -1.0f;
}

// Atmospheric scattering coefficients
constant constexpr float R_ground = 6360; // km
constant constexpr float R_top    = 6460;
constant constexpr float H        = ConstexprSqrt(R_top * R_top - R_ground * R_ground); // Maximum Geometric Horizon Distance