#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"

constexpr float ConstexprSqrt(float x) {
    float curr = x >= 1.0f ? x : 1.0f, prev = 0.0f;
    while (curr != prev) { prev = curr; curr = 0.5f * (curr + x / curr); }
    return curr;
}

constant constexpr float R_ground = 6360; // km
constant constexpr float R_top    = 6460;
constant constexpr float H        = ConstexprSqrt(R_top * R_top - R_ground * R_ground); // Maximum Geometric Horizon Distance

// Standard atmospheric extinction coefficients (in km^-1)
constant constexpr float3 sigmaRayleigh     = float3(0.005802, 0.013558, 0.033100);
constant constexpr float sigmaMieScattering = 0.003996;
constant constexpr float sigmaMieAbsorption = 0.004400;
constant constexpr float sigmaMie           = sigmaMieScattering + sigmaMieAbsorption; // 0.008396
constant constexpr float3 sigmaOzone        = float3(0.000650, 0.001881, 0.000085);

float3 threadToAtmosphereParam(uint2 threadIdx, uint width, uint height)
{
    // scale to [0, 1] with sub-texel accuracy, pixel 0 -> 0.0, pixel N - 1 -> 1.0
    float xu = (float)threadIdx.x / (width - 1.0);
    float xv = (float)threadIdx.y / (height - 1.0);

    // Decode geocentric distance r and altitude h from xv
    float rho = xv * H; // tangent point distance
    float r = sqrt(rho * rho + R_ground * R_ground);

    // Decode zenith cosine mu from xu
    float dMin = R_top - r;
    float dMax = rho + H;
    float d = dMin + xu * (dMax - dMin);
    if (d == 0.0)
        return float3(r, 1.0, d);

    float mu = (R_top * R_top - r * r - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);

    return float3(r, mu, d);
}

float densityRayleigh(float h) { return exp(-max(h, 0.0) / 8.0); }
float densityMie(float h)      { return exp(-max(h, 0.0) / 1.2); }

float densityOzone(float h) { return max(0.0, 1.0 - abs(h - 25.0) / 15.0); }

float3 GetExtinction(float h)
{
    return sigmaRayleigh * densityRayleigh(h) +
           sigmaMie * densityMie(h) +
           sigmaOzone * densityOzone(h);
}

float3 IntegrateTransmittance(float3 pos, float3 dir, float tMax)
{
    if (tMax <= 0.0) return float3(1.0);

    float3 opticalDepth = float3(0.0);
    constexpr uint stepCount = 40;

    float dt = tMax / float(stepCount);
    float3 p = pos + dir * (dt * 0.5);

    for (uint i = 0; i < stepCount; ++i)
    {
        float h = length(p) - R_ground;
        opticalDepth += GetExtinction(h) * dt;
        p += dir * dt;
    }

    return exp(-opticalDepth);
}

kernel void transmittance_main(
    uint2 threadIdx                                        [[thread_position_in_grid]],
    device AtmosphereTransmittanceLUTKernelArguments& args [[buffer(TransmittanceBufferIndex::KernelArguments)]])
{
    uint width = args.texSize.x;
    uint height = args.texSize.y;

    float3 params = threadToAtmosphereParam(threadIdx, width, height);
    float r       = params.x;
    float mu      = params.y;
    float d       = params.z;

    float3 rayOrigin = float3(0.0, r, 0.0);
    float3 rayDir    = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    float3 transmittance = IntegrateTransmittance(rayOrigin, rayDir, d);

    args.transmittanceLUT.write(float4(transmittance, 1.0), threadIdx);
}