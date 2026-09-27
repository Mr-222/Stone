#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "Utility.h"

void GetMediumCoefficients(float altitude, thread float3 &sigma_s, thread float3 &sigma_t, thread float3 &sigma_s_R, thread float3 &sigma_s_M)
{
    float h = max(0.0f, altitude);

    float densityR = exp(-h / 8.0f);
    float densityM = exp(-h / 1.2f);
    float densityO = max(0.0f, 1.0f - abs(h - 25.0f) / 15.0f);

    // Standard atmospheric extinction coefficients (in km^-1)
    constexpr float3 baseSigmaRayleigh = float3(0.005802f, 0.013558f, 0.033100f);
    constexpr float  baseSigmaMieS     = 0.003996f;
    constexpr float  baseSigmaMieA     = 0.004400f;
    constexpr float3 baseSigmaOzoneA   = float3(0.000650f, 0.001881f, 0.000085f);

    sigma_s_R = baseSigmaRayleigh * densityR;
    sigma_s_M = float3(baseSigmaMieS) * densityM;
    sigma_s   = sigma_s_R + sigma_s_M;

    float3 sigma_a_M = float3(baseSigmaMieA) * densityM;
    float3 sigma_a_O = baseSigmaOzoneA * densityO;

    // scattering + absorption
    sigma_t = sigma_s + sigma_a_M + sigma_a_O;
}

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
        float3 sigma_s, sigma_t, sigma_s_R, sigma_s_M;
        GetMediumCoefficients(h, sigma_s, sigma_t, sigma_s_R, sigma_s_M);
        opticalDepth += sigma_t * dt;
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