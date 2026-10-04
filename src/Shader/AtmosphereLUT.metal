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

// ==============================================================================
// Transmittance LUT
// ==============================================================================

float3 ThreadToAtmosphereParam(uint2 threadIdx, uint width, uint height)
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
    uint width = args.transmittanceLUT.get_width();
    uint height = args.transmittanceLUT.get_height();

    float3 params = ThreadToAtmosphereParam(threadIdx, width, height);
    float r       = params.x;
    float mu      = params.y;
    float d       = params.z;

    float3 rayOrigin = float3(0.0, r, 0.0);
    float3 rayDir    = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    float3 transmittance = IntegrateTransmittance(rayOrigin, rayDir, d);

    args.transmittanceLUT.write(float4(transmittance, 1.0), threadIdx);
}

// ==============================================================================
// SkyView LUT
// ==============================================================================

constant uint SAMPLE_COUNT = 30;

float3 UVToViewDirection(float2 uv)
{
    // Longitudinal azimuth
    float phi = uv.x * 2.0f * PI;

    // Latitudinal azimuth
    float coord = 2.0f * uv.y - 1.0f;
    float l = sign(coord) * PI * 0.5f * coord * coord;

    float cosL = cos(l);
    return normalize(float3(cosL * cos(phi), sin(l), cosL * sin(phi)));
}

float PhaseRayleigh(float cosTheta)
{
    return (3.0f / (16.0f * PI)) * (1.0f + cosTheta * cosTheta);
}

float PhaseMie(float cosTheta, float g = 0.8f)
{
    float g2 = g * g;
    float num = 3.0f * (1.0f - g2) * (1.0f + cosTheta * cosTheta);
    float denom = (8.0f * PI) * (2.0f + g2) * pow(abs(1.0f + g2 - 2.0f * g * cosTheta), 1.5f);
    return num / denom;
}

float3 SampleTransmittanceLUT(texture2d<float, access::sample> transmittanceLUT,
                              sampler s,
                              float3 worldPos,
                              float3 planetCenter,
                              float3 sunDir)
{
    uint width  = transmittanceLUT.get_width();
    uint height = transmittanceLUT.get_height();

    float3 up = worldPos - planetCenter;
    float  r  = length(up);
    up /= r;

    float mu = dot(up, sunDir);

    // Planetary Surface Occlusion Determination
    if (mu < 0.0f && (r * r * (mu * mu - 1.0f) + R_ground * R_ground > 0.0f))
        return float3(0.0f);

    float H     = sqrt(R_top * R_top - R_ground * R_ground);
    float rho   = sqrt(max(0.0f, r * r - R_ground * R_ground));

    float dMin = R_top - r;
    float dMax = rho + H;
    float discriminant = r * r * (mu * mu - 1.0f) + R_top * R_top;
    float d = -r * mu + sqrt(max(0.0f, discriminant));

    float u = (d - dMin) / (dMax - dMin);
    float v = rho / H;

    // Half texel offset correction
    u = 0.5f / width + u * (1.0f - 1.0f / width);
    v = 0.5f / height + v * (1.0f - 1.0f / height);

    return transmittanceLUT.sample(s, float2(u, v), level(0.0f)).rgb;
}

kernel void skyView_main(
    uint2 threadIdx [[thread_position_in_grid]],
    device AtmosphereSkyViewLUTKernelArguments& args [[buffer(SkyViewBufferIndex::KernelArguments)]])
{
    uint width = args.skyViewLUT.get_width();
    uint height = args.skyViewLUT.get_height();

    constexpr sampler linearClampSampler(coord::normalized, address::clamp_to_edge, filter::linear);

    if (threadIdx.x >= width || threadIdx.y >= height)
        return;

    // Half pixel centering
    float2 uv = (float2(threadIdx) + 0.5f) / float2(width, height);

    // Unproject the gaze vector and its origin (with the camera positioned on the Y-axis)
    float3 rayDir = UVToViewDirection(uv);
    float3 rayOrigin = float3(0.0f, args.param.cameraAltitude, 0.0f);
    float3 planetCenter = float3(args.param.planetCenter);

    // Treat first directional light as sun
    GPUDirectionalLight sun = args.directionalLights[0];
    float3 sunDir = -normalize(sun.direction.xyz);
    float3 sunIlluminance = sun.colorAndIlluminance.xyz * sun.colorAndIlluminance.w;

    // One compute thread for storing sun transmittance
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float3 cameraSunTransmittance = SampleTransmittanceLUT(args.transmittanceLUT, linearClampSampler, rayOrigin, planetCenter, sunDir);
        args.frameUniform.sunTransmittance = float4(cameraSunTransmittance, 1.0f);
    }

    // Determine whether the gaze ray hits the ground
    float distToGround = RaySphereIntersect(rayOrigin - planetCenter, rayDir, R_ground);
    float distToTop    = RaySphereIntersect(rayOrigin - planetCenter, rayDir, R_top);

    bool hitsGround = (distToGround > 0.0f);
    float rayLength = hitsGround ? distToGround : distToTop;

    // when camera is very near or above atmosphere boundary
    if (rayLength <= 0.0f)
    {
        args.skyViewLUT.write(float4(0.0f, 0.0f, 0.0f, 1.0f), threadIdx);
        return;
    }

    // Ray marching
    float stepSize = rayLength / float(SAMPLE_COUNT);
    float3 accumulatedLuminance = float3(0.0f);
    float3 accumulatedTransmittance = float3(1.0f);

    float cosTheta = clamp(dot(rayDir, sunDir), -1.0f, 1.0f);
    float phaseR   = PhaseRayleigh(cosTheta);
    float phaseM   = PhaseMie(cosTheta);

    for (uint i = 0; i < SAMPLE_COUNT; ++i)
    {
        // Take the sampling point at the center of the differential element
        float t = ((float)i + 0.5f) * stepSize;
        float3 samplePos = rayOrigin + rayDir * t;
        float sampleAltitude = length(samplePos - planetCenter) - R_ground;

        float3 sigma_s, sigma_t, sigma_s_R, sigma_s_M;
        GetMediumCoefficients(sampleAltitude, sigma_s, sigma_t, sigma_s_R, sigma_s_M);

        float3 transmittanceToSun = SampleTransmittanceLUT(args.transmittanceLUT, linearClampSampler, samplePos, planetCenter, sunDir);

        // Single in-scattering source term
        float3 inScattering = (sigma_s_R * phaseR + sigma_s_M * phaseM) * transmittanceToSun * sunIlluminance;

        // Extinction attenuation of a differential element
        float3 stepTransmittance = exp(-sigma_t * stepSize);

        // -----------------------------------------------------------------------------
        // Analytic segment integration (In-scattering & self-extinction):
        // Instead of a naive Riemann sum (inScattering * stepSize), which assumes constant
        // transmittance within the step and overestimates incoming energy (causing severe
        // slicing artifacts and energy blowout under coarse step counts or dense media),
        // we evaluate the definite integral assuming S (inScattering) and sigma_t are constant
        // along the sub-segment [0, dt]:
        //
        //   L_step = \int_{0}^{dt} S * exp(-sigma_t * x) dx
        //          = S * (1 - exp(-sigma_t * dt)) / sigma_t
        //          = (S - S * stepTransmittance) / sigma_t
        //
        // A safety epsilon (1e-5) guards against divide-by-zero in thin/vacuum regions.
        // -----------------------------------------------------------------------------
        float3 integratedScattering = (inScattering - inScattering * stepTransmittance) / max(sigma_t, float3(1e-5));
        accumulatedLuminance += integratedScattering * accumulatedTransmittance;

        accumulatedTransmittance *= stepTransmittance;
    }

    // Ground boundary diffuse reflection processing
    if (hitsGround)
    {
        float3 hitPos = rayOrigin + rayDir * distToGround;
        float3 normal = normalize(hitPos - planetCenter);
        float NoL     = saturate(dot(normal, sunDir));

        if (NoL > 0.0f)
        {
            float3 sunTransmittanceAtGround = SampleTransmittanceLUT(args.transmittanceLUT, linearClampSampler, hitPos, planetCenter, sunDir);
            float3 directIlluminance = sunIlluminance * sunTransmittanceAtGround;
            float3 groundL0 = (float3(args.param.groundAlbedo) / float3(PI)) * directIlluminance * NoL;

            accumulatedLuminance += accumulatedTransmittance * groundL0;
        }
    }

    args.skyViewLUT.write(float4(accumulatedLuminance, 1.0f) , threadIdx);
}