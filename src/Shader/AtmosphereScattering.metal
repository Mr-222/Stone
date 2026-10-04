#include <metal_stdlib>
using namespace metal;

#include "ShaderTypes.h"
#include "Utility.h"

struct VertexOut {
    float4 position [[position]];
    float3 viewDir; // camera to far plane world-space ray (unnormalized)
};

float2 ViewDirectionToSkyViewUV(float3 viewDir)
{
    // u: phi [0, 2pi] map to [0, 1]
    float phi = atan2(viewDir.z, viewDir.x);
    phi = phi < 0.0f ? phi + 2.0f * PI : phi;
    float u = phi / (2.0f * PI);

    // v: [-PI/2, PI/2], non-linear mapping
    float l = asin(clamp(viewDir.y, -1.0f, 1.0f));
    float v = 0.5f + 0.5f * sign(l) * sqrt(abs(l) / (0.5f * PI));

    return float2(u, v);
}

float3 SampleTransmittanceLUT(texture2d<float, access::sample> transmittanceLUT, sampler s, float3 worldPos, float3 planetCenter, float3 sunDir)
{
    uint width  = transmittanceLUT.get_width();
    uint height = transmittanceLUT.get_height();

    float3 up = worldPos - planetCenter;
    float  r  = length(up);
    up /= r;

    float mu = dot(up, sunDir);

    // Planet surface hard occlusion determination
    // Atmospheric transmittance drops to zero immediately after sunset when the sun sinks below the horizon
    if (mu < 0.0f && (r * r * (mu * mu - 1.0f) + R_ground * R_ground > 0.0f))
        return float3(0.0f);

    float rho = sqrt(max(0.0f, r * r - R_ground * R_ground));

    float dMin = R_top - r;
    float dMax = rho + H;
    float discriminant = r * r * (mu * mu - 1.0f) + R_top * R_top;
    float d = -r * mu + sqrt(max(0.0f, discriminant));

    float u = (d - dMin) / (dMax - dMin);
    float v = rho / H;

    // Sub-texel Offset
    u = 0.5f / width + u * (1.0f - 1.0f / width);
    v = 0.5f / height + v * (1.0f - 1.0f / height);

    return transmittanceLUT.sample(s, float2(u, v), level(0.0f)).rgb;
}

float3 RenderSunDisk(float3 viewDir, float3 sunDir, float3 sunIlluminance, float3 cameraPos, float3 planetCenter, texture2d<float, access::sample> transmittanceLUT, sampler s)
{
    float cosTheta = dot(viewDir, sunDir);
    float angle = acos(clamp(cosTheta, -1.0f, 1.0f));

    float angleFootprint = max(fwidth(angle), 1e-7f);

    // Sight out of sundisk radius, no direct light
    if (angle > SUN_ANGULAR_RADIUS)
        return float3(0.0f);

    // Sun disk is occluded by the planet where the view ray hits the ground
    if (RaySphereIntersect(cameraPos - planetCenter, viewDir, R_ground) > 0.0f)
        return float3(0.0f);

    float3 sunTransmittance = SampleTransmittanceLUT(transmittanceLUT, s, cameraPos, planetCenter, sunDir);

    // Sun per solid angle luminance: L_sun = E_sun / (PI * theta^2)
    float sunSolidAngle = PI * SUN_ANGULAR_RADIUS * SUN_ANGULAR_RADIUS;
    float3 sunRadiance  = sunIlluminance / sunSolidAngle * sunTransmittance;

    // Limb darkening: center brighter edge darker
    float normalizedRadius = saturate(angle / SUN_ANGULAR_RADIUS); // [0, 1]
    float muSun = sqrt(1.0f - normalizedRadius * normalizedRadius);
    // Second-order Eddington approximation
    float limbDarkening = 1.0f - 0.47f * (1.0f - muSun) - 0.23f * pow(1.0f - muSun, 2.0f);

    // Edge smooth (anti-aliasing) over one pixel footprint
    float edgeFactor = saturate((SUN_ANGULAR_RADIUS - angle) / angleFootprint);

    return sunRadiance * limbDarkening * edgeFactor;
}

vertex VertexOut atmosphereScattering_vertex(
    uint vertexID [[vertex_id]],
    constant FrameUniform& frame [[buffer(AtmosphereScatteringBufferIndex::FrameUniform)]])
{
    VertexOut out;

    // Fullscreen triangle: 3 vertices cover the entire screen
    constexpr float2 positions[3] = {
        float2(-1.0f, -1.0f),
        float2( 3.0f, -1.0f),
        float2(-1.0f,  3.0f)
    };
    float2 clipXY = positions[vertexID];
    out.position = float4(clipXY, 1.0f, 1.0f); // far plane

    // Points on the far plane are affine in NDC xy, so linear interpolation of viewDir is exact
    float4 worldPosH = frame.invViewProj * out.position;
    out.viewDir = worldPosH.xyz / worldPosH.w - frame.cameraPosition.xyz;

    return out;
}

fragment float4 atmosphereScattering_fragment(
    VertexOut in [[stage_in]],
    device AtmosphereScatteringFragmentArguments& args [[buffer(AtmosphereScatteringBufferIndex::FragmentArguments)]])
{
    // Azimuth wraps around, latitude clamps
    constexpr sampler skyViewSampler(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear);
    constexpr sampler linearClampSampler(coord::normalized, address::clamp_to_edge, filter::linear);

    float3 rayDir = normalize(in.viewDir);

    float2 skyViewUV   = ViewDirectionToSkyViewUV(rayDir);
    float3 skyRadiance = args.skyViewLUT.sample(skyViewSampler, skyViewUV, level(0.0f)).rgb;

    // Treat first directional light as sun
    const device GPUDirectionalLight& sun = args.directionalLights[0];
    float3 sunDir = -normalize(sun.direction.xyz);
    float3 sunIlluminance = sun.colorAndIlluminance.xyz * sun.colorAndIlluminance.w;

    // Same atmosphere frame as skyView_main: camera on +Y above planet center, in km
    float3 cameraPos    = float3(0.0f, args.param.cameraAltitude, 0.0f);
    float3 planetCenter = float3(args.param.planetCenter);

    float3 sunDisk = RenderSunDisk(rayDir, sunDir, sunIlluminance, cameraPos, planetCenter, args.transmittanceLUT, linearClampSampler);

    return float4(skyRadiance + sunDisk, 1.0f);
}