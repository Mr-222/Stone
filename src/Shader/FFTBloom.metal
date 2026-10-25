#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

constexpr constant uint kFFTHalfSize = kFFTBloomSize / 2;
constexpr constant uint kFFTStages = 9;
static_assert((1u << kFFTStages) == kFFTBloomSize, "kFFTStages must match kFFTBloomSize");

inline float2 ComplexMul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Unpacks the 2D DFTs A(u, v) and B(u, v) of two real signals packed as z = a + i*b using Hermitian symmetry
inline void UnpackTwoForOne(float2 z, float2 zMirror, thread float2& outA, thread float2& outB)
{
    outA = 0.5f * float2(z.x + zMirror.x, z.y - zMirror.y);
    outB = 0.5f * float2(z.y + zMirror.y, zMirror.x - z.x);
}

float3 BloomBrightPass(float3 color, float threshold, float knee, float maxLuma)
{
    float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));

    // soft-knee threshold
    float soft = clamp(luma - threshold + knee, 0.0f, 2.0f * knee);
    soft = (soft * soft) / (4.0f * knee);
    float extractedLuma = max(soft, luma - threshold);

    // highlight compression
    float compressedLuma = extractedLuma / (1.0f + extractedLuma / maxLuma);

    // desaturate towards white
    float compressRatio = extractedLuma / (extractedLuma + maxLuma);
    float3 chroma = mix(color / max(luma, 1e-4f), float3(1.0f), 0.3f * compressRatio);
    return chroma * compressedLuma;
}

kernel void fft_bloom_downsample(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomDownsampleArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    constexpr sampler linearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    constexpr float invSize = 1.0f / float(kFFTBloomSize);

    float threshold = args.threshold;
    float knee = max(args.knee, 1e-4f);
    float maxLuma = max(args.maxLuma, 1.0f);

    float2 centerUV = (float2(gid) + 0.5f) * invSize;
    float2 offset = float2(0.25f * invSize);

    float3 s0 = BloomBrightPass(args.sceneColorTexture.sample(linearSampler, centerUV + float2(-offset.x, -offset.y)).rgb, threshold, knee, maxLuma);
    float3 s1 = BloomBrightPass(args.sceneColorTexture.sample(linearSampler, centerUV + float2( offset.x, -offset.y)).rgb, threshold, knee, maxLuma);
    float3 s2 = BloomBrightPass(args.sceneColorTexture.sample(linearSampler, centerUV + float2(-offset.x,  offset.y)).rgb, threshold, knee, maxLuma);
    float3 s3 = BloomBrightPass(args.sceneColorTexture.sample(linearSampler, centerUV + float2( offset.x,  offset.y)).rgb, threshold, knee, maxLuma);

    args.outTexture.write(float4((s0 + s1 + s2 + s3) * 0.25f, 1.0f), gid);
}

kernel void fft_bloom_prefilter(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomPrefilterArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // Zero padding outside the content region guarantees linear (non-wrapping) convolution
    if (gid.x >= kFFTBloomContentSize || gid.y >= kFFTBloomContentSize) {
        args.outTexture.write(float4(0.0f), gid);
        return;
    }

    // Tent downsample spanning +-1 destination texel. Tents spaced one texel apart form a partition of unity, so every
    // source pixel contributes the same total energy regardless of its sub-texel position (no flicker as content moves).
    constexpr float kFootprint = float(kFFTBloomSize) / float(kFFTBloomContentSize);
    float2 center = (float2(gid) + 0.5f) * kFootprint;
    int2 pMin = max(int2(floor(center - kFootprint)), int2(0));
    int2 pMax = min(int2(ceil(center + kFootprint)), int2(kFFTBloomSize - 1));

    float3 accum = float3(0.0f);
    for (int y = pMin.y; y <= pMax.y; ++y) {
        float wy = max(1.0f - abs(float(y) + 0.5f - center.y) / kFootprint, 0.0f);
        if (wy <= 0.0f) continue;

        for (int x = pMin.x; x <= pMax.x; ++x) {
            float wx = max(1.0f - abs(float(x) + 0.5f - center.x) / kFootprint, 0.0f);
            if (wx <= 0.0f) continue;

            accum += args.inTexture.read(uint2(x, y)).rgb * (wx * wy);
        }
    }
    float3 extracted = accum / (kFootprint * kFootprint);

    // Two-for-one FFT packing: Z_RG = R + i*G in .xy, Z_B = B + i*0 in .zw
    args.outTexture.write(float4(extracted.r, extracted.g, extracted.b, 0.0f), gid);
}

kernel void fft_psf_generate(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomPSFArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // Wrap coordinates to place kernel impulse center at (0, 0)
    float dx = float(int(gid.x < kFFTHalfSize ? gid.x : gid.x - kFFTBloomSize));
    float dy = float(int(gid.y < kFFTHalfSize ? gid.y : gid.y - kFFTBloomSize));

    // Pixel aspect correction (content is a square buffer stretched over the screen) to preserve circular geometry
    float dx_screen = dx * args.pixelAspect;
    float r = sqrt(dx_screen * dx_screen + dy * dy);

    // Support radius (in vertical texels) chosen so the PSF never exceeds kFFTBloomKernelRadius texels on either axis
    float supportRadius = float(kFFTBloomKernelRadius - 1) * min(1.0f, args.pixelAspect);
    float t = saturate(r / supportRadius);
    float window = (1.0f - t * t) * (1.0f - t * t);

    // Shape parameters below are authored for a 512-texel screen height; rescale so the on-screen size is resolution independent
    constexpr float kRefScale = float(kFFTBloomSize) / float(kFFTBloomContentSize);
    float rRef = r * kRefScale;

    // Area-normalized 2D exponential lobe (integral of exp(-r/s) over 2D plane is 2*pi*s^2), so weights represent true energy shares
    auto haloLobe = [&](float weight, float sigma) {
        return weight * exp(-rRef / sigma) / (2.0f * M_PI_F * sigma * sigma);
    };

    // Multi-octave optical halo: tight core glare, bright near halo, medium corona, and wide atmospheric scatter
    float halo = haloLobe(0.65f, 1.5f) + haloLobe(0.23f, 4.5f) + haloLobe(0.09f, 12.0f) + haloLobe(0.03f, 30.0f);

    // Iris diffraction spikes: rotated by 30 degrees (pi / 6) to place spikes at 0, 60, 120, 180, 240, 300 deg
    float angle = (dx_screen == 0.0f && dy == 0.0f) ? 0.0f : atan2(dy, dx_screen);
    float rotAngle = angle - (M_PI_F / 6.0f);
    float spikeCount = max(float(args.diffractionSpikeCount), 2.0f);
    float sector = 2.0f * M_PI_F / spikeCount;
    float modAngle = fmod(rotAngle + 4.0f * M_PI_F, sector);
    float distToSpike = abs(modAngle - sector * 0.5f);

    // Soft angular profile (bright central beam with soft optical wings)
    float angularFalloff = exp(-distToSpike * 24.0f) + 0.20f * exp(-distToSpike * 7.0f);

    // Physical wavelength scaling: Red diffracts further than Green, Green further than Blue (normalized to match 2D halo flux scale)
    auto computeSpikeChannel = [&](float lambda) {
        float r_scaled = rRef / lambda;
        float radialDecay = exp(-r_scaled / 15.0f) / pow(r_scaled * 0.05f + 1.0f, 1.6f);
        float ripple = 0.80f + 0.20f * cos(r_scaled * 0.35f);
        return (angularFalloff * radialDecay * ripple / 200.0f) * args.diffractionIntensity;
    };

    float spikeR = computeSpikeChannel(1.15f);
    float spikeG = computeSpikeChannel(1.00f);
    float spikeB = computeSpikeChannel(0.85f);

    float3 psf = (float3(halo) + float3(spikeR, spikeG, spikeB)) * window;
    // Two-for-one FFT packing: Z_RG = R + i*G in .xy, Z_B = B + i*0 in .zw
    args.outTexture.write(float4(psf.r, psf.g, psf.b, 0.0f), gid);
}

inline void ExecuteFFT1D(bool isVertical, uint lineIdx, uint tid, device FFTBloom1DArguments& args, threadgroup float2* s_rg, threadgroup float2* s_b)
{
    uint i0 = tid;
    uint i1 = tid + kFFTHalfSize;

    // 2. Hardware bit-reversal permutation: 9-bit reversal via 32-bit reverse_bits() >> 23
    uint rev0 = reverse_bits(i0) >> (32 - kFFTStages);
    uint rev1 = reverse_bits(i1) >> (32 - kFFTStages);

    uint2 coord0 = isVertical ? uint2(lineIdx, i0) : uint2(i0, lineIdx);
    uint2 coord1 = isVertical ? uint2(lineIdx, i1) : uint2(i1, lineIdx);

    // 3. Coalesced global texture read, scattered into bit-reversed shared memory slots
    float4 c0 = args.inTexture.read(coord0);
    float4 c1 = args.inTexture.read(coord1);

    s_rg[rev0] = c0.xy;
    s_b[rev0] = c0.zw;

    s_rg[rev1] = c1.xy;
    s_b[rev1] = c1.zw;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Forward FFT uses exp(-i*2*pi*k/m), Inverse FFT uses exp(+i*2*pi*k/m)
    float sign = args.inverse ? 1.0f : -1.0f;
    for (uint s = 1; s <= kFFTStages; ++s) {
        uint m = 1u << s; // Sub-DFT size: 2, 4, 8, 16, 32, 64, 128, 256, 512
        uint half_m = m >> 1; // Butterfly stride: 1, 2, 4, 8, 16, 32, 64, 128, 256
        uint k = tid % half_m; // Index within butterfly group (0 .. half_m - 1)
        uint b = tid / half_m; // Which butterfly group this thread belongs to
        uint j = b * m + k; // Top element index E[k]; bottom element O[k] is (j + half_m)

        float angle = sign * 2.0f * M_PI_F * float(k) / float(m);
        float2 W = float2(cos(angle), sin(angle)); // Twiddle factor W_m^k

        // Complex multiply (t = W * v) and butterfly write-back for packed RG and B channels
        float2 u_rg = s_rg[j], t_rg = ComplexMul(W, s_rg[j + half_m]);
        float2 u_b = s_b[j], t_b = ComplexMul(W, s_b[j + half_m]);

        s_rg[j] = u_rg + t_rg; s_rg[j + half_m] = u_rg - t_rg;
        s_b[j] = u_b + t_b; s_b[j + half_m] = u_b - t_b;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = args.inverse ? (1.0f / float(kFFTBloomSize)) : 1.0f;
    if (isVertical && args.inverse) {
        // Final 2D IFFT pass: unpack real R = Re(Z_RG), G = Im(Z_RG), B = Re(Z_B) and clamp negative ringing
        float3 out0 = max(float3(s_rg[i0].x, s_rg[i0].y, s_b[i0].x) * scale, 0.0f);
        float3 out1 = max(float3(s_rg[i1].x, s_rg[i1].y, s_b[i1].x) * scale, 0.0f);
        args.outTexture.write(float4(out0, 1.0f), coord0);
        args.outTexture.write(float4(out1, 1.0f), coord1);
    } else {
        args.outTexture.write(float4(s_rg[i0] * scale, s_b[i0] * scale), coord0);
        args.outTexture.write(float4(s_rg[i1] * scale, s_b[i1] * scale), coord1);
    }
}

kernel void fft_1d_vertical(
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    device FFTBloom1DArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // 1. Allocate 8 KB of fast on-chip shared memory for two-for-one packed RG and B complex numbers
    threadgroup float2 s_rg[kFFTBloomSize];
    threadgroup float2 s_b[kFFTBloomSize];
    ExecuteFFT1D(true, threadgroup_pos.x, tid, args, s_rg, s_b);
}

kernel void fft_1d_horizontal(
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    device FFTBloom1DArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    threadgroup float2 s_rg[kFFTBloomSize];
    threadgroup float2 s_b[kFFTBloomSize];
    ExecuteFFT1D(false, threadgroup_pos.y, tid, args, s_rg, s_b);
}

kernel void fft_spectral_multiply(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomSpectralArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    uint2 mirrorGid = uint2((kFFTBloomSize - gid.x) & (kFFTBloomSize - 1), (kFFTBloomSize - gid.y) & (kFFTBloomSize - 1));

    float4 s = args.sceneTexture.read(gid);
    float4 sMirror = args.sceneTexture.read(mirrorGid);
    float4 k = args.kernelTexture.read(gid);
    float4 kMirror = args.kernelTexture.read(mirrorGid);

    float2 sR, sG, kR, kG;
    UnpackTwoForOne(s.xy, sMirror.xy, sR, sG);
    UnpackTwoForOne(k.xy, kMirror.xy, kR, kG);
    float2 sB = s.zw;
    float2 kB = k.zw;

    // DC component at (0, 0) for each channel: Re(Z_RG) = DC_R, Im(Z_RG) = DC_G, Re(Z_B) = DC_B
    float3 kernelDC = max(args.kernelTexture.read(uint2(0, 0)).xyz, float3(1e-4f));
    float3 invDC = 1.0f / kernelDC;

    // Complex multiplication in frequency domain: (A + Bi)(C + Di)
    float2 outR = ComplexMul(sR, kR) * invDC.r;
    float2 outG = ComplexMul(sG, kG) * invDC.g;
    float2 outB = ComplexMul(sB, kB) * invDC.b;

    // Re-pack R and G for two-for-one 2D IFFT: Out_RG = Out_R + i * Out_G
    float2 outRG = float2(outR.x - outG.y, outR.y + outG.x);
    args.outTexture.write(float4(outRG, outB), gid);
}
