#include <metal_stdlib>
#include "ShaderTypes.h"

using namespace metal;

constexpr constant uint kFFTSize = kFFTBloomSize;
constexpr constant uint kFFTStages = 9;
static_assert((1u << kFFTStages) == kFFTSize, "kFFTStages must match kFFTSize");

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

kernel void fft_bloom_prefilter(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomPrefilterArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // Zero padding outside the content region guarantees linear (non-wrapping) convolution
    if (gid.x >= kFFTBloomContentSize || gid.y >= kFFTBloomContentSize) {
        args.outRealTexture.write(float4(0.0f), gid);
        args.outImagTexture.write(float4(0.0f), gid);
        return;
    }

    float threshold = args.threshold;
    float knee = max(args.knee, 1e-4f);
    float maxLuma = max(args.maxLuma, 1.0f);

    // Tent downsample spanning +-1 destination texel. Tents spaced one texel apart form a partition of unity, so every
    // source pixel contributes the same total energy regardless of its sub-texel position (no flicker as content moves).
    int2 srcSize = int2(args.sceneColorTexture.get_width(), args.sceneColorTexture.get_height());
    float2 footprint = float2(srcSize) / float(kFFTBloomContentSize);
    float2 center = (float2(gid) + 0.5f) * footprint;
    int2 pMin = max(int2(floor(center - footprint)), int2(0));
    int2 pMax = min(int2(ceil(center + footprint)), srcSize - 1);

    float3 accum = float3(0.0f);
    for (int y = pMin.y; y <= pMax.y; ++y) {
        float wy = max(1.0f - abs(float(y) + 0.5f - center.y) / footprint.y, 0.0f);
        if (wy <= 0.0f)
            continue;

        for (int x = pMin.x; x <= pMax.x; ++x) {
            float wx = max(1.0f - abs(float(x) + 0.5f - center.x) / footprint.x, 0.0f);
            if (wx <= 0.0f)
                continue;

            float3 color = args.sceneColorTexture.read(uint2(x, y)).rgb;
            accum += BloomBrightPass(color, threshold, knee, maxLuma) * (wx * wy);
        }
    }
    float3 extracted = accum / (footprint.x * footprint.y);

    args.outRealTexture.write(float4(extracted, 1.0f), gid);
    args.outImagTexture.write(float4(0.0f), gid);
}

kernel void fft_psf_generate(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomPSFArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // Wrap coordinates to place kernel impulse center at (0, 0)
    float dx = float(int(gid.x < (kFFTSize / 2) ? gid.x : gid.x - kFFTSize));
    float dy = float(int(gid.y < (kFFTSize / 2) ? gid.y : gid.y - kFFTSize));

    // Pixel aspect correction (content is a square buffer stretched over the screen) to preserve circular geometry
    float dx_screen = dx * args.pixelAspect;
    float r = sqrt(dx_screen * dx_screen + dy * dy);

    // Support radius (in vertical texels) chosen so the PSF never exceeds kFFTBloomKernelRadius texels on either axis
    float supportRadius = float(kFFTBloomKernelRadius - 1) * min(1.0f, args.pixelAspect);
    float t = saturate(r / supportRadius);
    float window = (1.0f - t * t) * (1.0f - t * t);

    // Shape parameters below are authored for a 512-texel screen height; rescale so the on-screen size is resolution independent
    constexpr float kRefScale = float(kFFTSize) / float(kFFTBloomContentSize);
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
    args.outRealTexture.write(float4(psf, 1.0f), gid);
    args.outImagTexture.write(float4(0.0f), gid);
}

kernel void fft_1d_vertical(
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    device FFTBloom1DArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    // 1. Allocate 12 KB of fast on-chip shared memory for R, G, B complex numbers (float2 = real, imag)
    threadgroup float2 s_r[kFFTSize];
    threadgroup float2 s_g[kFFTSize];
    threadgroup float2 s_b[kFFTSize];

    uint col = threadgroup_pos.x;
    uint i0 = tid;
    uint i1 = tid + 256;

    // 2. Hardware bit-reversal permutation: 9-bit reversal via 32-bit reverse_bits() >> 23
    uint rev0 = reverse_bits(i0) >> (32 - kFFTStages);
    uint rev1 = reverse_bits(i1) >> (32 - kFFTStages);

    // 3. Coalesced global texture read, scattered into bit-reversed shared memory slots
    float4 r0 = args.inRealTexture.read(uint2(col, i0));
    float4 im0 = args.inImagTexture.read(uint2(col, i0));
    float4 r1 = args.inRealTexture.read(uint2(col, i1));
    float4 im1 = args.inImagTexture.read(uint2(col, i1));

    s_r[rev0] = float2(r0.r, im0.r);
    s_g[rev0] = float2(r0.g, im0.g);
    s_b[rev0] = float2(r0.b, im0.b);

    s_r[rev1] = float2(r1.r, im1.r);
    s_g[rev1] = float2(r1.g, im1.g);
    s_b[rev1] = float2(r1.b, im1.b);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Forward FFT uses exp(-i*2*pi*k/m), Inverse FFT uses exp(+i*2*pi*k/m)
    float sign = args.inverse ? 1.0f : -1.0f;
    for (uint s = 1; s <= kFFTStages; ++s) {
        uint m = 1u << s; // Sub-DFT size: 2, 4, 8, 16, 32, 64, 128, 256, 512
        uint half_m = m >> 1; // Butterfly stride: 1, 2, 4, 8, 16, 32, 64, 128, 256
        uint k = tid % half_m; // Index within butterfly group (0 .. half_m - 1)
        uint b = tid / half_m; // Which butterfly group this thread belongs to
        uint j = b * m + k; // Top element index E[k]; bottom element O[k] is (j + half_m)，

        float angle = sign * 2.0f * M_PI_F * float(k) / float(m);
        float2 W = float2(cos(angle), sin(angle));  // Twiddle factor W_m^k

        float2 u_r = s_r[j];
        float2 v_r = s_r[j + half_m];
        // Complex multiply: t = W * v
        float2 t_r = float2(W.x * v_r.x - W.y * v_r.y, W.x * v_r.y + W.y * v_r.x);

        float2 u_g = s_g[j];
        float2 v_g = s_g[j + half_m];
        float2 t_g = float2(W.x * v_g.x - W.y * v_g.y, W.x * v_g.y + W.y * v_g.x);

        float2 u_b = s_b[j];
        float2 v_b = s_b[j + half_m];
        float2 t_b = float2(W.x * v_b.x - W.y * v_b.y, W.x * v_b.y + W.y * v_b.x);

        // Butterfly write-back (same for G and B)
        s_r[j] = u_r + t_r;
        s_r[j + half_m] = u_r - t_r;

        s_g[j] = u_g + t_g;
        s_g[j + half_m] = u_g - t_g;

        s_b[j] = u_b + t_b;
        s_b[j + half_m] = u_b - t_b;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = args.inverse ? (1.0f / float(kFFTSize)) : 1.0f;
    float3 out0 = float3(s_r[i0].x * scale, s_g[i0].x * scale, s_b[i0].x * scale);
    float3 out1 = float3(s_r[i1].x * scale, s_g[i1].x * scale, s_b[i1].x * scale);

    // Clamp negative ringing from spectral truncation (light energy cannot be negative)
    if (args.inverse) {
        out0 = max(out0, 0.0f);
        out1 = max(out1, 0.0f);
    }

    args.outRealTexture.write(float4(out0, 1.0f), uint2(col, i0));
    args.outImagTexture.write(float4(s_r[i0].y * scale, s_g[i0].y * scale, s_b[i0].y * scale, 0.0f), uint2(col, i0));
    args.outRealTexture.write(float4(out1, 1.0f), uint2(col, i1));
    args.outImagTexture.write(float4(s_r[i1].y * scale, s_g[i1].y * scale, s_b[i1].y * scale, 0.0f), uint2(col, i1));
}

kernel void fft_1d_horizontal(
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    device FFTBloom1DArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    threadgroup float2 s_r[kFFTSize];
    threadgroup float2 s_g[kFFTSize];
    threadgroup float2 s_b[kFFTSize];

    uint row = threadgroup_pos.y;
    uint i0 = tid;
    uint i1 = tid + 256;

    uint rev0 = reverse_bits(i0) >> (32 - kFFTStages);
    uint rev1 = reverse_bits(i1) >> (32 - kFFTStages);

    float4 r0 = args.inRealTexture.read(uint2(i0, row));
    float4 im0 = args.inImagTexture.read(uint2(i0, row));
    float4 r1 = args.inRealTexture.read(uint2(i1, row));
    float4 im1 = args.inImagTexture.read(uint2(i1, row));

    s_r[rev0] = float2(r0.r, im0.r);
    s_g[rev0] = float2(r0.g, im0.g);
    s_b[rev0] = float2(r0.b, im0.b);

    s_r[rev1] = float2(r1.r, im1.r);
    s_g[rev1] = float2(r1.g, im1.g);
    s_b[rev1] = float2(r1.b, im1.b);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sign = args.inverse ? 1.0f : -1.0f;
    for (uint s = 1; s <= kFFTStages; ++s) {
        uint m = 1u << s;
        uint half_m = m >> 1;
        uint k = tid % half_m;
        uint b = tid / half_m;
        uint j = b * m + k;

        float angle = sign * 2.0f * M_PI_F * float(k) / float(m);
        float2 W = float2(cos(angle), sin(angle));

        float2 u_r = s_r[j];
        float2 v_r = s_r[j + half_m];
        float2 t_r = float2(W.x * v_r.x - W.y * v_r.y, W.x * v_r.y + W.y * v_r.x);

        float2 u_g = s_g[j];
        float2 v_g = s_g[j + half_m];
        float2 t_g = float2(W.x * v_g.x - W.y * v_g.y, W.x * v_g.y + W.y * v_g.x);

        float2 u_b = s_b[j];
        float2 v_b = s_b[j + half_m];
        float2 t_b = float2(W.x * v_b.x - W.y * v_b.y, W.x * v_b.y + W.y * v_b.x);

        s_r[j] = u_r + t_r;
        s_r[j + half_m] = u_r - t_r;

        s_g[j] = u_g + t_g;
        s_g[j + half_m] = u_g - t_g;

        s_b[j] = u_b + t_b;
        s_b[j + half_m] = u_b - t_b;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = args.inverse ? (1.0f / float(kFFTSize)) : 1.0f;
    args.outRealTexture.write(float4(s_r[i0].x * scale, s_g[i0].x * scale, s_b[i0].x * scale, 1.0f), uint2(i0, row));
    args.outImagTexture.write(float4(s_r[i0].y * scale, s_g[i0].y * scale, s_b[i0].y * scale, 0.0f), uint2(i0, row));
    args.outRealTexture.write(float4(s_r[i1].x * scale, s_g[i1].x * scale, s_b[i1].x * scale, 1.0f), uint2(i1, row));
    args.outImagTexture.write(float4(s_r[i1].y * scale, s_g[i1].y * scale, s_b[i1].y * scale, 0.0f), uint2(i1, row));
}

kernel void fft_spectral_multiply(
    uint2 gid [[thread_position_in_grid]],
    device FFTBloomSpectralArguments& args [[buffer(FFTBloomBufferIndex::KernelArguments)]])
{
    float4 sR = args.sceneRealTexture.read(gid);
    float4 sI = args.sceneImagTexture.read(gid);
    float4 kR = args.kernelRealTexture.read(gid);
    float4 kI = args.kernelImagTexture.read(gid);

    // DC component for each channel to normalize total flux to 1.0
    float3 kernelDC = max(args.kernelRealTexture.read(uint2(0, 0)).rgb, float3(1e-4f));
    float3 invDC = 1.0f / kernelDC;

    // Complex multiplication: (A + Bi)(C + Di) = (AC - BD) + (AD + BC)i
    float outReal_R = (sR.r * kR.r - sI.r * kI.r) * invDC.r;
    float outImag_R = (sR.r * kI.r + sI.r * kR.r) * invDC.r;

    float outReal_G = (sR.g * kR.g - sI.g * kI.g) * invDC.g;
    float outImag_G = (sR.g * kI.g + sI.g * kR.g) * invDC.g;

    float outReal_B = (sR.b * kR.b - sI.b * kI.b) * invDC.b;
    float outImag_B = (sR.b * kI.b + sI.b * kR.b) * invDC.b;

    args.outRealTexture.write(float4(outReal_R, outReal_G, outReal_B, 1.0f), gid);
    args.outImagTexture.write(float4(outImag_R, outImag_G, outImag_B, 0.0f), gid);
}
