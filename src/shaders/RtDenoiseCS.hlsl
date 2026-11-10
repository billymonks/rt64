//
// RT64
//
// Edge-aware a-trous wavelet filter of the ray traced lighting (Graphics menu "Denoiser"), one pass per dispatch after
// RtTemporalCS. Each pass blends a 5x5 neighborhood spread out by the pass's step (1, 2, 4, ...), so a few passes cover
// a wide area cheaply. A neighbor counts less the more it differs from the pixel in depth (against the depth slope), in
// shading normal and in lighting. The lighting tolerance follows the pixel's estimated noise: a surface just uncovered
// (little history, high variance) is smoothed widely, while converged lighting keeps its shadow edges. The light
// factors carry no texture (ComposePS multiplies the albedo in afterwards), so texture detail stays sharp.

struct DenoiseCB {
    uint2 resolution;
    uint stepSize;
    uint padding0;
    float luminanceSigma;
    float depthSigma;
    float normalPower;
    float padding1;
};

[[vk::push_constant]] ConstantBuffer<DenoiseCB> gConstants : register(b0);
RWTexture2D<float4> gGuide : register(u1);
RWTexture2D<float4> gDirectIn : register(u2);
RWTexture2D<float4> gIndirectIn : register(u3);
RWTexture2D<float4> gVarianceIn : register(u4);
RWTexture2D<float4> gDirectOut : register(u5);
RWTexture2D<float4> gIndirectOut : register(u6);
RWTexture2D<float4> gVarianceOut : register(u7);

// B3 spline weights of the 5x5 kernel, from the center out.
static const float kKernel[3] = { 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0 };

float luminance(float3 color) {
    return dot(color, float3(0.3, 0.59, 0.11));
}

// Same as in RtTemporalCS.
float indirectLuminance(float4 indirect) {
    return luminance(indirect.rgb) + indirect.a;
}

int2 clampPixel(int2 p, int2 size) {
    return clamp(p, int2(0, 0), size - 1);
}

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    const int2 size = int2(gConstants.resolution);
    const int2 pix = int2(threadId);
    if (any(pix >= size)) {
        return;
    }

    const float4 guide = gGuide[pix];
    const float4 direct = gDirectIn[pix];
    const float4 indirect = gIndirectIn[pix];
    const float4 centerVariance = gVarianceIn[pix];
    if (guide.w <= 0.0) {
        gDirectOut[pix] = direct;
        gIndirectOut[pix] = indirect;
        gVarianceOut[pix] = centerVariance;
        return;
    }

    // The pixel's variance, smoothed over 3x3 so one noisy estimate doesn't leave a blotch.
    float2 variance = 0.0;
    [unroll]
    for (int vy = -1; vy <= 1; vy++) {
        [unroll]
        for (int vx = -1; vx <= 1; vx++) {
            const float weight = ((vx == 0) ? 0.5 : 0.25) * ((vy == 0) ? 0.5 : 0.25);
            variance += gVarianceIn[clampPixel(pix + int2(vx, vy), size)].xy * weight;
        }
    }

    // Depth slope, from the smaller one-sided difference so a silhouette next to the pixel doesn't read as a slope.
    const float depth = guide.w;
    const float left = gGuide[clampPixel(pix - int2(1, 0), size)].w;
    const float right = gGuide[clampPixel(pix + int2(1, 0), size)].w;
    const float up = gGuide[clampPixel(pix - int2(0, 1), size)].w;
    const float down = gGuide[clampPixel(pix + int2(0, 1), size)].w;
    const float2 slope = float2(min(abs(right - depth), abs(depth - left)), min(abs(down - depth), abs(depth - up)));

    const float2 centerLuminance = float2(luminance(direct.rgb), indirectLuminance(indirect));
    const float2 tolerance = gConstants.luminanceSigma * sqrt(variance) + 1e-4;
    const float centerWeight = kKernel[0] * kKernel[0];
    float3 directSum = direct.rgb * centerWeight;
    float4 indirectSum = indirect * centerWeight;
    float2 weightSum = centerWeight;
    float2 varianceSum = centerVariance.xy * centerWeight * centerWeight;
    const int step = int(gConstants.stepSize);
    [unroll]
    for (int y = -2; y <= 2; y++) {
        [unroll]
        for (int x = -2; x <= 2; x++) {
            if ((x == 0) && (y == 0)) {
                continue;
            }

            const int2 offset = int2(x, y) * step;
            const int2 q = pix + offset;
            if (any(q < 0) || any(q >= size)) {
                continue;
            }

            const float4 guideQ = gGuide[q];
            if (guideQ.w <= 0.0) {
                continue;
            }

            const float depthWeight = exp(-abs(guideQ.w - depth) / (gConstants.depthSigma * dot(slope, abs(float2(offset))) + depth * 0.002 + 1e-3));
            const float normalWeight = pow(saturate(dot(guide.xyz, guideQ.xyz)), gConstants.normalPower);
            const float geometryWeight = kKernel[abs(x)] * kKernel[abs(y)] * depthWeight * normalWeight;
            if (geometryWeight < 1e-6) {
                continue;
            }

            const float4 directQ = gDirectIn[q];
            const float4 indirectQ = gIndirectIn[q];
            const float2 luminanceQ = float2(luminance(directQ.rgb), indirectLuminance(indirectQ));
            const float2 weight = geometryWeight * exp(-abs(luminanceQ - centerLuminance) / tolerance);
            directSum += directQ.rgb * weight.x;
            indirectSum += indirectQ * weight.y;
            weightSum += weight;
            varianceSum += gVarianceIn[q].xy * weight * weight;
        }
    }

    // Alpha of the direct light (its history length) passes through.
    gDirectOut[pix] = float4(directSum / weightSum.x, direct.a);
    gIndirectOut[pix] = indirectSum / weightSum.y;
    gVarianceOut[pix] = float4(varianceSum / (weightSum * weightSum), 0.0, 0.0);
}
