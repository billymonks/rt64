//
// RT64
//
// Temporal accumulation of the ray traced lighting (Graphics menu "Light Accumulation"). DirectRayGen and
// IndirectRayGen write this frame's samples; each surface point is found in last frame's image (moved back by its
// velocity, with the depth and normal checks of reprojectHistory in RtLibrary.hlsl) and blended with its history.
//
// The history is first clipped to the mean +- RtParams.temporalClipGamma standard deviations of this frame's samples in
// the 3x3 neighborhood. Where the lighting changed (a shadow moving off the ground, a light going out) the old value
// lies outside what the current frame shows and can't linger as a trail; in steady noise the bounds are wide and the
// history is kept.
//
// Also written: the luminance moments of the samples, from which the variance RtDenoiseCS uses to tell noise from
// detail, and the filter guide (shading normal and hit distance).

#include "shared/rt64_raytracing_params.h"

ConstantBuffer<RaytracingParams> RtParams : register(b0);
RWTexture2D<float> gDepth : register(u1);
RWTexture2D<float4> gShadingPosition : register(u2);
RWTexture2D<float4> gShadingNormal : register(u3);
RWTexture2D<float4> gNormalRoughness : register(u4);
RWTexture2D<float4> gPrevNormalRoughness : register(u5);
RWTexture2D<float4> gRawDirect : register(u6);
RWTexture2D<float4> gRawIndirect : register(u7);
RWTexture2D<float4> gPrevDirect : register(u8);
RWTexture2D<float4> gPrevIndirect : register(u9);
RWTexture2D<float4> gPrevMoments : register(u10);
RWTexture2D<float4> gDirectOut : register(u11);
RWTexture2D<float4> gIndirectOut : register(u12);
RWTexture2D<float4> gMomentsOut : register(u13);
RWTexture2D<float4> gVarianceOut : register(u14);
RWTexture2D<float4> gGuideOut : register(u15);

// Must match kTemporalDepthTolerance and kTemporalNormalTolerance in RtLibrary.hlsl.
static const float kDepthTolerance = 0.05;
static const float kNormalTolerance = 0.8;
// With less history than this (in frames), the variance comes from the neighborhood rather than the pixel's moments.
static const float kMomentsMinHistory = 4.0;
static const float kMomentsMinBlend = 0.2;

// Same camera ray as cameraRay in RtLibrary.hlsl.
bool cameraRay(float2 p, out float3 origin, out float3 direction) {
    float2 vpSize = max(RtParams.viewport.zw, float2(1.0, 1.0));
    float2 vp = (p - RtParams.viewport.xy) / vpSize;
    float ndcX = vp.x * 2.0 - 1.0;
    float clipY = -lerp(-RtParams.nearDist, RtParams.nearDist, vp.y);
    float4 midH = mul(RtParams.viewI, float4(ndcX, clipY, 0.0, 1.0));
    float3 midPos = midH.xyz / midH.w;
    float4 originH = mul(RtParams.viewI, float4(0.0, 0.0, 1.0, 0.0));
    if (abs(originH.w) > 1e-12) {
        origin = originH.xyz / originH.w;
    }
    else {
        float4 nearH = mul(RtParams.viewI, float4(ndcX, clipY, -1.0, 1.0));
        origin = nearH.xyz / nearH.w;
    }

    direction = midPos - origin;
    float len2 = dot(direction, direction);
    if (!(len2 > 1e-12) || any(!isfinite(origin)) || any(!isfinite(direction))) {
        return false;
    }

    direction *= rsqrt(len2);
    return true;
}

float luminance(float3 color) {
    return dot(color, float3(0.3, 0.59, 0.11));
}

// The indirect term's fill and ambient occlusion together, for its variance.
float indirectLuminance(float4 indirect) {
    return luminance(indirect.rgb) + indirect.a;
}

bool reprojectHistory(int2 pix, int2 size, out int2 previousPixel) {
    previousPixel = int2(0, 0);
    float3 origin, direction;
    if ((RtParams.historyReset != 0) || !cameraRay(float2(pix) + 0.5 + RtParams.pixelJitter, origin, direction)) {
        return false;
    }

    const float3 position = origin + direction * gDepth[pix];
    const float3 velocity = gShadingPosition[pix].xyz;
    const float4 clip = mul(RtParams.prevViewProj, float4(position - velocity, 1.0));
    if (clip.w <= 1e-4) {
        return false;
    }

    const float2 ndc = clip.xy / clip.w;
    const float2 vp = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y / (2.0 * RtParams.nearDist));
    previousPixel = int2(floor(RtParams.viewport.xy + vp * RtParams.viewport.zw));
    if (any(previousPixel < 0) || any(previousPixel >= size)) {
        return false;
    }

    const float4 previous = gPrevNormalRoughness[previousPixel];
    if (previous.w <= 0.0) {
        return false;
    }

    float3 geometricNormal = gNormalRoughness[pix].xyz * 2.0 - 1.0;
    geometricNormal *= rsqrt(max(dot(geometricNormal, geometricNormal), 1e-8));
    const bool depthMatches = abs(previous.w - clip.w) < (clip.w * kDepthTolerance + 0.5);
    const bool normalMatches = dot(previous.xyz * 2.0 - 1.0, geometricNormal) > kNormalTolerance;
    return depthMatches && normalMatches;
}

[numthreads(8, 8, 1)]
void CSMain(uint2 threadId : SV_DispatchThreadID) {
    const int2 size = int2(RtParams.resolution.xy);
    const int2 pix = int2(threadId);
    if (any(pix >= size)) {
        return;
    }

    // The direct light pass writes alpha 0 where the camera ray missed.
    const float4 rawDirect = gRawDirect[pix];
    const float4 rawIndirect = gRawIndirect[pix];
    if (rawDirect.a <= 0.5) {
        gDirectOut[pix] = float4(1.0, 1.0, 1.0, 0.0);
        gIndirectOut[pix] = float4(0.0, 0.0, 0.0, 1.0);
        gMomentsOut[pix] = 0.0;
        gVarianceOut[pix] = 0.0;
        gGuideOut[pix] = 0.0;
        return;
    }

    const float4 normalRoughness = gNormalRoughness[pix];
    float3 shadingNormal = gShadingNormal[pix].xyz;
    const float shadingLength2 = dot(shadingNormal, shadingNormal);
    shadingNormal = (shadingLength2 > 1e-8) ? (shadingNormal * rsqrt(shadingLength2)) : normalize(normalRoughness.xyz * 2.0 - 1.0);
    // The guide's depth is the full precision hit distance along the camera ray.
    gGuideOut[pix] = float4(shadingNormal, max(gDepth[pix], 1e-3));

    // F10 debug views show this frame's values as they are.
    if (RtParams.debugView != 0) {
        gDirectOut[pix] = float4(rawDirect.rgb, 1.0);
        gIndirectOut[pix] = rawIndirect;
        gMomentsOut[pix] = 0.0;
        gVarianceOut[pix] = 0.0;
        return;
    }

    // This frame's samples around the pixel, on surfaces only.
    float3 sumDirect = 0.0;
    float3 sumDirect2 = 0.0;
    float4 sumIndirect = 0.0;
    float4 sumIndirect2 = 0.0;
    float2 sumLuminance = 0.0;
    float2 sumLuminance2 = 0.0;
    float count = 0.0;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            const int2 q = clamp(pix + int2(x, y), int2(0, 0), size - 1);
            const float4 direct = gRawDirect[q];
            if (direct.a > 0.5) {
                const float4 indirect = gRawIndirect[q];
                const float2 l = float2(luminance(direct.rgb), indirectLuminance(indirect));
                sumDirect += direct.rgb;
                sumDirect2 += direct.rgb * direct.rgb;
                sumIndirect += indirect;
                sumIndirect2 += indirect * indirect;
                sumLuminance += l;
                sumLuminance2 += l * l;
                count += 1.0;
            }
        }
    }

    const float3 meanDirect = sumDirect / count;
    const float3 sigmaDirect = sqrt(max(sumDirect2 / count - meanDirect * meanDirect, 0.0));
    const float4 meanIndirect = sumIndirect / count;
    const float4 sigmaIndirect = sqrt(max(sumIndirect2 / count - meanIndirect * meanIndirect, 0.0));
    const float2 meanLuminance = sumLuminance / count;
    const float2 spatialVariance = max(sumLuminance2 / count - meanLuminance * meanLuminance, 0.0);

    const float2 rawLuminance = float2(luminance(rawDirect.rgb), indirectLuminance(rawIndirect));
    float4 moments = float4(rawLuminance.x, rawLuminance.x * rawLuminance.x, rawLuminance.y, rawLuminance.y * rawLuminance.y);
    float3 direct = rawDirect.rgb;
    float4 indirect = rawIndirect;
    float historyLength = 1.0;
    int2 previousPixel;
    if (reprojectHistory(pix, size, previousPixel)) {
        const float4 previousDirect = gPrevDirect[previousPixel];
        historyLength = min(previousDirect.a + 1.0, RtParams.temporalMaxHistory);
        float3 historyDirect = previousDirect.rgb;
        float4 historyIndirect = gPrevIndirect[previousPixel];
        const float gamma = RtParams.temporalClipGamma;
        if (gamma > 0.0) {
            historyDirect = clamp(historyDirect, meanDirect - gamma * sigmaDirect, meanDirect + gamma * sigmaDirect);
            historyIndirect = clamp(historyIndirect, meanIndirect - gamma * sigmaIndirect, meanIndirect + gamma * sigmaIndirect);
        }

        const float blend = max(1.0 / historyLength, RtParams.temporalMinBlend);
        // A sudden brightening (a spell burst, a hit spark) is taken up quickly so it doesn't lag behind.
        const float directBlend = (luminance(meanDirect) > (luminance(historyDirect) * 1.5 + 0.05)) ? max(blend, 0.5) : blend;
        direct = lerp(historyDirect, rawDirect.rgb, directBlend);
        indirect = lerp(historyIndirect, rawIndirect, blend);
        moments = lerp(gPrevMoments[previousPixel], moments, max(1.0 / historyLength, kMomentsMinBlend));
    }

    // Variance of the blended value: the variance of one frame's samples over the number of frames it averages.
    float2 variance = (historyLength >= kMomentsMinHistory) ? max(moments.yw - moments.xz * moments.xz, 0.0) : spatialVariance;
    const float minBlend = clamp(RtParams.temporalMinBlend, 0.01, 1.0);
    variance /= min(historyLength, (2.0 - minBlend) / minBlend);

    gDirectOut[pix] = float4(direct, historyLength);
    gIndirectOut[pix] = indirect;
    gMomentsOut[pix] = moments;
    gVarianceOut[pix] = float4(variance, 0.0, 0.0);
}
