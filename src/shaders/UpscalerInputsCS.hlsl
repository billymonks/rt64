//
// RT64
//
// Inputs of the temporal upscaler at the ray tracing resolution, from the primary hits: motion vectors, device depth
// and the reactive mask.
//
// Motion vectors are in UV units, from the current pixel to where its surface was last frame, without jitter: a hit
// point is moved back by its world space velocity and projected with last frame's view-projection. Pixels that missed
// the scene move with the camera only (the point at infinity along the ray).
//
// The reactive mask tells the upscaler to trust the current frame where history would be wrong even with correct
// motion: water (its texture scrolls and its ripples move without any motion vector), lighting that changed on a surface
// that didn't move (emissive lights fading, flashes, shadows of moving casters), and lighting that was just reset.

#include "shared/rt64_raytracing_params.h"

ConstantBuffer<RaytracingParams> RtParams : register(b0);
RWTexture2D<float> gDepth : register(u1);
RWTexture2D<float4> gShadingPosition : register(u2);
RWTexture2D<float4> gDirectLightAccum : register(u3);
RWTexture2D<float4> gPrevNormalRoughness : register(u4);
Texture2D<float4> gDiffuse : register(t5);
Texture2D<float4> gRefraction : register(t6);
Texture2D<float4> gColor : register(t7);
Texture2D<float4> gPrevColor : register(t8);
RWTexture2D<float2> gMotionVectors : register(u9);
RWTexture2D<float> gDeviceDepth : register(u10);
RWTexture2D<float> gReactiveMask : register(u11);
SamplerState gSampler : register(s12);

// Must match kTemporalDepthTolerance in RtLibrary.hlsl.
static const float kDepthTolerance = 0.05;
static const float kWaterReactive = 0.9;
// A change of the composed brightness above this, on a surface that didn't move, is lighting that changed.
static const float kShadingChangeThreshold = 0.08;
static const float kShadingChangeReactive = 0.7;
// Lighting with less history than this (in frames) was just reset or disoccluded.
static const float kFreshHistoryFrames = 3.0;
static const float kFreshHistoryReactive = 0.3;

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

// Ray tracing pixel position of a clip space point, as in reprojectHistory in RtLibrary.hlsl.
float2 clipToPixel(float4 clip) {
    const float2 ndc = clip.xy / clip.w;
    const float2 vp = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y / (2.0 * RtParams.nearDist));
    return RtParams.viewport.xy + vp * RtParams.viewport.zw;
}

float luminance(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

[numthreads(8, 8, 1)]
void CSMain(uint2 pix : SV_DispatchThreadID) {
    if ((pix.x >= uint(RtParams.resolution.x)) || (pix.y >= uint(RtParams.resolution.y))) {
        return;
    }

    const float2 samplePixel = float2(pix) + 0.5 + RtParams.pixelJitter;
    float2 motion = 0.0;
    float deviceDepth = 1.0;
    float reactive = 0.0;

    float3 origin, direction;
    const bool validRay = cameraRay(samplePixel, origin, direction);
    const bool hit = validRay && (gDiffuse.Load(int3(pix, 0)).a > 1e-4);
    if (hit) {
        const float3 position = origin + direction * gDepth[pix];
        const float3 velocity = gShadingPosition[pix].xyz;
        const float4 clip = mul(RtParams.viewProj, float4(position, 1.0));
        if (clip.w > 1e-4) {
            // The game's projections are OpenGL style: NDC depth from -1 (near) to 1 (far).
            deviceDepth = saturate((clip.z / clip.w) * 0.5 + 0.5);
        }

        const float4 previousClip = mul(RtParams.prevViewProj, float4(position - velocity, 1.0));
        if ((clip.w > 1e-4) && (previousClip.w > 1e-4)) {
            const float2 previousPixel = clipToPixel(previousClip);
            motion = (previousPixel - clipToPixel(clip)) * RtParams.resolution.zw;

            // Lighting that changed on a surface that is where it was last frame.
            const int2 previousTexel = int2(floor(previousPixel));
            if (all(previousTexel >= 0) && all(previousTexel < int2(RtParams.resolution.xy))) {
                const float previousViewDepth = gPrevNormalRoughness[previousTexel].w;
                if ((previousViewDepth > 0.0) && (abs(previousViewDepth - previousClip.w) < (previousClip.w * kDepthTolerance + 0.5))) {
                    const float current = luminance(gColor.Load(int3(pix, 0)).rgb);
                    const float previous = luminance(gPrevColor.SampleLevel(gSampler, previousPixel * RtParams.resolution.zw, 0).rgb);
                    reactive = max(reactive, saturate((abs(current - previous) - kShadingChangeThreshold) * 4.0) * kShadingChangeReactive);
                }
            }
        }

        const float historyLength = gDirectLightAccum[pix].a;
        if (historyLength < kFreshHistoryFrames) {
            reactive = max(reactive, saturate((kFreshHistoryFrames - historyLength) / (kFreshHistoryFrames - 1.0)) * kFreshHistoryReactive);
        }

        // The water pass (RefractionRayGen) writes its layer with the water's coverage in alpha, for opaque water (path
        // traced draws with kMaterialWaterPassFlag) and blended water in front of the scene alike.
        if (gRefraction.Load(int3(pix, 0)).a > 1e-3) {
            reactive = max(reactive, kWaterReactive);
        }
    }
    else if (validRay) {
        const float4 clip = mul(RtParams.viewProj, float4(direction, 0.0));
        const float4 previousClip = mul(RtParams.prevViewProj, float4(direction, 0.0));
        if ((clip.w > 1e-6) && (previousClip.w > 1e-6)) {
            motion = (clipToPixel(previousClip) - clipToPixel(clip)) * RtParams.resolution.zw;
        }
    }

    if (any(!isfinite(motion))) {
        motion = 0.0;
    }

    gMotionVectors[pix] = motion;
    gDeviceDepth[pix] = deviceDepth;
    gReactiveMask[pix] = reactive;
}
