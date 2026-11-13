//
// RT64
//

#include "Color.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

#include "shared/rt64_raytracing_params.h"

// The volumetric fog and god rays over the final 3D image, after the draws rasterized on top of the path traced image
// (blended water layers, effects), so they're in the fog too. It shares PostProcessPS's descriptor set.
ConstantBuffer<RaytracingParams> RtParams : register(b0);
SamplerState gSampler : register(s4);
Texture2D<float4> gVolumetric : register(t6);

// The fog is blurred over about 5x5 pixels (bilinear taps 1.5 pixels apart). Neighbors whose fog differs much from
// the center's, the fog in front of another surface, count less, so it stays sharp at silhouettes.
static const float kFogBlurSpacing = 1.5f;
static const float kFogEdgeSharpness = 30.0f;

float4 volumetricFog(float2 uv) {
    const float4 center = gVolumetric.SampleLevel(gSampler, uv, 0);
    if ((center.a >= 0.9999f) && all(center.rgb <= 1e-5f)) {
        return center;
    }

    uint width, height;
    gVolumetric.GetDimensions(width, height);
    const float2 spacing = kFogBlurSpacing / float2(max(width, 1u), max(height, 1u));
    float4 sum = center;
    float weightSum = 1.0f;
    [unroll]
    for (int y = -1; y <= 1; y++) {
        [unroll]
        for (int x = -1; x <= 1; x++) {
            if ((x == 0) && (y == 0)) {
                continue;
            }

            const float4 tap = gVolumetric.SampleLevel(gSampler, uv + float2(x, y) * spacing, 0);
            const float glowDifference = abs(max(tap.r, max(tap.g, tap.b)) - max(center.r, max(center.g, center.b)));
            const float weight = exp(-(abs(tap.a - center.a) + glowDifference) * kFogEdgeSharpness);
            sum += tap * weight;
            weightSum += weight;
        }
    }

    return sum / weightSum;
}

// The air in shadow darkens the image (alpha) and the glow brightens it like light added on a screen, so neither
// pushes colors past white: color * a * (1 - glow) + glow, blended as source + destination * second source.
void PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0
    , [[vk::location(0)]] [[vk::index(0)]] out float4 glowOut : SV_TARGET0
    , [[vk::location(0)]] [[vk::index(1)]] out float4 keepOut : SV_TARGET1)
{
    const float4 volumetric = volumetricFog(uv - RtParams.pixelJitter * RtParams.resolution.zw);
    const float3 glow = saturate(volumetric.rgb);
    glowOut = float4(glow, 0.0f);
    keepOut = float4(saturate(volumetric.a) * (1.0f - glow), 1.0f);
}
