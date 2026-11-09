//
// RT64
//

#include "Color.hlsli"
#include "Constants.hlsli"
#include "Math.hlsli"

SamplerState gSampler : register(s0);
Texture2D<float4> gFlow : register(t1);
Texture2D<float4> gDiffuse : register(t2);
Texture2D<float4> gDirectLight : register(t3);
Texture2D<float4> gIndirectLight : register(t4);
Texture2D<float4> gReflection : register(t5);
Texture2D<float4> gRefraction : register(t6);
Texture2D<float4> gTransparent : register(t7);

// Where a primary ray hit the scene (alpha set), gDiffuse holds the path traced material color and gTransparent the
// fog color and amount of that material's blender. The direct light factor is 1 on fully lit surfaces; the
// indirect fill only lifts what the direct light leaves dark, so the product stays in the game's display space.
// Everything else (sky, background) is the rasterized frame and passes through unchanged.
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    float4 diffuse = gDiffuse.SampleLevel(gSampler, uv, 0);
    if (diffuse.a > EPSILON) {
        float3 directLight = gDirectLight.SampleLevel(gSampler, uv, 0).rgb;
        float3 indirectFill = gIndirectLight.SampleLevel(gSampler, uv, 0).rgb;
        float4 fog = gTransparent.SampleLevel(gSampler, uv, 0);
        float3 light = directLight + (1.0f - saturate(directLight)) * indirectFill;
        float3 lit = saturate(diffuse.rgb * light);
        return float4(lerp(lit, fog.rgb, saturate(fog.a)), 1.0f);
    }
    else {
        return float4(diffuse.rgb, 1.0f);
    }
}
