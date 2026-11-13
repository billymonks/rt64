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
Texture2D<float4> gVolumetric : register(t8);

// Strength of the ray traced ambient occlusion, from 0 (off) to 1.
static const float kAmbientOcclusionStrength = 0.75f;

// Where a primary ray hit the scene (alpha set), gDiffuse holds the path traced material color and gTransparent the
// fog color and amount of that material's blender. The direct light factor is 1 on fully lit surfaces; the
// indirect fill only lifts what the direct light leaves dark, so the product stays in the game's display space.
// Ambient occlusion (indirect alpha) darkens the ambient part of the light factor (gFlow.x, the value of a surface
// that neither faces nor sees the light) and the fill, but not direct sunlight. Everything else (sky, background)
// is the rasterized frame and passes through unchanged. The volumetric fog goes over the final image (VolumetricPS).
float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    float4 diffuse = gDiffuse.SampleLevel(gSampler, uv, 0);
    if (diffuse.a > EPSILON) {
        float3 directLight = gDirectLight.SampleLevel(gSampler, uv, 0).rgb;
        float4 indirect = gIndirectLight.SampleLevel(gSampler, uv, 0);
        float ambient = gFlow.SampleLevel(gSampler, uv, 0).x;
        float4 fog = gTransparent.SampleLevel(gSampler, uv, 0);
        float occlusion = lerp(1.0f, saturate(indirect.a), kAmbientOcclusionStrength);
        float3 ambientPart = min(directLight, ambient.xxx);
        float3 light = ambientPart * occlusion + (directLight - ambientPart) + (1.0f - saturate(directLight)) * indirect.rgb * occlusion;
        // Specular light of the surface material (ReflectionRayGen) adds on top; alpha is what remains of the diffuse.
        float4 specular = gReflection.SampleLevel(gSampler, uv, 0);
        float3 lit = saturate(diffuse.rgb * light * specular.a + specular.rgb);
        float3 color = lerp(lit, fog.rgb, saturate(fog.a));
        // Water in front of the surface (RefractionRayGen, Advanced Water Rendering): alpha is how much of the surface
        // it hides, rgb its color, reflection and glint. Zero everywhere else.
        float4 water = gRefraction.SampleLevel(gSampler, uv, 0);
        color = saturate(color * (1.0f - saturate(water.a)) + water.rgb);
        return float4(color, 1.0f);
    }
    else {
        // Blended water seen against the sky adds its reflection on the rasterized frame.
        float4 water = gRefraction.SampleLevel(gSampler, uv, 0);
        float3 color = saturate(diffuse.rgb * (1.0f - saturate(water.a)) + water.rgb);
        return float4(color, 1.0f);
    }
}
