//
// RT64
//

struct ShadowApplyCB {
    float4x4 lightViewProj;
    float4 screenScaleOffset;
    float4 resolutionShadow;
    float4 lightDirectionBias;
    float4 shadowTint;
};

[[vk::push_constant]] ConstantBuffer<ShadowApplyCB> gConstants : register(b0, space0);

Texture2D<float> gShadowMap : register(t1);

#if defined(MULTISAMPLING)
Texture2DMS<float> gSceneDepth : register(t2);

float loadSceneDepth(int2 pixelPos) {
    return gSceneDepth.Load(pixelPos, 0);
}
#else
Texture2D<float> gSceneDepth : register(t2);

float loadSceneDepth(int2 pixelPos) {
    return gSceneDepth.Load(int3(pixelPos, 0));
}
#endif

float4 PSMain(in float4 iPosition : SV_POSITION, in float3 iWorld : TEXCOORD0) : SV_TARGET0 {
    // The depth test already rejects hidden surfaces. Also reject pixels where the scene is farther away than this
    // surface, which happens where a cutout texel was transparent and the background shows through.
    const float depthSlope = max(abs(ddx(iPosition.z)), abs(ddy(iPosition.z)));
    float3 normal = cross(ddx(iWorld), ddy(iWorld));
    if (loadSceneDepth(int2(iPosition.xy)) > (iPosition.z + depthSlope * 2.0f + 2e-6f)) {
        discard;
    }

    float ndotl = 1.0f;
    float normalLen2 = dot(normal, normal);
    float3 lightDir = gConstants.lightDirectionBias.xyz;
    float lightLen2 = dot(lightDir, lightDir);
    float3 offsetNormal = float3(0.0f, 0.0f, 0.0f);
    if ((normalLen2 > 1e-8f) && (lightLen2 > 1e-8f)) {
        offsetNormal = normal * rsqrt(normalLen2);
        const float facing = dot(offsetNormal, lightDir) * rsqrt(lightLen2);
        ndotl = abs(facing);
        offsetNormal *= (facing < 0.0f) ? -1.0f : 1.0f;
    }

    // Look the shadow up slightly off the surface instead of biasing the depth by a large constant, which would
    // detach shadows from the objects casting them.
    float2 shadowSize = max(gConstants.resolutionShadow.zw, float2(1.0f, 1.0f));
    const float texelWorld = 2.0f / max(length(gConstants.lightViewProj[0].xyz) * shadowSize.x, 1e-8f);
    const float depthPerWorld = length(gConstants.lightViewProj[2].xyz);
    const float3 lookupWorld = iWorld + offsetNormal * texelWorld * lerp(2.0f, 0.75f, ndotl);
    const float depthBias = texelWorld * depthPerWorld + gConstants.lightDirectionBias.w;

    float4 lightClip = mul(gConstants.lightViewProj, float4(lookupWorld, 1.0f));
    float shadow = 1.0f;
    if (lightClip.w > 1e-4f) {
    float3 ndc = lightClip.xyz / lightClip.w;
    float2 uv = float2(ndc.x * 0.5f + 0.5f, -ndc.y * 0.5f + 0.5f);
    if ((uv.x > 0.0f) && (uv.y > 0.0f) && (uv.x < 1.0f) && (uv.y < 1.0f) && (ndc.z > 0.0f) && (ndc.z < 1.0f)) {
        float2 texel = 1.0f / shadowSize;
        float lit = 0.0f;
        [unroll]
        for (int y = -1; y <= 1; y++) {
            [unroll]
            for (int x = -1; x <= 1; x++) {
                int2 coord = int2((uv + float2(x, y) * texel) * shadowSize);
                coord = clamp(coord, int2(0, 0), int2(shadowSize) - 1);
                float closest = gShadowMap.Load(int3(coord, 0));
                lit += (ndc.z <= closest + depthBias) ? 1.0f : 0.0f;
            }
        }
        shadow = lit / 9.0f;
    }
    }

    float3 shade = lerp(gConstants.shadowTint.www, float3(1.0f, 1.0f, 1.0f), saturate(ndotl)) * lerp(gConstants.shadowTint.rgb, float3(1.0f, 1.0f, 1.0f), saturate(shadow));
    return float4(shade, 1.0f);
}
