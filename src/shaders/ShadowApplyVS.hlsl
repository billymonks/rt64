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

void VSMain(in float4 iScreen : POSITION, in float4 iWorld : TEXCOORD0, out float4 oPosition : SV_POSITION, out float3 oWorld : TEXCOORD0) {
    float4 ndcPos = iScreen;
    ndcPos.xy -= gConstants.resolutionShadow.xy * 0.5f;
    ndcPos.xy /= float2(gConstants.resolutionShadow.x * 0.5f, gConstants.resolutionShadow.y * -0.5f);
    ndcPos.xyz *= ndcPos.w;
    ndcPos.xy = (ndcPos.xy * gConstants.screenScaleOffset.xy) + gConstants.screenScaleOffset.zw * ndcPos.w;
    oPosition = ndcPos;
    oWorld = iWorld.xyz;
}
