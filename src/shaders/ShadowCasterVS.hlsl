//
// RT64
//

struct ShadowCasterCB {
    float4x4 lightViewProj;
};

[[vk::push_constant]] ConstantBuffer<ShadowCasterCB> gConstants : register(b0, space0);

void VSMain(in float4 iWorld : POSITION, out float4 oPosition : SV_POSITION) {
    oPosition = mul(gConstants.lightViewProj, float4(iWorld.xyz, 1.0f));
}
