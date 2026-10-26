//
// RT64
//

// Positions arrive already in the light's clip space.
void VSMain(in float4 iPosition : POSITION, in float2 iUV : TEXCOORD, in float4 iColor : COLOR, out float4 oPosition : SV_POSITION, out float2 oUV : TEXCOORD, out float4 oColor : COLOR0) {
    oPosition = iPosition;
    oUV = iUV;
    oColor = iColor;
}
