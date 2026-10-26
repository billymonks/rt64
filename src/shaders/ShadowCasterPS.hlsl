//
// RT64
//

float PSMain(in float4 position : SV_POSITION) : SV_Target {
    const float slope = max(abs(ddx(position.z)), abs(ddy(position.z)));
    return saturate(position.z + slope * 0.5f);
}
