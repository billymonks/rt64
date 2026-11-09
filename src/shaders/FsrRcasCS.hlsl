//
// RT64
//
// AMD FidelityFX Super Resolution 1 robust contrast adaptive sharpening (RCAS), run on the EASU output at the display
// resolution.

#define FFX_GPU 1
#define FFX_HLSL 1
#define FFX_HALF 0
#define FSR_RCAS_F 1

#include "ffx_core.h"

struct RcasCB {
    uint4 con;
    uint2 outputSize;
    uint2 padding;
};

[[vk::push_constant]] ConstantBuffer<RcasCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
RWTexture2D<float4> gOutput : register(u2);

FfxFloat32x4 FsrRcasLoadF(FfxInt32x2 p) {
    return gInput.Load(int3(clamp(p, int2(0, 0), int2(gConstants.outputSize) - 1), 0));
}

void FsrRcasInputF(inout FfxFloat32 r, inout FfxFloat32 g, inout FfxFloat32 b) { }

#include "ffx_fsr1.h"

[numthreads(8, 8, 1)]
void CSMain(uint2 coord : SV_DispatchThreadID) {
    if (any(coord >= gConstants.outputSize)) {
        return;
    }

    FfxFloat32 r, g, b;
    FsrRcasF(r, g, b, coord, gConstants.con);
    gOutput[coord] = float4(saturate(float3(r, g, b)), 1.0f);
}
