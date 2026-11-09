//
// RT64
//
// AMD FidelityFX Super Resolution 1 edge adaptive spatial upsampling (EASU): scales the composed ray traced image from
// its render resolution up to the display resolution. The input is in display space (gamma encoded, 0 to 1), which
// is what EASU expects.

#define FFX_GPU 1
#define FFX_HLSL 1
#define FFX_HALF 0
#define FFX_FSR_EASU_FLOAT 1

#include "ffx_core.h"

struct EasuCB {
    uint4 con0;
    uint4 con1;
    uint4 con2;
    uint4 con3;
    uint2 outputSize;
    uint2 padding;
};

[[vk::push_constant]] ConstantBuffer<EasuCB> gConstants : register(b0);
Texture2D<float4> gInput : register(t1);
RWTexture2D<float4> gOutput : register(u2);
SamplerState gSampler : register(s3);

FfxFloat32x4 FsrEasuRF(FfxFloat32x2 p) {
    return gInput.GatherRed(gSampler, p);
}

FfxFloat32x4 FsrEasuGF(FfxFloat32x2 p) {
    return gInput.GatherGreen(gSampler, p);
}

FfxFloat32x4 FsrEasuBF(FfxFloat32x2 p) {
    return gInput.GatherBlue(gSampler, p);
}

#include "ffx_fsr1.h"

[numthreads(8, 8, 1)]
void CSMain(uint2 coord : SV_DispatchThreadID) {
    if (any(coord >= gConstants.outputSize)) {
        return;
    }

    FfxFloat32x3 color;
    ffxFsrEasuFloat(color, coord, gConstants.con0, gConstants.con1, gConstants.con2, gConstants.con3);
    gOutput[coord] = float4(saturate(color), 1.0f);
}
