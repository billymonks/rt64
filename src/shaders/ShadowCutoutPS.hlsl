//
// RT64
//

#include "shared/rt64_color_combiner.h"
#include "shared/rt64_raster_params.h"

#include "FbRendererCommon.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"

// Same push constants as the raster shaders, so it shares their pipeline layout and descriptor sets.
[[vk::push_constant]] ConstantBuffer<RasterParams> gConstants : register(b0, space0);

// Writes the shadow depth only where the draw would leave a visible texel.
float PSMain(in float4 vertexPosition : SV_POSITION, in float2 vertexUV : TEXCOORD, in float4 vertexColor : COLOR0) : SV_Target {
    const uint instanceIndex = instanceRenderIndices[gConstants.renderIndex].instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    uint randomSeed = initRand(FrParams.frameCount, instanceIndex, 16);

    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction;
    computeLOD(otherMode, instanceRenderIndices[gConstants.renderIndex].rdpTileCount, instanceRDPParams[instanceIndex].primLOD, 1.0f, ddx(vertexUV.x), ddy(vertexUV.y), tileIndex0, tileIndex1, lodFraction);

    float4 texVal0 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 texVal1 = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = instanceRenderIndices[gConstants.renderIndex].rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        texVal0 = sampleTexture(otherMode, rp.flags, vertexUV, ddx(vertexUV), ddy(vertexUV), rdpTile, GPUTiles[globalTileIndex], false);
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = instanceRenderIndices[gConstants.renderIndex].rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }

        texVal1 = sampleTexture(otherMode, rp.flags, vertexUV, ddx(vertexUV), ddy(vertexUV), rdpTile, GPUTiles[globalTileIndex], oneCycleHardwareBug);
    }

    float4 combinerColor;
    float alphaCompareValue;
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = instanceRDPParams[instanceIndex].primColor;
    ccInputs.shadeColor = vertexColor;
    ccInputs.envColor = instanceRDPParams[instanceIndex].envColor;
    ccInputs.keyCenter = instanceRDPParams[instanceIndex].keyCenter;
    ccInputs.keyScale = instanceRDPParams[instanceIndex].keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = instanceRDPParams[instanceIndex].primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (instanceRDPParams[instanceIndex].convertK[4] / 255.0f);
    ccInputs.K5 = (instanceRDPParams[instanceIndex].convertK[5] / 255.0f);
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    if ((otherMode.alphaCompare() == G_AC_THRESHOLD) && (alphaCompareValue < instanceRDPParams[instanceIndex].blendColor.a)) {
        discard;
    }

    if ((otherMode.alphaCompare() == G_AC_DITHER) && (alphaCompareValue < 0.5f)) {
        discard;
    }

    if (otherMode.cvgXAlpha() && (combinerColor.a < 0.5f)) {
        discard;
    }

    const float slope = max(abs(ddx(vertexPosition.z)), abs(ddy(vertexPosition.z)));
    return saturate(vertexPosition.z + slope * 0.5f);
}
