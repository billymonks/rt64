//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    enum class VisualizationMode {
        Final,
        ShadingPosition,
        ShadingNormal,
        ShadingSpecular,
        Diffuse,
        InstanceId,
        DirectLightRaw,
        DirectLightFiltered,
        IndirectLightRaw,
        IndirectLightFiltered,
        Reflection,
        Refraction,
        Transparent,
        Flow,
        ReactiveMask,
        LockMask,
        Depth,
        Count
    };

    struct RaytracingParams {
        float4x4 view;
        float4x4 viewI;
        float4x4 projection;
        float4x4 projectionI;
        float4x4 viewProj;
        float4x4 prevViewProj;
        float4 cameraU;
        float4 cameraV;
        float4 cameraW;
        float4 viewport;
        float4 resolution;
        float4 ambientBaseColor;
        float4 ambientNoGIColor;
        float4 eyeLightDiffuseColor;
        float4 eyeLightSpecularColor;
        float2 pixelJitter;
        float fovRadians;
        float nearDist;
        float farDist;
        float giDiffuseStrength;
        float giBackgroundStrength;
        float motionBlurStrength;
        float tonemapExposure;
        float tonemapWhite;
        float tonemapBlack;
        uint diSamples;
        uint giSamples;
        uint diReproject;
        uint giReproject;
        uint binaryLockMask;
        uint maxLights;
        uint motionBlurSamples;
        uint lightsCount;
        uint interleavedRastersCount;
        VisualizationMode visualizationMode;
        // Angle between the camera rays of two neighboring pixels, for texture LOD at ray hits.
        float pixelSpreadAngle;
        // The rasterizer culls a triangle seen by a camera ray when dot(geometric normal, ray direction) has this sign.
        float cullWindingSign;
        // SceneLights[0, emissiveLightStart) are the scene lights (sun, moon or room light first); the emissive point
        // lights of flames and spell effects follow.
        uint emissiveLightStart;
        uint emissiveLightCount;
        // Texture probes for material guesses follow the emissive lights in SceneLights, sampled by the same dispatch.
        uint materialProbeCount;
        // Ray traced lighting debug view (F10), 0 for the normal image.
        uint debugView;
        // rt_lights.json "room settings": the sun or moon is off in this room (lit only by manual and emissive lights).
        uint sunDisabled;
        // Set for one frame when the lighting changes at once (the sun switched off or on): no temporal history.
        uint historyReset;
        // Diffuse bounces of the indirect light: 1, or 2 with the Graphics menu's "Second light bounce" on.
        uint indirectBounces;
        // Water draws ray traced this frame (Graphics menu "Advanced Water Rendering"); 0 skips the water pass.
        uint waterDraws;
        // Display pixels per ray tracing pixel: above 1 when the ray traced image is rendered smaller and upscaled.
        // The rasterized frame stays at the display resolution.
        float displayScaleX;
        float displayScaleY;
        // Set while the ray traced image is upscaled: PostProcessPS writes the primary hit coverage as alpha, so pixels
        // that missed the scene (sky, background) keep the rasterized frame at the display resolution.
        uint upscaleCoverage;
        // Seconds since the renderer started (wrapped every hour), for the water waves.
        float waterTime;
        // Render index + 1 of up to six path traced decals whose camera hits the F9 capture counts (0 = unused slot).
        uint diagDecal0;
        uint diagDecal1;
        uint diagDecal2;
        uint diagDecal3;
        uint diagDecal4;
        uint diagDecal5;
        // Path traced decals this frame (0 skips the camera ray's decal query), and set on the frames whose decal hits
        // are counted for the decal check (see RaytracingResources::decalRasterFallback).
        uint decalsPresent;
        uint decalCheck;
        // Where the draws' emission words start in gRtMaterials (after one material word per draw), or 0 when no draw
        // of the frame is emissive.
        uint emissionOffset;
        // Where the draws' foliage centers start in gRtMaterials (three words per draw: X, Z, radius), or 0 when no draw
        // of the frame is foliage.
        uint foliageOffset;

#   ifdef HLSL_CPU
        RaytracingParams() {
            view = FLOAT4X4_IDENTITY;
            viewI = FLOAT4X4_IDENTITY;
            projection = FLOAT4X4_IDENTITY;
            projectionI = FLOAT4X4_IDENTITY;
            viewProj = FLOAT4X4_IDENTITY;
            prevViewProj = FLOAT4X4_IDENTITY;
            cameraU = { 0.0f, 0.0f, 0.0f, 0.0f };
            cameraV = { 0.0f, 0.0f, 0.0f, 0.0f };
            cameraW = { 0.0f, 0.0f, 0.0f, 0.0f };
            viewport = { 0.0f, 0.0f, 0.0f, 0.0f };
            resolution = { 0.0f, 0.0f, 0.0f, 0.0f };
            ambientBaseColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            ambientNoGIColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            eyeLightDiffuseColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            eyeLightSpecularColor = { 0.0f, 0.0f, 0.0f, 0.0f };
            pixelJitter = { 0.0f, 0.0f };
            fovRadians = 0.707f;
            nearDist = 1.0f;
            farDist = 1000.0f;
            giDiffuseStrength = 0.0f;
            giBackgroundStrength = 0.0f;
            motionBlurStrength = 0.0f;
            tonemapExposure = 0.6f;
            tonemapWhite = 1.0f;
            tonemapBlack = 0.0f;
            diSamples = 0;
            giSamples = 0;
            diReproject = 0;
            giReproject = 0;
            binaryLockMask = 0;
            maxLights = 1;
            motionBlurSamples = 32;
            visualizationMode = VisualizationMode::Final;
            pixelSpreadAngle = 0.001f;
            cullWindingSign = 1.0f;
            emissiveLightStart = 0;
            emissiveLightCount = 0;
            materialProbeCount = 0;
            debugView = 0;
            sunDisabled = 0;
            historyReset = 0;
            indirectBounces = 1;
            waterDraws = 0;
            displayScaleX = 1.0f;
            displayScaleY = 1.0f;
            upscaleCoverage = 0;
            waterTime = 0.0f;
            diagDecal0 = 0;
            diagDecal1 = 0;
            diagDecal2 = 0;
            diagDecal3 = 0;
            diagDecal4 = 0;
            diagDecal5 = 0;
            decalsPresent = 0;
            decalCheck = 0;
            emissionOffset = 0;
            foliageOffset = 0;
        }
#   endif
    };
#ifdef HLSL_CPU
};
#endif