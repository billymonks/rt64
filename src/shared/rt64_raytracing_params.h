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
        // Graphics menu "Light Samples": sun shadow, light, bounce and occlusion rays per pixel and frame (1, 2 or 4).
        uint lightSamples;
        // Graphics menu "Light Accumulation" (RtTemporalCS): the smallest blend weight of a new frame, the longest history
        // in frames, and how far history may stray from the current frame's neighborhood, in standard deviations.
        float temporalMinBlend;
        float temporalMaxHistory;
        float temporalClipGamma;
        // The pixel the F9 capture describes, as a fraction of the image (the mouse cursor when F9 was pressed), or
        // negative for the image center.
        float diagPixelX;
        float diagPixelY;
        // Volumetric fog of the room (rt_lights.json room settings "fog", Graphics menu "Volumetric Fog"): extinction
        // per world unit at or below fogBaseHeight (0 turns the fog off), thinning by exp(-fogHeightFalloff * height
        // above it). The tint of the light it scatters, a uniform glow (ambient), how strongly the sun or moon and the
        // point lights glow in it, the Henyey-Greenstein anisotropy, and how far from the camera it reaches.
        float fogDensity;
        float fogHeightFalloff;
        float fogBaseHeight;
        float fogColorR;
        float fogColorG;
        float fogColorB;
        float fogAmbient;
        float fogSunScatter;
        float fogLightScatter;
        float fogAnisotropy;
        float fogDistance;
        // Multiplier of the game's own fog where the volumetric fog is on.
        float fogGameScale;
        // March steps per camera ray, and 0 when last frame's fog can't be reused (it was off or the room changed).
        uint fogSteps;
        uint fogHistory;
        // How much the air the sun or moon doesn't reach darkens what is behind it (the shafts between shadows).
        float fogShadows;
        // God rays (0 strength when off): strength, reach (share of the viewport's larger side), color, window
        // brightness, the direction towards their light where the sun or moon doesn't give it, and whether it does.
        float raysStrength;
        float raysLength;
        float raysColorR;
        float raysColorG;
        float raysColorB;
        float raysWindows;
        float raysDirX;
        float raysDirY;
        float raysDirZ;
        uint raysFromSun;
        // How far the windows' rays turn from the light's direction towards the way they face out, and those ways:
        // the frame's windows grouped by facing (up to 4, none when unknown or not turned).
        float raysWindowBias;
        uint raysWindowGroups;
        float raysWindowGroup0X;
        float raysWindowGroup0Y;
        float raysWindowGroup0Z;
        float raysWindowGroup1X;
        float raysWindowGroup1Y;
        float raysWindowGroup1Z;
        float raysWindowGroup2X;
        float raysWindowGroup2Y;
        float raysWindowGroup2Z;
        float raysWindowGroup3X;
        float raysWindowGroup3Y;
        float raysWindowGroup3Z;

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
            lightSamples = 1;
            temporalMinBlend = 0.1f;
            temporalMaxHistory = 16.0f;
            temporalClipGamma = 1.5f;
            diagPixelX = -1.0f;
            diagPixelY = -1.0f;
            fogDensity = 0.0f;
            fogHeightFalloff = 0.0f;
            fogBaseHeight = 0.0f;
            fogColorR = 1.0f;
            fogColorG = 1.0f;
            fogColorB = 1.0f;
            fogAmbient = 0.0f;
            fogSunScatter = 0.0f;
            fogLightScatter = 0.0f;
            fogAnisotropy = 0.0f;
            fogDistance = 0.0f;
            fogGameScale = 1.0f;
            fogSteps = 0;
            fogHistory = 0;
            fogShadows = 0.0f;
            raysStrength = 0.0f;
            raysLength = 0.5f;
            raysColorR = 1.0f;
            raysColorG = 1.0f;
            raysColorB = 1.0f;
            raysWindows = 0.0f;
            raysDirX = 0.0f;
            raysDirY = 1.0f;
            raysDirZ = 0.0f;
            raysFromSun = 0;
            raysWindowBias = 0.0f;
            raysWindowGroups = 0;
            raysWindowGroup0X = raysWindowGroup0Y = raysWindowGroup0Z = 0.0f;
            raysWindowGroup1X = raysWindowGroup1Y = raysWindowGroup1Z = 0.0f;
            raysWindowGroup2X = raysWindowGroup2Y = raysWindowGroup2Z = 0.0f;
            raysWindowGroup3X = raysWindowGroup3Y = raysWindowGroup3Z = 0.0f;
        }
#   endif
    };
#ifdef HLSL_CPU
};
#endif