//
// RT64
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "common/rt64_user_configuration.h"
#include "preset/rt64_preset_scene.h"
#include "render/rt64_buffer_uploader.h"
#include "render/rt64_raytracing_materials.h"
#include "render/rt64_upscaler.h"
#include "shared/rt64_interleaved_raster.h"
#include "shared/rt64_point_light.h"
#include "shared/rt64_raytracing_params.h"

namespace RT64 {
    struct InstanceDrawCall;
    struct RenderTarget;
    struct RenderFramebufferStorage;
    struct ShaderLibrary;

    // Upscaler of the ray traced image chosen in the Graphics menu.
    enum class RaytracingUpscaler : uint32_t {
        Off = 0,
        FSR1 = 1,
        FSR3 = 2
    };

    struct RaytracingConfiguration {
        int diSamples = 1;
        int giSamples = 1;
        int maxLights = 8;
        int maxReflections = 0;
        float motionBlurStrength = 0.0f;
        int motionBlurSamples = 1;
        interop::VisualizationMode visualizationMode = interop::VisualizationMode::Final;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        Upscaler::QualityMode upscalerQualityMode = Upscaler::QualityMode::Native;
        float upscalerSharpness = 0.0f;
        bool upscalerResolutionOverride = false;
        float resolutionScale = 1.0f;
        bool upscalerReactiveMask = false;
        bool upscalerLockMask = false;
        bool denoiserEnabled = true;
    };

    // Instance masks. Camera and bounce rays see the drawn geometry; shadow rays see the shadow casters, where
    // camera facing billboards are replaced by light facing proxies.
    static const uint32_t RtCameraRayMask = 0x1;
    static const uint32_t RtShadowRayMask = 0x2;
    // Water surfaces (Advanced Water Rendering): only the water pass's camera rays see them.
    static const uint32_t RtWaterRayMask = 0x4;
    // Decals (decal depth mode): only the camera ray's decal query sees them, never camera, bounce or shadow rays.
    static const uint32_t RtDecalRayMask = 0x8;
    // Flags in the high bits of a draw's gRtMaterials word (above the material class): the water pass shades this path
    // traced draw as water, and (for water found by its scrolling texture) only where its color is blue or blue-green.
    static const uint32_t RtMaterialWaterPassFlag = 1u << 28;
    static const uint32_t RtMaterialWaterHueGateFlag = 1u << 29;
    // A decal draw (decal depth mode), path traced on top of the surface it lies on; and one that is alpha blended, so its
    // camera hits blend its color over that surface's by its alpha.
    static const uint32_t RtMaterialDecalFlag = 1u << 30;
    static const uint32_t RtMaterialDecalBlendFlag = 1u << 31;
    // Large path traced draws (scenery rather than characters or small objects), which take decals drawn before them.
    // It replaces the top bit of the material class, which the shaders don't read.
    static const uint32_t RtMaterialSceneryFlag = 1u << 27;
    // Foliage (upright cutout trees and bushes, see looksLikeFoliage): a soft normal around the tree's axis, wrapped
    // light and no self-shadowing. The rest of the material class bits are cleared; the shaders don't read the class.
    static const uint32_t RtMaterialFoliageFlag = 1u << 26;
    static const uint32_t RtMaterialClassBits = 0x07000000u;
    // A camera facing billboard with a light facing shadow proxy: proxies close to it don't shadow it (its own, and
    // the other draws of a tree split into several).
    static const uint32_t RtMaterialBillboardFlag = 1u << 25;
    // A window (rt_materials.json class "window", stained glass): god rays stream from its pixels in its color.
    static const uint32_t RtMaterialWindowFlag = 1u << 24;
    // Largest learned offset of a decal from its surface (kDecalMaxOffset in RtLibrary.hlsl).
    static const float RtDecalMaxOffset = 16.0f;

    struct RaytracingInstance {
        // Index into the frame's RenderIndices, used as the instance ID so hit shaders can find the draw's material.
        uint32_t renderIndex = 0;
        uint32_t bottomLevelIndex = 0;
        bool opaque = true;
        bool alphaTested = false;
        bool culling = false;
        bool proxy = false;
        uint32_t queryMask = 0xFF;
        bool cullDisable = true;
    };

    // How a candidate light source is drawn. The GPU applies a different color test to each (see emissiveGlow in
    // RtLibrary.hlsl): additive sprites glow when bright, blended ones when bright and saturated, and small cutout
    // sprites only when bright, saturated and warm like a flame.
    enum class EmissiveCategory : uint32_t {
        Additive = 0,
        Blended = 1,
        Cutout = 2
    };

    // A piece of an effect sprite drawn in the 3D view (flames, spell effects) that may emit light.
    struct EmissiveCandidate {
        float center[3] = {};
        float extent = 0.0f;
        uint32_t renderIndex = 0;
        uint32_t triangleCount = 0;
        uint64_t key = 0;
        EmissiveCategory category = EmissiveCategory::Blended;
        // A long, narrow strip. Thin pieces never merge into clusters, and rings of them are rejected.
        bool thin = false;
        // An untextured effect (hit sparks), keyed by its combiner and primitive color.
        bool untextured = false;
        // The draw the piece comes from, matched against rt_emitters.json.
        RtEmitterSignature signature;
    };

    struct EmissiveRejections {
        uint32_t groundFlat = 0;
        uint32_t tooLarge = 0;
        uint32_t tooSmall = 0;
        uint32_t clusterTooLarge = 0;
        uint32_t thinLinePieces = 0;
        uint32_t untexturedDraws = 0;
        // Cutout draws lit with vertex normals: model parts, never flames.
        uint32_t litCutouts = 0;
    };

    // Details of a candidate draw for the diagnostics log.
    struct EmissiveDrawInfo {
        uint64_t key = 0;
        EmissiveCategory category = EmissiveCategory::Blended;
        uint32_t triangleCount = 0;
        uint32_t pieces = 0;
        uint32_t thinPieces = 0;
        uint32_t accepted = 0;
        float maxAspect = 0.0f;
        float maxExtent = 0.0f;
    };

    struct RaytracingScene {
        // Draw calls of the scene. They are rasterized as usual; the ray tracer only adds lighting on top.
        std::vector<uint32_t> instanceIndices;
        std::vector<RaytracingInstance> instances;
        uint32_t skippedDraws = 0;
        uint32_t rasterAfterDraws = 0;
        uint32_t rasterOnlyOpaqueDraws = 0;

        // Material categories of the path traced draws, for the diagnostics log.
        uint32_t drawsTwoCycle = 0;
        uint32_t drawsTextureLOD = 0;
        uint32_t drawsTexture1 = 0;
        uint32_t drawsUntextured = 0;
        uint32_t drawsFlatShade = 0;
        uint32_t drawsTexturePerspectiveOff = 0;
        uint32_t drawsCulling = 0;
        uint32_t incompatibleProjections = 0;
        uint32_t skippedLayeredDraws = 0;
        // Decal depth mode draws: path traced, kept rasterized after because their world positions don't match the
        // raster's or they're additive, and all decals rasterized after (also the ones through another camera).
        uint32_t decalsTraced = 0;
        uint32_t decalsWorldMismatch = 0;
        uint32_t decalsAdditive = 0;
        uint32_t decalsRasterAfter = 0;
        // Decals rasterized after because an earlier capture found the tie-break couldn't show them.
        uint32_t decalsFallback = 0;
        // The path traced decals of the frame, for the F9 log (capped).
        struct DecalInfo {
            uint64_t key;
            uint64_t id;
            uint32_t renderIndex;
            uint32_t triangles;
            float extent;
            bool blended;
            bool alphaTested;
            bool culling;
            uint32_t queryMask;
            uint32_t combiner[2];
            float primAlpha;
            float envAlpha;
        };
        std::vector<DecalInfo> decalInfos;
        // Path traced draws with an emissive material (lava), for the F9 log, and their triangles gathered into cells
        // of a world grid (sums weighted by area), from which the lava lights are placed.
        struct EmissiveDraw {
            uint64_t key;
            uint32_t renderIndex;
            uint32_t triangles;
            float radiance[3];
        };
        std::vector<EmissiveDraw> emissiveDraws;
        struct EmissiveCell {
            float position[3] = {};
            float normal[3] = {};
            float radiance[3] = {};
            float area = 0.0f;
        };
        std::unordered_map<int64_t, EmissiveCell> emissiveCells;
        // Path traced draws shaded as foliage.
        uint32_t foliageDraws = 0;
        uint32_t billboardsCylindrical = 0;
        uint32_t billboardsSpherical = 0;
        uint32_t billboardProxies = 0;

        std::vector<EmissiveCandidate> emissiveCandidates;
        uint32_t emissiveCandidateDraws = 0;
        EmissiveRejections emissiveRejections;
        std::vector<EmissiveDrawInfo> emissiveDrawInfos;
        uint32_t emissiveUntexturedAdditiveDraws = 0;
        uint32_t emissiveDepthWritingAdditiveDraws = 0;
        // Opaque 3D draws outside the path traced scene that only cast shadows.
        uint32_t shadowOnlyCasters = 0;
        // Water surfaces shaded by the water pass (Advanced Water Rendering): blended ones it ray traces instead of
        // rasterizing after the path traced pass, and path traced (opaque) ones it shades on top.
        struct WaterDraw {
            uint64_t key;
            uint32_t renderIndex;
            uint32_t triangles;
            float extent;
            float upFacing;
            // "listed" (rt_materials.json class water) or "detected" (flat, wide, alpha blended, no depth write).
            const char *source;
        };
        std::vector<WaterDraw> waterDraws;
        // Draws that looked like water but were left alone: a red tint (lava) or a material entry of another class.
        uint32_t waterRejected = 0;
        // Why some large flat draws were not taken as water, for the F9 log (capped).
        struct WaterReject {
            uint64_t key;
            uint32_t renderIndex;
            float extent;
            const char *reason;
        };
        std::vector<WaterReject> waterRejects;
        // Textures to sample this frame for material guesses, and the texture key of each path traced draw by render
        // index (0 when untextured or not path traced).
        std::vector<RtMaterialProbe> materialProbes;
        std::vector<uint64_t> materialKeys;
        // 3D draws kept out of the path traced scene because their world positions don't project to where the raster
        // drew them (a failed world matrix split), with a few examples for the log.
        uint32_t drawsWorldMismatch = 0;
        struct WorldMismatch {
            uint32_t renderIndex;
            uint64_t texture;
            uint32_t vertexAddress;
            uint32_t triangles;
            float error;
        };
        std::vector<WorldMismatch> worldMismatchSamples;
        // Draw signature of each textured path traced draw by render index, for the F9 "center draw" line.
        std::vector<RtEmitterSignature> drawSignatures;
        struct MaterialPreview {
            uint64_t key;
            RenderTexture *texture;
            uint32_t width;
            uint32_t height;
        };
        std::vector<MaterialPreview> materialPreviews;
        // Where the game is (see SceneLight), for the emitter area filters and "first seen" notes.
        int areaMap = -1;
        int areaRoom = -1;
        const char *areaName = "";
        interop::float4x4 curViewMatrix;
        interop::float4x4 curProjMatrix;
        interop::float4x4 prevViewMatrix;
        interop::float4x4 prevProjMatrix;
        RenderViewport viewport;
        RenderRect scissor;
        float viewportScaleY = 1.0f;
        PresetScene presetScene;
        int screenWidth = 0;
        int screenHeight = 0;
        const interop::PointLight *pointLights = nullptr;
        uint32_t lightCount = 0;
        std::vector<interop::InterleavedRaster> interleavedRasters;
        float deltaTime = 0.0f;

        // One vertex of the first ray traced draw, used by the diagnostics log and probe ray.
        bool probeValid = false;
        uint32_t probeVertex = 0;
        float probeWorld[3] = {};
        float probeClipRSP[4] = {};

        // The game's own N64 fog on the path traced draw with the most triangles that uses it: the fog color and the
        // RSP fog multiplier and offset (vertex fog alpha = max(z, 0) / w * mul + offset, out of 255).
        // Path traced draws of window textures (rt_materials.json class "window"), the indoor sources of god rays.
        uint32_t windowDraws = 0;
        // 3D draws are rasterized after this scene, and the volumetric fog goes over the image after them.
        bool volumetricLater = false;
        // Their facing (see measureWindow), which bends their god rays to shine in through them.
        struct WindowInfo {
            float center[3] = {};
            float normal[3] = {};
            float area = 0.0f;
        };

        std::vector<WindowInfo> windows;
        bool gameFogValid = false;
        uint32_t gameFogTriangles = 0;
        float gameFogColor[3] = {};
        float gameFogMul = 0.0f;
        float gameFogOffset = 0.0f;
    };

    struct RaytracingResources {
        static const uint32_t MaxHitQueries = 24;

        RenderDevice *device = nullptr;
        interop::RaytracingParams rtParams;
        BufferPair rtParamsBuffer;

        bool swapBuffers = false;
        interop::float4x4 previousViewProj;
        bool previousViewProjValid = false;
        bool skipReprojection = false;
        bool denoiserEnabled = true;
        bool upscaleActive = false;
        bool upscalerReactiveMask = false;
        bool upscalerLockMask = false;
        bool transitionOutputBuffers = true;
        bool updateOutputBuffers = true;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        int maxReflections = 0;
        // Ray tracing resolution. It's below the display resolution while an upscaler is active.
        uint32_t textureWidth = 1;
        uint32_t textureHeight = 1;
        uint32_t displayWidth = 1;
        uint32_t displayHeight = 1;

        // Upscaling of the ray traced image (Graphics menu "Upscaler"), see rt64_raytracing_upscale.cpp.
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        const ShaderLibrary *shaderLibrary = nullptr;
        std::unique_ptr<Upscaler> fsr1Upscaler;
        std::unique_ptr<Upscaler> fsr3Upscaler;
        Upscaler *activeUpscaler = nullptr;
        RaytracingUpscaler requestedUpscaler = RaytracingUpscaler::Off;
        RaytracingUpscaler activeUpscalerKind = RaytracingUpscaler::Off;
        RaytracingUpscaler previousUpscalerKind = RaytracingUpscaler::Off;
        Upscaler::QualityMode upscalerQuality = Upscaler::QualityMode::Quality;
        float upscalerSharpness = 0.5f;
        bool fsr3Unavailable = false;
        bool fsr3FallbackLogged = false;
        std::string fsr3FailureReason;
        bool upscalerResetPending = true;
        bool upscalerResetThisFrame = false;
        uint32_t upscalerResets = 0;
        uint64_t lastRaytracedFrame = 0;
        bool lastRaytracedFrameValid = false;
        float lastCameraOrigin[3] = {};
        float lastCameraForward[3] = {};
        bool lastCameraValid = false;
        // Near and far planes of the game's projection (rtParams.nearDist and farDist are repurposed).
        float cameraNearPlane = 1.0f;
        float cameraFarPlane = 1000.0f;
        int jitterPhaseCount = 1;
        int jitterIndex = 0;
        std::unique_ptr<RenderTexture> motionVectorTexture;
        std::unique_ptr<RenderTexture> deviceDepthTexture;
        std::unique_ptr<struct UpscalerInputsDescriptorSet> upscalerInputsSet;
        std::unique_ptr<RenderPipelineLayout> upscalerInputsPipelineLayout;
        std::unique_ptr<RenderPipeline> upscalerInputsPipeline;
        bool upscalerInputsPipelineCreated = false;

        // GPU time of each ray tracing pass, averaged. Timestamps go through a ring of query pools so each is read a
        // few frames after it was written.
        enum class PassTimestamp : uint32_t {
            Start,
            Primary,
            Emissive,
            Direct,
            Indirect,
            Refraction,
            Reflection,
            Temporal,
            Denoise,
            Compose,
            Upscale,
            Post,
            Count
        };

        static const uint32_t PassTimingDepth = 3;
        static const uint32_t PassCount = uint32_t(PassTimestamp::Count) - 1;
        std::unique_ptr<RenderQueryPool> passQueryPools[PassTimingDepth];
        bool passQueryWritten[PassTimingDepth] = {};
        uint32_t passQueryBounces[PassTimingDepth] = {};
        uint32_t passQueryFrame = 0;
        RenderQueryPool *activePassQueryPool = nullptr;
        uint32_t activePassQuerySlot = 0;
        float passMs[PassCount] = {};
        float totalPassMs = 0.0f;

        // GPU time of the bottom and top level acceleration structure builds, through their own ring of query pools.
        std::unique_ptr<RenderQueryPool> buildQueryPools[PassTimingDepth];
        bool buildQueryWritten[PassTimingDepth] = {};
        uint32_t buildQueryFrame = 0;
        RenderQueryPool *activeBuildQueryPool = nullptr;
        uint32_t activeBuildQuerySlot = 0;
        float accelerationBuildMs = 0.0f;

        // Temporal accumulation and spatial filter of the ray traced lighting (rt64_raytracing_denoise.cpp). The light
        // passes write this frame's samples to filteredDirectLightTexture[1] and filteredIndirectLightTexture[1];
        // RtTemporalCS blends them into directLightTexture and indirectLightTexture, and RtDenoiseCS filters those into
        // filteredDirectLightTexture[1] and filteredIndirectLightTexture[1] for ComposePS.
        static const uint32_t MaxDenoisePasses = 5;
        uint32_t lightSamplesLevel = 0;
        uint32_t denoiserLevel = 0;
        uint32_t accumulationLevel = 0;
        uint32_t denoisePassCount = 0;
        std::unique_ptr<RenderTexture> momentsTexture[2];
        std::unique_ptr<RenderTexture> lightVarianceTexture[2];
        std::unique_ptr<RenderTexture> lightGuideTexture;
        std::unique_ptr<struct RtTemporalDescriptorSet> lightTemporalSet;
        std::unique_ptr<struct RtDenoiseDescriptorSet> lightDenoiseSets[MaxDenoisePasses];
        std::unique_ptr<RenderPipelineLayout> lightTemporalPipelineLayout;
        std::unique_ptr<RenderPipeline> lightTemporalPipeline;
        std::unique_ptr<RenderPipelineLayout> lightDenoisePipelineLayout;
        std::unique_ptr<RenderPipeline> lightDenoisePipeline;
        bool lightFilterPipelinesCreated = false;

        std::unique_ptr<RenderTexture> viewDirectionTexture;
        std::unique_ptr<RenderTexture> shadingPositionTexture;
        std::unique_ptr<RenderTexture> shadingNormalTexture;
        std::unique_ptr<RenderTexture> shadingSpecularTexture;
        std::unique_ptr<RenderTexture> diffuseTexture;
        std::unique_ptr<RenderTexture> instanceIdTexture;
        std::unique_ptr<RenderTexture> directLightTexture[2];
        std::unique_ptr<RenderTexture> indirectLightTexture[2];
        std::unique_ptr<RenderTexture> reflectionTexture;
        std::unique_ptr<RenderTexture> refractionTexture;
        std::unique_ptr<RenderTexture> transparentTexture;
        std::unique_ptr<RenderTexture> flowTexture;
        std::unique_ptr<RenderTexture> reactiveMaskTexture;
        std::unique_ptr<RenderTexture> lockMaskTexture;
        std::unique_ptr<RenderTexture> normalRoughnessTexture[2];
        std::unique_ptr<RenderTexture> depthTexture[2];
        std::unique_ptr<RenderTexture> filteredDirectLightTexture[2];
        std::unique_ptr<RenderTexture> filteredIndirectLightTexture[2];
        // Volumetric fog (gVolumetric), this frame's and last frame's by swapBuffers like the light accumulation.
        std::unique_ptr<RenderTexture> volumetricTexture[2];
        std::unique_ptr<RenderTexture> outputTexture[2];
        std::unique_ptr<RenderFramebuffer> outputFramebuffer[2];
        std::unique_ptr<RenderTexture> upscaledOutputTexture;
        std::unique_ptr<RenderTexture> downscaledOutputTexture;
        std::unique_ptr<RenderTexture> lumaAverageTexture;

        std::unique_ptr<RenderBuffer> hitVelocityDistanceBuffer;
        std::unique_ptr<RenderBuffer> hitColorBuffer;
        std::unique_ptr<RenderBuffer> hitNormalFogBuffer;
        std::unique_ptr<RenderBuffer> hitInstanceIdBuffer;
        std::unique_ptr<RenderBufferFormattedView> hitVelocityDistanceBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitColorBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitNormalFogBufferView;
        std::unique_ptr<RenderBufferFormattedView> hitInstanceIdBufferView;

        std::unique_ptr<RenderBuffer> lightsBuffer;
        std::unique_ptr<RenderBuffer> lightsUpload;
        uint64_t lightsUploadCapacity = 0;
        std::unique_ptr<RenderBuffer> shaderBindingTableBuffer;
        std::unique_ptr<RenderBuffer> shaderBindingTableUpload;
        uint64_t shaderBindingTableUploadCapacity = 0;
        std::unique_ptr<RenderBuffer> instancesUpload;
        uint64_t instancesUploadCapacity = 0;
        std::unique_ptr<RenderBuffer> scratchBuffer;
        uint64_t scratchCapacity = 0;
        std::unique_ptr<RenderBuffer> instancesBuffer;
        std::unique_ptr<RenderBuffer> dummyVertexBuffer;
        std::unique_ptr<RenderBuffer> dummyIndexBuffer;
        std::unique_ptr<RenderBuffer> dummyVertexUpload;
        std::unique_ptr<RenderBuffer> dummyIndexUpload;
        uint64_t dummyVertexUploadCapacity = 0;
        uint64_t dummyIndexUploadCapacity = 0;
        uint64_t lightsCapacity = 0;
        uint64_t instancesCapacity = 0;
        uint64_t shaderBindingTableCapacity = 0;
        RenderShaderBindingTableInfo shaderBindingTableInfo;

        // Every bottom level structure of the frame lives in bottomLevelPool at its offset, and builds in its own range
        // of the scratch buffer so the builds can overlap. A build that starts a new scratch batch reuses the scratch
        // from offset 0 and must wait for the batch before it.
        struct BottomLevel {
            RenderBottomLevelASMesh mesh;
            RenderBottomLevelASBuildInfo buildInfo;
            uint64_t offset = 0;
            uint64_t scratchOffset = 0;
            bool newScratchBatch = false;
            std::unique_ptr<RenderAccelerationStructure> accelerationStructure;
        };
        std::vector<BottomLevel> bottomLevelASVector;
        std::unique_ptr<RenderBuffer> bottomLevelPool;
        uint64_t bottomLevelPoolCapacity = 0;
        uint64_t bottomLevelPoolUsed = 0;
        uint32_t bottomLevelScratchBatches = 0;
        std::vector<RenderBottomLevelASMesh> pendingMeshes;
        // Index into bottomLevelASVector for each pending mesh, or -1 if it was not built.
        std::vector<int32_t> pendingMeshLevels;

        // Light facing shadow proxies of billboard draws, built on the CPU each frame. Their pending meshes get the
        // buffer references once the data is uploaded in updateBottomLevelASResources.
        struct ProxyMeshRef {
            uint32_t meshIndex = 0;
            uint64_t vertexOffset = 0;
            uint64_t indexOffset = 0;
        };
        std::vector<float> proxyVertexData;
        std::vector<uint32_t> proxyIndexData;
        std::vector<ProxyMeshRef> proxyMeshRefs;
        std::unique_ptr<RenderBuffer> proxyVertexBuffer;
        std::unique_ptr<RenderBuffer> proxyIndexBuffer;
        std::unique_ptr<RenderBuffer> proxyVertexUpload;
        std::unique_ptr<RenderBuffer> proxyIndexUpload;
        uint64_t proxyVertexCapacity = 0;
        uint64_t proxyIndexCapacity = 0;
        uint64_t proxyVertexUploadCapacity = 0;
        uint64_t proxyIndexUploadCapacity = 0;
        std::unique_ptr<RenderBuffer> topLevelBuffer;
        std::unique_ptr<RenderAccelerationStructure> topLevelAS;
        RenderTopLevelASBuildInfo topLevelBuildInfo;
        std::vector<RenderBuffer *> geometryBuffers;
        std::vector<RenderBuffer *> shaderReadBuffers;

        std::unique_ptr<struct GaussianFilterDescriptorSet> indirectFilterSets[2];
        std::unique_ptr<struct RaytracingComposeDescriptorSet> composeSet;
        std::unique_ptr<struct PostProcessDescriptorSet> postProcessSet;
        std::unique_ptr<struct HistogramSetDescriptorSet> lumaSetSet;
        std::unique_ptr<struct LuminanceHistogramDescriptorSet> lumaSet;
        std::unique_ptr<struct HistogramAverageDescriptorSet> lumaAvgSet;
        std::unique_ptr<struct HistogramClearDescriptorSet> lumaClearSet;
        std::unique_ptr<struct BicubicScalingDescriptorSet> downscaleSet;
        std::unique_ptr<RenderBuffer> histogramBuffer;

        std::vector<std::unique_ptr<RenderTarget>> interleavedColorTargetVector;
        std::vector<std::unique_ptr<RenderTarget>> interleavedDepthTargetVector;
        std::vector<std::unique_ptr<RenderFramebufferStorage>> interleavedFramebufferStorageVector;

        // Emissive point lights. Clusters of candidate sprites are tracked across frames so each light keeps its slot
        // (which indexes its smoothed color on the GPU) and its position can be smoothed.
        static const uint32_t MaxEmissiveLights = 32;
        // Manual point lights (rt_lights.json) used per area, on top of the emissive lights.
        // Manual lights (rt_lights.json) shown at once: the area's most important lights for the camera this frame (see
        // updateEmissiveLights). An area may have up to MaxAreaManualLights.
        static const uint32_t MaxManualLights = 24;
        static const uint32_t MaxAreaManualLights = 1024;
        // How long a light takes to fade in or out as it joins or leaves the shown set, how much a shown light's rank is
        // raised so close calls don't trade places, and how far beyond its reach a light may be and still count.
        static constexpr float ManualLightFadeSeconds = 0.4f;
        static constexpr float ManualLightHysteresis = 1.25f;
        static constexpr float ManualLightViewReach = 600.0f;
        struct ManualLightState {
            float fade = 0.0f;
            float distance = 0.0f;
            float score = 0.0f;
            bool selected = false;
            bool seen = false;
        };
        std::unordered_map<std::string, ManualLightState> manualLightFades;
        float manualLightsLastSeconds = 0.0f;
        uint32_t frameAreaManualLights = 0;
        struct ManualLightInfo {
            std::string id;
            float distance;
            float fade;
            float score;
            bool selected;
        };
        std::vector<ManualLightInfo> frameManualLights;
        // Lights cast by emissive materials (lava): one per cell of emissive triangles nearest the camera, fading in and
        // out as cells enter and leave that set. Separate from the effect and manual lights.
        static const uint32_t MaxLavaLights = 12;
        struct LavaLight {
            int64_t cell = 0;
            float position[3] = {};
            float radiance[3] = {};
            float range = 0.0f;
            float radius = 0.0f;
            float fade = 0.0f;
            bool selected = false;
        };
        std::vector<LavaLight> lavaLights;
        uint32_t frameLavaLightCount = 0;
        std::vector<std::string> frameManualLightIds;
        // The room's sun and moon setting (rt_lights.json "room settings") last frame and where it came from.
        int32_t lastSunDisabled = -1;
        uint32_t lastIndirectBounces = 0;
        // The room's volumetric fog (rt_lights.json room settings "fog") last frame: whether it was on, in which area,
        // and where it came from.
        bool lastFogActive = false;
        int32_t lastFogMap = -1;
        int32_t lastFogRoom = -1;
        float lastFogHour = -1.0f;
        const char *fogSettingSource = "default";
        const char *fogSettingScope = "no entry";
        // Where the game's own fog starts and ends this frame, in world units from the camera (-1 when it has none).
        // Path traced window draws last frame (god ray sources indoors).
        uint32_t lastWindowDraws = 0;
        // Where the path traced scene's image went, for the volumetric fog pass over it (FramebufferRenderer::submitVolumetric).
        RenderRect volumetricScissor;
        float gameFogStart = -1.0f;
        float gameFogEnd = -1.0f;
        // Decals (by texture, vertex load and size) that three decal checks in a row saw mostly missed: rasterized after
        // the path traced pass, by the frame they fell back on, and tried again after a while. Each decal's learned
        // offset from its surface (normal distance, up to 16 units) and its count of bad checks in a row.
        std::unordered_map<uint64_t, uint32_t> decalRasterFallback;
        std::unordered_map<uint64_t, float> decalOffsets;
        std::unordered_map<uint64_t, uint8_t> decalBadChecks;
        static const uint32_t DecalFallbackRetryFrames = 1800;
        // The decals whose camera hits the last capture counted (RtParams.diagDecal0-5).
        static const uint32_t DiagDecalSlots = 6;
        RaytracingScene::DecalInfo diagDecals[DiagDecalSlots] = {};
        uint32_t diagDecalCount = 0;
        // Decal check: every few seconds the camera hits of a few decals are counted and read back (counters only).
        std::unique_ptr<RenderBuffer> decalCheckReadback;
        int32_t decalCheckDelay = -1;
        uint32_t decalCheckCursor = 0;
        bool decalCheckThisFrame = false;
        // GPU time of the indirect light pass, averaged, with 1 and with 2 bounces. Timestamps go through a ring of
        // query pools so each is read a few frames after it was written.
        static const uint32_t IndirectTimingDepth = 3;
        std::unique_ptr<RenderQueryPool> indirectQueryPools[IndirectTimingDepth];
        bool indirectQueryWritten[IndirectTimingDepth] = {};
        uint32_t indirectQueryBounces[IndirectTimingDepth] = {};
        uint32_t indirectQueryFrame = 0;
        float indirectPassMs[2] = {};
        const char *sunSettingSource = "default";
        const char *sunSettingScope = "no entry";
        bool manualLightCapLogged = false;
        static const uint32_t EmissiveSlots = 64;
        // Lights fade in and out instead of popping: fade rises while the source is seen and falls once it's gone, so a
        // light outlives its source by a few frames.
        struct TrackedEmissive {
            float position[3] = {};
            float extent = 0.0f;
            float fade = 0.0f;
            uint64_t key = 0;
            uint32_t slot = 0;
            uint32_t renderIndex = 0;
            uint32_t triangleCount = 0;
            EmissiveCategory category = EmissiveCategory::Blended;
            bool fresh = true;
            bool seen = false;
            bool suppressed = false;
            bool keyChanged = false;
            // From an untextured effect (hit sparks): a brief, modest flash.
            bool untextured = false;
            // The rt_emitters.json entry the light comes from.
            RtEmitterParams params;
        };
        std::vector<TrackedEmissive> trackedEmissives;
        std::vector<interop::PointLight> frameLights;

        // What the GPU found each source's color to be, by key, from earlier frames. Sources it keeps rejecting
        // (dark, dust-like, not flame colored) stop taking light slots, and get sampled again now and then.
        struct EmissiveVerdict {
            float glow = 0.0f;
            uint32_t samples = 0;
            uint32_t lastSampledFrame = 0;
            uint32_t lastSeenFrame = 0;
            bool rejected = false;
            // Detected as a piece of a ground ring (the battle circle): never a light.
            bool ring = false;
            // Highest glow seen, decaying slowly, and the run of dark samples. A source that has clearly emitted is
            // not demoted by a few dark samples (a faint animation frame, a sample on a transparent texel).
            float peakGlow = 0.0f;
            uint32_t darkStreak = 0;
            // Cool colored (water, mist) in its last sample, the frame since which it has been on screen without a
            // break, and whether it has stayed on screen long enough to be scenery rather than a spell.
            bool coolTint = false;
            uint32_t visibleSince = 0;
            bool persistent = false;
        };
        std::unordered_map<uint64_t, EmissiveVerdict> emissiveVerdicts;
        uint32_t emissiveFrame = 0;
        uint32_t emissiveVerdictRejections = 0;
        uint32_t emissivePersistentRejections = 0;
        // Candidate pieces whose texture isn't listed in rt_emitters.json, or is listed but disabled or filtered out.
        uint32_t emissiveUnlistedPieces = 0;
        uint32_t emissiveDisabledPieces = 0;
        // Unlisted untextured pieces lit by the legacy glow.
        uint32_t emissiveLegacyPieces = 0;
        uint32_t emissiveRingKeys = 0;
        uint32_t emissiveRingKeyPieces = 0;

        // The glow of every light slot is read back a few frames later, with the key each slot held when it was sampled.
        static const uint32_t EmissiveReadbackDepth = 3;
        std::unique_ptr<RenderBuffer> emissiveReadback[EmissiveReadbackDepth];
        uint64_t emissiveReadbackKeys[EmissiveReadbackDepth][EmissiveSlots] = {};
        bool emissiveReadbackSampled[EmissiveReadbackDepth][EmissiveSlots] = {};
        bool emissiveReadbackPending[EmissiveReadbackDepth] = {};
        uint32_t emissiveCopyRingIndex = 0;
        uint64_t emissiveSlotKeys[EmissiveSlots] = {};
        bool emissiveSlotSampled[EmissiveSlots] = {};
        uint32_t emissiveClusterCount = 0;
        uint32_t emissiveClustersTooLarge = 0;
        uint32_t emissiveRingPieces = 0;
        uint32_t emissiveRings = 0;

        // Texture probes for material guesses: sampled by the emissive color dispatch into gHitNormalFog after the
        // emissive slots (two float4 per probe), and read back with the glow.
        static const uint32_t MaterialProbeSlots = 16;
        static const uint32_t HitNormalFogEntries = EmissiveSlots * 2 + MaterialProbeSlots * 2;
        std::vector<RtMaterialProbe> materialProbeRing[EmissiveReadbackDepth];
        std::vector<RtMaterialProbe> frameMaterialProbes;
        uint32_t materialGuessesThisSession = 0;
        // Texture key of each path traced draw by render index, this frame and as of the last diagnostics copy.
        std::vector<uint64_t> frameMaterialKeys;
        std::vector<uint64_t> diagMaterialKeys;
        std::vector<RtEmitterSignature> frameDrawSignatures;
        std::vector<RtEmitterSignature> diagDrawSignatures;
        // Texture previews being read back from the GPU for the material file.
        struct PendingMaterialPreview {
            uint64_t key = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t rowWidth = 0;
            uint32_t frame = 0;
            std::unique_ptr<RenderBuffer> readback;
        };
        std::vector<PendingMaterialPreview> pendingMaterialPreviews;

        // Diagnostics written to rt.log. The primary ray shader counts rays into hitInstanceIdBuffer
        // and writes probe values into hitColorBuffer; both are copied to diagReadback on a few frames.
        uint32_t diagFrame = 0;
        int32_t diagReadbackDelay = -1;
        bool diagCaptureThisFrame = false;
        std::unique_ptr<RenderBuffer> diagReadback;
        std::unique_ptr<RenderBuffer> diagZeroUpload;
        std::unique_ptr<RenderBuffer> diagTextureReadback;
        RenderFormat diagTextureFormats[8] = {};
        bool diagTexturesCopied = false;

        RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI graphicsAPI);
        ~RaytracingResources();
        void resetBottomLevelAS();
        void addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh);
        uint32_t addBillboardProxyMesh(const std::vector<float> &positions);
        void updateBottomLevelASResources(RenderWorker *worker);
        void submitBottomLevelASCreation(RenderWorker *worker);
        void updateTopLevelASResources(RenderWorker *worker, const std::vector<RaytracingInstance> &sceneInstances);
        void submitTopLevelASCreation(RenderWorker *worker);
        void setShaderReadBuffers(RenderBuffer *normalBuffer, RenderBuffer *shadedColorBuffer);
        void prepareForDispatch(RenderWorker *worker);
        void createShaderBindingTable(RenderWorker *worker, const struct RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups);
        void updateLightsBuffer(RenderWorker *worker);
        void updateEmissiveLights(const RaytracingScene &rtScene, const float cameraPosition[3]);
        void updateMaterialProbes(const RaytracingScene &rtScene);
        void copyMaterialPreviews(RenderWorker *worker, const RaytracingScene &rtScene);
        void readMaterialPreviews();
        void copyEmissiveGlow(RenderWorker *worker);
        void readEmissiveGlow();
        bool emissiveKeyRejected(uint64_t key) const;
        bool emissiveKeySuppressed(uint64_t key) const;
        float emissiveKeyGlow(uint64_t key) const;
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
        void createOutputBuffers(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t displayWidth, uint32_t displayHeight);
        void resolveUpscaler(RenderWorker *worker, uint32_t displayWidth, uint32_t displayHeight, uint32_t &renderWidth, uint32_t &renderHeight);
        void updateUpscalerFrame(uint64_t frameCount);
        void submitUpscale(RenderWorker *worker, float deltaTimeMs);
        void submitUpscalerInputs(RenderWorker *worker);
        void beginPassTiming(RenderWorker *worker);
        void writePassTimestamp(RenderWorker *worker, PassTimestamp timestamp);
        void endPassTiming();
        void beginBuildTiming(RenderWorker *worker);
        void endBuildTiming(RenderWorker *worker);
        void logUpscalerDiagnostics();
        void updateLightQuality();
        void createLightFilterPipelines();
        void submitLightTemporal(RenderWorker *worker);
        bool submitLightDenoise(RenderWorker *worker);
        void logLightQualityDiagnostics();
        void updateInterleavedRenderTargets(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t count, const RenderMultisampling &multisampling, bool usesHDR);
        void updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary);
        void updateMultisampling();
        Upscaler *getUpscaler(UpscaleMode mode) const;
        void logSceneDiagnostics(const RaytracingScene &rtScene);
        void clearDiagnosticBuffers(RenderWorker *worker);
        void copyDiagnosticBuffers(RenderWorker *worker);
        void assignDecalSlots(const RaytracingScene &rtScene, bool rotate);
        void evaluateDecalSlots(const uint32_t *counters, bool log);
        void copyDecalCheckCounters(RenderWorker *worker);
        void readDecalCheckCounters();
        void copyDiagnosticTextures(RenderWorker *worker, RenderTexture *finalColor, RenderFormat finalColorFormat, RenderTextureLayout finalColorLayout, uint32_t finalX, uint32_t finalY);
        void readDiagnosticBuffers();
    };

    void rtDiagLog(const char *fmt, ...);

    // Asks the renderer to write a full diagnostics dump, including every 3D draw that isn't path traced, for the next
    // ray traced frame. Safe to call from any thread (the game binds it to F9).
    void requestRaytracingDiagnosticsCapture();
    // The pixel the capture's "center" lines describe, as a fraction of the image; negative for the image center.
    void setRaytracingDiagnosticsPixel(float x, float y);
    void raytracingDiagnosticsPixel(float &x, float &y);
    void cycleRaytracingDebugView();
    uint32_t raytracingDebugView();
    void setRaytracingIndirectBounces(uint32_t bounces);
    void setRaytracingAdvancedWater(bool enabled);
    bool raytracingAdvancedWater();
    // Graphics menu "Volumetric Fog": the fog of rt_lights.json's room settings is ray marched.
    void setRaytracingVolumetricFog(bool enabled);
    bool raytracingVolumetricFog();
    // Graphics menu "God Rays": screen space rays from the sky around the sun or moon and from window textures.
    void setRaytracingGodRays(bool enabled);
    bool raytracingGodRays();
    void toggleRaytracingSecondBounce();
    uint32_t raytracingIndirectBounces();
    bool raytracingDiagnosticsCaptureRequested();
    bool consumeRaytracingDiagnosticsCapture();
    // Graphics menu "Upscaler", "Quality" and "Sharpness": mode is a RaytracingUpscaler, quality is 0 (Native AA),
    // 1 (Quality), 2 (Balanced), 3 (Performance) or 4 (Ultra Performance), sharpness is from 0 to 1.
    void setRaytracingUpscaler(uint32_t mode, uint32_t quality, float sharpness);
    // Graphics menu "Light Samples" (0: 1 ray, 1: 2 rays, 2: 4 rays), "Denoiser" (0: Off, 1: Low, 2: High) and "Light
    // Accumulation" (0: Responsive, 1: Balanced, 2: Stable).
    void setRaytracingLightQuality(uint32_t samples, uint32_t denoiser, uint32_t accumulation);
};
