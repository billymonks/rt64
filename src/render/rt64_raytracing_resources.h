//
// RT64
//

#pragma once

#include <memory>
#include <vector>

#include "common/rt64_common.h"
#include "common/rt64_plume.h"
#include "common/rt64_user_configuration.h"
#include "preset/rt64_preset_scene.h"
#include "render/rt64_buffer_uploader.h"
#include "render/rt64_upscaler.h"
#include "shared/rt64_interleaved_raster.h"
#include "shared/rt64_point_light.h"
#include "shared/rt64_raytracing_params.h"

namespace RT64 {
    struct InstanceDrawCall;
    struct RenderTarget;
    struct RenderFramebufferStorage;
    struct ShaderLibrary;

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
        uint32_t billboardsCylindrical = 0;
        uint32_t billboardsSpherical = 0;
        uint32_t billboardProxies = 0;
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
    };

    struct RaytracingResources {
        static const uint32_t MaxHitQueries = 24;

        RenderDevice *device = nullptr;
        interop::RaytracingParams rtParams;
        BufferPair rtParamsBuffer;

        bool swapBuffers = false;
        bool skipReprojection = false;
        bool denoiserEnabled = true;
        bool upscaleActive = false;
        bool upscalerReactiveMask = false;
        bool upscalerLockMask = false;
        bool transitionOutputBuffers = true;
        bool updateOutputBuffers = true;
        UpscaleMode upscalerMode = UpscaleMode::Bilinear;
        int maxReflections = 0;
        uint32_t textureWidth = 1;
        uint32_t textureHeight = 1;

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

        struct BottomLevel {
            RenderBottomLevelASMesh mesh;
            RenderBottomLevelASBuildInfo buildInfo;
            std::unique_ptr<RenderBuffer> buffer;
            std::unique_ptr<RenderAccelerationStructure> accelerationStructure;
        };
        std::vector<BottomLevel> bottomLevelASVector;
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
        void updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene);
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
        void createOutputBuffers(RenderWorker *worker, uint32_t width, uint32_t height);
        void updateInterleavedRenderTargets(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t count, const RenderMultisampling &multisampling, bool usesHDR);
        void updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary);
        void updateMultisampling();
        Upscaler *getUpscaler(UpscaleMode mode) const;
        void logSceneDiagnostics(const RaytracingScene &rtScene);
        void clearDiagnosticBuffers(RenderWorker *worker);
        void copyDiagnosticBuffers(RenderWorker *worker);
        void copyDiagnosticTextures(RenderWorker *worker, RenderTexture *finalColor, RenderFormat finalColorFormat, RenderTextureLayout finalColorLayout, uint32_t finalX, uint32_t finalY);
        void readDiagnosticBuffers();
    };

    void rtDiagLog(const char *fmt, ...);
};
