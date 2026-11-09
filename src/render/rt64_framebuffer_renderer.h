//
// RT64
//

#pragma once

#include <stdint.h>
#include <string>

#include "common/rt64_emulator_configuration.h"
#include "common/rt64_user_configuration.h"
#include "hle/rt64_framebuffer_manager.h"
#include "hle/rt64_workload.h"
#include "preset/rt64_preset_scene.h"
#include "shared/rt64_frame_params.h"
#include "shared/rt64_gpu_tile.h"
#include "shared/rt64_render_indices.h"
#include "shared/rt64_render_params.h"

#include "rt64_buffer_uploader.h"
#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raster_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_rsp_processor.h"
#include "rt64_vertex_processor.h"

#if RT_ENABLED
#   include "rt64_raytracing_resources.h"
#   include "rt64_raytracing_shader_cache.h"
#endif

namespace RT64 {
    struct DynamicTextureView {
        const RenderTexture *texture = nullptr;
        uint32_t dstIndex = 0;
        const RenderTextureView *textureView = nullptr;
    };

    struct RasterScene {
        std::vector<uint32_t> instanceIndices;
        bool lightable = false;
        // The ray traced scene that follows lights this scene, so the shadow map pass is skipped.
        bool rtLit = false;
        float nativeWidth = 0.0f;
        float nativeHeight = 0.0f;
        float viewportX = 0.0f;
        float viewportY = 0.0f;
        float viewportWidth = 0.0f;
        float viewportHeight = 0.0f;

        RasterScene();
    };

    struct RenderTargetDrawCall {
        typedef std::pair<uint32_t, bool> SceneIndexPair;

        RenderFramebufferStorage *fbStorage = nullptr;
        std::vector<RasterScene> rasterScenes;
        std::vector<SceneIndexPair> sceneIndices;
#   if RT_ENABLED
        std::vector<RaytracingScene> rtScenes;
#   endif
    };

    struct RSPSmoothNormalGenerationCB {
        uint32_t indexStart;
        uint32_t indexCount;
    };

    struct FramebufferRenderer {
        std::vector<uint32_t> textureCacheVersions;
        std::vector<Texture *> textureCacheTextures;
        std::vector<Texture *> textureCacheTextureReplacements;
        std::vector<uint32_t> textureCacheFreeSpaces;
        uint32_t textureCacheSize = 0;
        uint32_t textureCacheGlobalVersion = 0;
        bool textureCacheReplacementMapEnabled = false;
        std::vector<InstanceDrawCall> instanceDrawCallVector;
        std::vector<RenderPipelineProgram> hitGroupVector;
        std::vector<interop::RenderIndices> renderIndicesVector;
        // Ray traced surface material of each render index (see RtMaterialDatabase), kept out of RenderIndices because
        // the raster shaders share that struct's layout.
        std::vector<uint32_t> rtMaterialsVector;
        std::vector<DynamicTextureView> dynamicTextureViewVector;
        std::vector<RenderTextureBarrier> dynamicTextureBarrierVector;
        std::unique_ptr<BufferUploader> shaderUploader;
        std::vector<RSPSmoothNormalGenerationCB> rspSmoothNormalVector;
        std::array<RenderInputSlot, 3> vertexInputSlots;
        std::array<RenderVertexBufferView, 3> indexedVertexViews;
        std::array<RenderVertexBufferView, 3> rawVertexViews;
        RenderIndexBufferView indexBufferView;
        RenderBuffer *testZIndexBuffer = nullptr;
        RenderIndexBufferView testZIndexBufferView;
        BufferPair renderIndicesBuffer;
        BufferPair rtMaterialsBuffer;
        BufferPair interleavedRastersBuffer;
        uint32_t interleavedRastersCount = 0;
        BufferPair frameParamsBuffer;
        RenderPipelineLayout *rendererPipelineLayout = nullptr;
        RenderPipeline *postBlendDitherNoiseAddPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubPipeline = nullptr;
        RenderPipeline *postBlendDitherNoiseSubNegativePipeline = nullptr;
        std::unique_ptr<FramebufferRendererDescriptorCommonSet> descCommonSet;
        std::unique_ptr<FramebufferRendererDescriptorTextureSet> descTextureSet;
        std::unique_ptr<RenderTexture> dummyColorTarget;
        std::unique_ptr<RenderTexture> dummyDepthTarget;
        std::unique_ptr<RenderTextureView> dummyColorTargetView;
        std::unique_ptr<RenderTextureView> dummyDepthTargetView;
        bool dummyColorTargetTransitioned = false;
        bool dummyDepthTargetTransitioned = false;
        std::vector<uint32_t> descriptorTextureVersions;
        uint32_t descriptorTextureGlobalVersion = 0;
        bool descriptorTextureReplacementMapEnabled = false;
        std::unique_ptr<RSPSmoothNormalDescriptorSet> smoothDescSet;
        std::unique_ptr<RSPVertexTestZDescriptorSet> vertexTestZSet;
        interop::FrameParams frameParams;
        const ShaderLibrary *shaderLibrary = nullptr;

#   if RT_ENABLED
        const RenderTexture *blueNoiseTexture = nullptr;
        const RaytracingState *rtState = nullptr;
        const RenderPipelineLayout *rtPipelineLayout = nullptr;
        std::unique_ptr<RaytracingResources> rtResources;
        bool rtSupport = false;
        // Emissive candidates and shadow casters collected from every 3D draw of the frame, including the ones outside
        // the path traced scene (drawn after the HUD split, in another framebuffer pair or through another camera).
        // They are merged into the chosen ray traced scene in endFramebuffers.
        RaytracingScene rtFrameExtras;
        // Per draw details for an on demand diagnostics capture (F9).
        bool rtDiagRecording = false;
        std::vector<std::string> rtDiagDrawRecords;
#   endif

        struct Framebuffer {
            std::unique_ptr<RenderBuffer> paramsBuffer;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descRealFbSet;
            std::unique_ptr<FramebufferRendererDescriptorFramebufferSet> descDummyFbSet;
            std::set<RenderTarget *> transitionRenderTargetSet;
            RenderTargetDrawCall renderTargetDrawCall;
            RenderViewport viewport;
        };

        std::vector<Framebuffer> framebufferVector;
        std::vector<BufferUploader *> pendingUploaders;
        uint32_t framebufferCount = 0;
        std::unique_ptr<RenderTexture> shadowColor;
        std::unique_ptr<RenderTextureView> shadowColorView;
        std::unique_ptr<RenderTexture> shadowDepth;
        std::unique_ptr<RenderFramebuffer> shadowFramebuffer;
        std::vector<std::unique_ptr<ShadowMapDescriptorSet>> shadowMapSets;
        uint32_t shadowMapSetCursor = 0;
        const RenderBuffer *lightingWorldPosBuffer = nullptr;
        uint32_t lightingWorldPosSize = 0;
        const DrawData *lightingDrawData = nullptr;
        SceneLight lightingSceneLight;
        std::unique_ptr<RenderPipeline> shadowCutoutPipeline;
        const RenderPipelineLayout *shadowCutoutPipelineLayout = nullptr;

        // Light space positions for the cutout shadow casters, written by the CPU. A chunk is never reused
        // within a workload because its earlier contents may still be read by commands recorded before.
        struct ShadowCutoutChunk {
            std::unique_ptr<RenderBuffer> buffer;
            uint64_t capacity = 0;
        };
        std::vector<ShadowCutoutChunk> shadowCutoutChunks;
        uint32_t shadowCutoutChunkCursor = 0;

        struct DrawParams {
            RenderWorker *worker;
            RenderFramebufferStorage *fbStorage;
            const Workload *curWorkload;
            uint32_t fbPairIndex;
            uint32_t fbWidth;
            uint32_t fbHeight;
            uint32_t targetWidth;
            uint32_t targetHeight;
            RasterShaderCache *rasterShaderCache;
            hlslpp::float2 resolutionScale;
            float aspectRatioSource;
            float aspectRatioTarget;
            float extAspectPercentage;
            float horizontalMisalignment;
            PresetScene presetScene;
            bool rtEnabled = false;
            bool advancedLighting = false;
            uint64_t submissionFrame;
            float deltaTimeMs;
            bool ubershadersOnly;
            bool fixRectLR;
            bool postBlendNoise;
            bool postBlendNoiseNegative;
            uint32_t maxGameCall;
        };

        FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary);
        ~FramebufferRenderer();
        void resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling);
        void updateTextureCache(TextureCache *textureCache);
        void createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, TextureCache *textureCache, uint64_t submissionFrame);
        uint32_t getDestinationIndex();
        uint32_t getTextureIndex(RenderTarget *renderTarget);
        uint32_t getTextureIndex(const FramebufferManager::TileCopy &tileCopy);
        void updateMultisampling();
        void updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers);
        void updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool raytracingEnabled);
        void submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers);
        bool submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState);
        void submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState);
        void submitSceneLighting(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState);
        void addFramebuffer(const DrawParams &p);
        void endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled);
        void recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor, VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled);
        void recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex);
        void waitForUploaders();
        void advanceFrame(bool rtEnabled);

#   if RT_ENABLED
        void resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture);
        void updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene);
        void submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene);
        void setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged);
#   endif
    };
};