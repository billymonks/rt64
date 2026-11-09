//
// RT64
//

#include "rt64_raytracing_shader_cache.h"

#include <chrono>
#include <cstdio>

#if defined(_WIN32)
#include "plume_d3d12.h"
#endif
#include "rt64_descriptor_sets.h"
#include "rt64_shader_library.h"

#if defined(_WIN32)
#include "shaders/RtLibrary.hlsl.dxil.h"
#endif
#if !defined(__APPLE__)
#include "shaders/RtLibrary.hlsl.spirv.h"
#endif

namespace RT64 {
    void rtDiagLog(const char *fmt, ...);

    RaytracingShaderCache::RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary) {
        this->device = device;
        this->shaderFormat = shaderFormat;
        this->shaderLibrary = shaderLibrary;
    }

    bool RaytracingShaderCache::isSetup() const {
        return setupDone;
    }

    int32_t RaytracingShaderCache::getActiveState() const {
        return 0;
    }

    void RaytracingShaderCache::setNextState() {
    }

    void RaytracingShaderCache::submit(const ShaderDescription &) {
        // Every surface uses the uber hit group. Specialized materials can replace this later.
    }

    RaytracingShaderCache::~RaytracingShaderCache() {
        if (setupThread.joinable()) {
            setupThread.join();
        }
    }

    void RaytracingShaderCache::setup() {
        if (setupDone) {
            return;
        }

        setupDone = true;
        states.resize(1);
        if ((device == nullptr) || (shaderLibrary == nullptr)) {
            return;
        }

        setupThread = std::thread([this]() {
            const auto start = std::chrono::steady_clock::now();
            buildPipeline();
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            rtDiagLog("ray tracing pipeline built in %.2f s on a background thread (%s)", seconds, pendingState.valid ? "ready" : "failed");
            setupFinished = true;
        });
    }

    void RaytracingShaderCache::update() {
        if (!setupFinished || !setupThread.joinable()) {
            return;
        }

        setupThread.join();
        if (pendingState.valid) {
            pipelineLayout = std::move(pendingPipelineLayout);
            libraryShader = std::move(pendingLibraryShader);
            states[0] = std::move(pendingState);
        }
    }

    void RaytracingShaderCache::buildPipeline() {
        const void *blob = nullptr;
        uint64_t blobSize = 0;
#if defined(_WIN32)
        if (shaderFormat == RenderShaderFormat::DXIL) {
            blob = RtLibraryBlobDXIL;
            blobSize = sizeof(RtLibraryBlobDXIL);
        }
#endif
#if !defined(__APPLE__)
        if (shaderFormat == RenderShaderFormat::SPIRV) {
            blob = RtLibraryBlobSPIRV;
            blobSize = sizeof(RtLibraryBlobSPIRV);
        }
#endif
        if ((blob == nullptr) || (blobSize == 0)) {
            fprintf(stderr, "Ray tracing shaders are not built for this graphics API.\n");
            return;
        }

        RenderPipelineLayoutBuilder layoutBuilder;
        FramebufferRendererDescriptorCommonSet commonSet(shaderLibrary->samplerLibrary, true);
        FramebufferRendererDescriptorTextureSet textureSet;
        FramebufferRendererDescriptorFramebufferSet framebufferSet;
        // Descriptors are bound on the command list, so this stays a global root signature. The texture cache sets
        // stay boundless: the hit shaders index gTextures/gTMEM like the raster shaders do.
        layoutBuilder.begin(false);
        layoutBuilder.addDescriptorSet(commonSet);
        layoutBuilder.addDescriptorSet(textureSet);
        layoutBuilder.addDescriptorSet(textureSet);
        layoutBuilder.addDescriptorSet(framebufferSet);
        layoutBuilder.end();
        pendingPipelineLayout = layoutBuilder.create(device);

        pendingLibraryShader = device->createShader(blob, blobSize, nullptr, shaderFormat);

        // dxc -exports keeps these library entry points unmangled. Mangled names crash the driver
        // when a hit group imports them.
        static const char *kPrimaryRayGen = "PrimaryRayGen";
        static const char *kDirectRayGen = "DirectRayGen";
        static const char *kIndirectRayGen = "IndirectRayGen";
        static const char *kReflectionRayGen = "ReflectionRayGen";
        static const char *kRefractionRayGen = "RefractionRayGen";
        static const char *kEmissiveColorRayGen = "EmissiveColorRayGen";
        static const char *kPrimaryMiss = "PrimaryMiss";
        static const char *kShadowMiss = "ShadowMiss";
        static const char *kSurfaceClosestHit = "SurfaceClosestHit";
        static const char *kSurfaceAnyHit = "SurfaceAnyHit";
        static const char *kShadowAnyHit = "ShadowAnyHit";

        RenderRaytracingPipelineLibrarySymbol symbols[] = {
            { kPrimaryRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kPrimaryRayGen },
            { kDirectRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kDirectRayGen },
            { kIndirectRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kIndirectRayGen },
            { kReflectionRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kReflectionRayGen },
            { kRefractionRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kRefractionRayGen },
            { kEmissiveColorRayGen, RenderRaytracingPipelineLibrarySymbolType::RAYGEN, kEmissiveColorRayGen },
            { kPrimaryMiss, RenderRaytracingPipelineLibrarySymbolType::MISS, kPrimaryMiss },
            { kShadowMiss, RenderRaytracingPipelineLibrarySymbolType::MISS, kShadowMiss },
            { kSurfaceClosestHit, RenderRaytracingPipelineLibrarySymbolType::CLOSEST_HIT, kSurfaceClosestHit },
            { kSurfaceAnyHit, RenderRaytracingPipelineLibrarySymbolType::ANY_HIT, kSurfaceAnyHit },
            { kShadowAnyHit, RenderRaytracingPipelineLibrarySymbolType::ANY_HIT, kShadowAnyHit },
        };
        RenderRaytracingPipelineHitGroup hitGroups[] = {
            { "SurfaceHitGroup", kSurfaceClosestHit, kSurfaceAnyHit, nullptr },
            { "ShadowHitGroup", nullptr, kShadowAnyHit, nullptr },
        };
        RenderRaytracingPipelineLibrary libraries[] = {
            { pendingLibraryShader.get(), symbols, uint32_t(std::size(symbols)) },
        };

        RenderRaytracingPipelineDesc pipelineDesc;
        pipelineDesc.libraries = libraries;
        pipelineDesc.librariesCount = 1;
        pipelineDesc.hitGroups = hitGroups;
        pipelineDesc.hitGroupsCount = uint32_t(std::size(hitGroups));
        pipelineDesc.pipelineLayout = pendingPipelineLayout.get();
        // HitPayload in RtLibrary.hlsl: 19 floats.
        pipelineDesc.maxPayloadSize = 80;
        pipelineDesc.maxAttributeSize = sizeof(float) * 2;
        pipelineDesc.maxRecursionDepth = 1;

        RaytracingState &state = pendingState;
        state.pipeline = device->createRaytracingPipeline(pipelineDesc);
#if defined(_WIN32)
        auto *nativePipeline = static_cast<plume::D3D12RaytracingPipeline *>(state.pipeline.get());
        const bool pipelineReady = (nativePipeline != nullptr) && (nativePipeline->stateObject != nullptr) && (nativePipeline->nameProgramMap.find(kPrimaryRayGen) != nativePipeline->nameProgramMap.end());
#else
        const bool pipelineReady = state.pipeline != nullptr;
#endif
        if (!pipelineReady) {
            fprintf(stderr, "Failed to create the ray tracing pipeline.\n");
            return;
        }

        auto program = [&](const char *name) {
            return state.pipeline->getProgram(name);
        };
        RaytracingShaderPrograms programs;
        programs.surface = program("SurfaceHitGroup");
        programs.shadow = program("ShadowHitGroup");
        state.shaderProgramsMap[UberShaderHash] = programs;
        state.rayGenPrograms = {
            program(kPrimaryRayGen),
            program(kDirectRayGen),
            program(kIndirectRayGen),
            program(kReflectionRayGen),
            program(kRefractionRayGen),
            program(kEmissiveColorRayGen),
        };
        state.missPrograms = {
            program(kPrimaryMiss),
            program(kShadowMiss),
        };
        state.valid = true;
        fprintf(stdout, "Ray tracing pipeline ready.\n");
    }
};
