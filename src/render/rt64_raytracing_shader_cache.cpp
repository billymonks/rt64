//
// RT64
//

#include "rt64_raytracing_shader_cache.h"

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

    void RaytracingShaderCache::setup() {
        if (setupDone || (device == nullptr) || (shaderLibrary == nullptr)) {
            setupDone = true;
            return;
        }

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
            setupDone = true;
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
        pipelineLayout = layoutBuilder.create(device);

        libraryShader = device->createShader(blob, blobSize, nullptr, shaderFormat);

        // dxc -exports keeps these library entry points unmangled. Mangled names crash the driver
        // when a hit group imports them.
        static const char *kPrimaryRayGen = "PrimaryRayGen";
        static const char *kDirectRayGen = "DirectRayGen";
        static const char *kIndirectRayGen = "IndirectRayGen";
        static const char *kReflectionRayGen = "ReflectionRayGen";
        static const char *kRefractionRayGen = "RefractionRayGen";
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
            { libraryShader.get(), symbols, uint32_t(std::size(symbols)) },
        };

        RenderRaytracingPipelineDesc pipelineDesc;
        pipelineDesc.libraries = libraries;
        pipelineDesc.librariesCount = 1;
        pipelineDesc.hitGroups = hitGroups;
        pipelineDesc.hitGroupsCount = uint32_t(std::size(hitGroups));
        pipelineDesc.pipelineLayout = pipelineLayout.get();
        pipelineDesc.maxPayloadSize = 64;
        pipelineDesc.maxAttributeSize = sizeof(float) * 2;
        pipelineDesc.maxRecursionDepth = 1;

        states.resize(1);
        RaytracingState &state = states[0];
        state.pipeline = device->createRaytracingPipeline(pipelineDesc);
#if defined(_WIN32)
        auto *nativePipeline = static_cast<plume::D3D12RaytracingPipeline *>(state.pipeline.get());
        const bool pipelineReady = (nativePipeline != nullptr) && (nativePipeline->stateObject != nullptr) && (nativePipeline->nameProgramMap.find(kPrimaryRayGen) != nativePipeline->nameProgramMap.end());
#else
        const bool pipelineReady = state.pipeline != nullptr;
#endif
        if (!pipelineReady) {
            fprintf(stderr, "Failed to create the ray tracing pipeline.\n");
            setupDone = true;
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
        };
        state.missPrograms = {
            program(kPrimaryMiss),
            program(kShadowMiss),
        };
        state.valid = true;
        setupDone = true;
        fprintf(stdout, "Ray tracing pipeline ready.\n");
    }
};
