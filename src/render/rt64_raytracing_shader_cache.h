//
// RT64
//

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/rt64_plume.h"

namespace RT64 {
    struct ShaderDescription;
    struct ShaderLibrary;

    // The framebuffer renderer looks up hit groups with this key.
    constexpr uint64_t UberShaderHash = 0;

    struct RaytracingShaderPrograms {
        RenderPipelineProgram surface;
        RenderPipelineProgram shadow;
    };

    struct RaytracingState {
        std::unique_ptr<RenderPipeline> pipeline;
        std::unordered_map<uint64_t, RaytracingShaderPrograms> shaderProgramsMap;
        std::vector<RenderPipelineProgram> rayGenPrograms;
        std::vector<RenderPipelineProgram> missPrograms;
        bool valid = false;
    };

    struct RaytracingShaderCache {
        RenderDevice *device = nullptr;
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        const ShaderLibrary *shaderLibrary = nullptr;
        std::unique_ptr<RenderPipelineLayout> pipelineLayout;
        std::unique_ptr<RenderShader> libraryShader;
        std::vector<RaytracingState> states;
        bool setupDone = false;

        // The driver compiles the ray tracing pipeline for several seconds, so it's built on its own thread and
        // swapped in by update() once ready. Until then states[0] stays invalid and frames are rasterized.
        std::thread setupThread;
        std::atomic<bool> setupFinished = false;
        std::unique_ptr<RenderPipelineLayout> pendingPipelineLayout;
        std::unique_ptr<RenderShader> pendingLibraryShader;
        RaytracingState pendingState;

        RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary);
        ~RaytracingShaderCache();
        bool isSetup() const;
        void setup();
        void update();
        void buildPipeline();
        void submit(const ShaderDescription &shaderDesc);
        int32_t getActiveState() const;
        void setNextState();
    };
};
