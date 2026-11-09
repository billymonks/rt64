//
// RT64
//

#pragma once

#include <memory>
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

        RaytracingShaderCache(RenderDevice *device, RenderShaderFormat shaderFormat, const ShaderLibrary *shaderLibrary);
        bool isSetup() const;
        void setup();
        void submit(const ShaderDescription &shaderDesc);
        int32_t getActiveState() const;
        void setNextState();
    };
};
