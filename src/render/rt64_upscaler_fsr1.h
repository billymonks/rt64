//
// RT64
//

#pragma once

#include "rt64_upscaler.h"

namespace RT64 {
    struct FsrUpscaleDescriptorSet;
    struct SamplerLibrary;

    // AMD FidelityFX Super Resolution 1: EASU spatial upscaling followed by RCAS sharpening. Works on any GPU and API.
    // The pipelines are created the first time it upscales.
    struct FSR1Upscaler : Upscaler {
        RenderDevice *device = nullptr;
        RenderShaderFormat shaderFormat = RenderShaderFormat::UNKNOWN;
        const SamplerLibrary *samplerLibrary = nullptr;
        std::unique_ptr<RenderPipelineLayout> easuPipelineLayout;
        std::unique_ptr<RenderPipeline> easuPipeline;
        std::unique_ptr<RenderPipelineLayout> rcasPipelineLayout;
        std::unique_ptr<RenderPipeline> rcasPipeline;
        std::unique_ptr<FsrUpscaleDescriptorSet> easuSet;
        std::unique_ptr<FsrUpscaleDescriptorSet> rcasSet;
        std::unique_ptr<RenderTexture> sharpenInput;
        uint32_t sharpenInputWidth = 0;
        uint32_t sharpenInputHeight = 0;
        bool pipelinesCreated = false;
        bool pipelinesValid = false;

        FSR1Upscaler(RenderDevice *device, RenderShaderFormat shaderFormat, const SamplerLibrary *samplerLibrary);
        ~FSR1Upscaler() override;
        void set(RenderWorker *worker, QualityMode inQuality, int renderWidth, int renderHeight, int displayWidth, int displayHeight) override;
        bool getQualityInformation(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight) override;
        int getJitterPhaseCount(int renderWidth, int displayWidth) override;
        void getJitterOffset(int index, int phaseCount, float &x, float &y) override;
        void upscale(RenderWorker *worker, const UpscaleParameters &p) override;
        bool isInitialized() const override;
        bool requiresNonShaderResourceInputs() const override;
        bool isTemporal() const override;
        const char *getName() const override;
        bool createPipelines();
    };
};
