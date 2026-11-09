//
// RT64
//

#include "rt64_upscaler_fsr1.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "rt64_descriptor_sets.h"
#include "rt64_sampler_library.h"

#include "shaders/FsrEasuCS.hlsl.spirv.h"
#include "shaders/FsrRcasCS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/FsrEasuCS.hlsl.dxil.h"
#   include "shaders/FsrRcasCS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/FsrEasuCS.hlsl.metal.h"
#   include "shaders/FsrRcasCS.hlsl.metal.h"
#endif

namespace RT64 {
    // Push constants of FsrEasuCS.hlsl and FsrRcasCS.hlsl.
    struct FsrEasuCB {
        uint32_t con0[4];
        uint32_t con1[4];
        uint32_t con2[4];
        uint32_t con3[4];
        uint32_t outputSize[2];
        uint32_t padding[2];
    };

    struct FsrRcasCB {
        uint32_t con[4];
        uint32_t outputSize[2];
        uint32_t padding[2];
    };

    static uint32_t floatBits(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        return bits;
    }

    static std::unique_ptr<RenderShader> createFsrShader(RenderDevice *device, RenderShaderFormat format, bool easu) {
        const void *blob = nullptr;
        uint64_t blobSize = 0;
#   ifdef _WIN32
        if (format == RenderShaderFormat::DXIL) {
            blob = easu ? FsrEasuCSBlobDXIL : FsrRcasCSBlobDXIL;
            blobSize = easu ? sizeof(FsrEasuCSBlobDXIL) : sizeof(FsrRcasCSBlobDXIL);
        }
#   elif defined(__APPLE__)
        if (format == RenderShaderFormat::METAL) {
            blob = easu ? FsrEasuCSBlobMSL : FsrRcasCSBlobMSL;
            blobSize = easu ? sizeof(FsrEasuCSBlobMSL) : sizeof(FsrRcasCSBlobMSL);
        }
#   endif
        if (format == RenderShaderFormat::SPIRV) {
            blob = easu ? FsrEasuCSBlobSPIRV : FsrRcasCSBlobSPIRV;
            blobSize = easu ? sizeof(FsrEasuCSBlobSPIRV) : sizeof(FsrRcasCSBlobSPIRV);
        }

        if (blob == nullptr) {
            return nullptr;
        }

        return device->createShader(blob, blobSize, "CSMain", format);
    }

    // FSR1Upscaler

    FSR1Upscaler::FSR1Upscaler(RenderDevice *device, RenderShaderFormat shaderFormat, const SamplerLibrary *samplerLibrary) {
        this->device = device;
        this->shaderFormat = shaderFormat;
        this->samplerLibrary = samplerLibrary;
    }

    FSR1Upscaler::~FSR1Upscaler() = default;

    bool FSR1Upscaler::createPipelines() {
        if (pipelinesCreated) {
            return pipelinesValid;
        }

        pipelinesCreated = true;
        if ((device == nullptr) || (samplerLibrary == nullptr)) {
            return false;
        }

        std::unique_ptr<RenderShader> easuShader = createFsrShader(device, shaderFormat, true);
        std::unique_ptr<RenderShader> rcasShader = createFsrShader(device, shaderFormat, false);
        if ((easuShader == nullptr) || (rcasShader == nullptr)) {
            return false;
        }

        RenderPipelineLayoutBuilder layoutBuilder;
        FsrUpscaleDescriptorSet easuLayoutSet(*samplerLibrary, true);
        layoutBuilder.begin();
        layoutBuilder.addPushConstant(0, 0, sizeof(FsrEasuCB), RenderShaderStageFlag::COMPUTE);
        layoutBuilder.addDescriptorSet(easuLayoutSet);
        layoutBuilder.end();
        easuPipelineLayout = layoutBuilder.create(device);
        easuPipeline = device->createComputePipeline(RenderComputePipelineDesc(easuPipelineLayout.get(), easuShader.get(), 8, 8, 1));

        FsrUpscaleDescriptorSet rcasLayoutSet(*samplerLibrary, false);
        layoutBuilder.begin();
        layoutBuilder.addPushConstant(0, 0, sizeof(FsrRcasCB), RenderShaderStageFlag::COMPUTE);
        layoutBuilder.addDescriptorSet(rcasLayoutSet);
        layoutBuilder.end();
        rcasPipelineLayout = layoutBuilder.create(device);
        rcasPipeline = device->createComputePipeline(RenderComputePipelineDesc(rcasPipelineLayout.get(), rcasShader.get(), 8, 8, 1));

        easuSet = std::make_unique<FsrUpscaleDescriptorSet>(*samplerLibrary, true, device);
        rcasSet = std::make_unique<FsrUpscaleDescriptorSet>(*samplerLibrary, false, device);
        pipelinesValid = (easuPipeline != nullptr) && (rcasPipeline != nullptr);
        return pipelinesValid;
    }

    void FSR1Upscaler::set(RenderWorker *worker, QualityMode inQuality, int renderWidth, int renderHeight, int displayWidth, int displayHeight) {
        createPipelines();
    }

    bool FSR1Upscaler::getQualityInformation(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight) {
        getRenderSize(quality, displayWidth, displayHeight, renderWidth, renderHeight);
        return true;
    }

    int FSR1Upscaler::getJitterPhaseCount(int renderWidth, int displayWidth) {
        return 1;
    }

    void FSR1Upscaler::getJitterOffset(int index, int phaseCount, float &x, float &y) {
        x = 0.0f;
        y = 0.0f;
    }

    void FSR1Upscaler::upscale(RenderWorker *worker, const UpscaleParameters &p) {
        if (!createPipelines() || (p.inColor == nullptr) || (p.outColor == nullptr) || (p.outWidth == 0) || (p.outHeight == 0)) {
            return;
        }

        const bool sharpen = (p.sharpness > 0.0f);
        if (sharpen && ((sharpenInput == nullptr) || (sharpenInputWidth != p.outWidth) || (sharpenInputHeight != p.outHeight))) {
            RenderTextureDesc desc = RenderTextureDesc::Texture2D(p.outWidth, p.outHeight, 1, RenderFormat::R16G16B16A16_FLOAT, RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS);
            desc.committed = true;
            sharpenInput = device->createTexture(desc);
            sharpenInputWidth = p.outWidth;
            sharpenInputHeight = p.outHeight;
        }

        RenderTexture *easuOutput = sharpen ? sharpenInput.get() : p.outColor;
        const uint32_t dispatchX = (p.outWidth + 7) / 8;
        const uint32_t dispatchY = (p.outHeight + 7) / 8;

        // EASU, from the render resolution to the display resolution.
        {
            const float inputWidth = float(p.inRect.w);
            const float inputHeight = float(p.inRect.h);
            // The input texture is exactly the size of the ray traced image.
            const float textureWidth = inputWidth;
            const float textureHeight = inputHeight;
            const float outputWidth = float(p.outWidth);
            const float outputHeight = float(p.outHeight);
            FsrEasuCB cb = {};
            cb.con0[0] = floatBits(inputWidth / outputWidth);
            cb.con0[1] = floatBits(inputHeight / outputHeight);
            cb.con0[2] = floatBits(0.5f * inputWidth / outputWidth - 0.5f);
            cb.con0[3] = floatBits(0.5f * inputHeight / outputHeight - 0.5f);
            cb.con1[0] = floatBits(1.0f / textureWidth);
            cb.con1[1] = floatBits(1.0f / textureHeight);
            cb.con1[2] = floatBits(1.0f / textureWidth);
            cb.con1[3] = floatBits(-1.0f / textureHeight);
            cb.con2[0] = floatBits(-1.0f / textureWidth);
            cb.con2[1] = floatBits(2.0f / textureHeight);
            cb.con2[2] = floatBits(1.0f / textureWidth);
            cb.con2[3] = floatBits(2.0f / textureHeight);
            cb.con3[0] = floatBits(0.0f);
            cb.con3[1] = floatBits(4.0f / textureHeight);
            cb.outputSize[0] = p.outWidth;
            cb.outputSize[1] = p.outHeight;

            easuSet->setTexture(easuSet->gInput, p.inColor, RenderTextureLayout::SHADER_READ);
            easuSet->setTexture(easuSet->gOutput, easuOutput, RenderTextureLayout::GENERAL);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(easuOutput, RenderTextureLayout::GENERAL));
            worker->commandList->setPipeline(easuPipeline.get());
            worker->commandList->setComputePipelineLayout(easuPipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &cb);
            worker->commandList->setComputeDescriptorSet(easuSet->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);
        }

        // RCAS on the EASU output. The sharpness is in stops of reduction from the strongest: 0 at full sharpness,
        // 2 (a quarter of it) near the lowest setting.
        if (sharpen) {
            const float stops = (1.0f - std::min(p.sharpness, 1.0f)) * 2.0f;
            FsrRcasCB cb = {};
            cb.con[0] = floatBits(std::exp2(-stops));
            cb.outputSize[0] = p.outWidth;
            cb.outputSize[1] = p.outHeight;

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
                RenderTextureBarrier(sharpenInput.get(), RenderTextureLayout::SHADER_READ),
                RenderTextureBarrier(p.outColor, RenderTextureLayout::GENERAL)
            });

            rcasSet->setTexture(rcasSet->gInput, sharpenInput.get(), RenderTextureLayout::SHADER_READ);
            rcasSet->setTexture(rcasSet->gOutput, p.outColor, RenderTextureLayout::GENERAL);
            worker->commandList->setPipeline(rcasPipeline.get());
            worker->commandList->setComputePipelineLayout(rcasPipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &cb);
            worker->commandList->setComputeDescriptorSet(rcasSet->get(), 0);
            worker->commandList->dispatch(dispatchX, dispatchY, 1);
        }
    }

    bool FSR1Upscaler::isInitialized() const {
        return !pipelinesCreated || pipelinesValid;
    }

    bool FSR1Upscaler::requiresNonShaderResourceInputs() const {
        return false;
    }

    bool FSR1Upscaler::isTemporal() const {
        return false;
    }

    const char *FSR1Upscaler::getName() const {
        return "FSR 1 (EASU + RCAS)";
    }
};
