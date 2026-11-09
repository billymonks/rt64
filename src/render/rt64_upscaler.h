//
// RT64
//

#pragma once

#include "common/rt64_common.h"
#include "common/rt64_plume.h"

#include "rt64_render_worker.h"

namespace RT64 {
    struct RenderWorker;

    // Scales the composed ray traced image from the ray tracing resolution up to the display resolution.
    struct Upscaler {
        enum class QualityMode : int {
            UltraPerformance = 0,
            Performance,
            Balanced,
            Quality,
            UltraQuality,
            Native,
            Auto,
            MAX
        };

        struct UpscaleParameters {
            RectI inRect;
            RenderTexture *inDiffuseAlbedo = nullptr;
            RenderTexture *inSpecularAlbedo = nullptr;
            RenderTexture *inNormalRoughness = nullptr;
            RenderTexture *inColor = nullptr;
            RenderTexture *inFlow = nullptr;
            RenderTexture *inReactiveMask = nullptr;
            RenderTexture *inLockMask = nullptr;
            // Device depth (0 near, 1 far) at the ray tracing resolution.
            RenderTexture *inDepth = nullptr;
            // Current to previous frame, in UV units, without jitter.
            RenderTexture *inMotionVectors = nullptr;
            RenderTexture *outColor = nullptr;
            uint32_t outWidth = 0;
            uint32_t outHeight = 0;
            // Offset of the pixel center from the sample position, in pixels.
            float jitterX = 0.0f;
            float jitterY = 0.0f;
            // Milliseconds since the last frame.
            float deltaTime = 0.0f;
            float nearPlane = 0.0f;
            float farPlane = 0.0f;
            float fovY = 0.0f;
            // From 0 (none) to 1 (strongest).
            float sharpness = 0.0f;
            bool resetAccumulation = false;
        };

        virtual ~Upscaler() = default;
        virtual void set(RenderWorker *worker, QualityMode inQuality, int renderWidth, int renderHeight, int displayWidth, int displayHeight) = 0;
        virtual bool getQualityInformation(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight) = 0;
        virtual int getJitterPhaseCount(int renderWidth, int displayWidth) = 0;
        virtual void getJitterOffset(int index, int phaseCount, float &x, float &y);
        virtual void upscale(RenderWorker *worker, const UpscaleParameters &p) = 0;
        virtual bool isInitialized() const = 0;
        virtual bool requiresNonShaderResourceInputs() const = 0;
        // Temporal upscalers need jitter, motion vectors, depth and the reactive mask.
        virtual bool isTemporal() const = 0;
        virtual const char *getName() const = 0;

        static QualityMode getQualityAuto(int displayWidth, int displayHeight);
        // Display size over render size per dimension, the same ratios as AMD FSR and NVIDIA DLSS.
        static float getUpscaleRatio(QualityMode quality);
        static void getRenderSize(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight);
        static const char *getQualityName(QualityMode quality);
    };
};
