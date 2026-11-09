//
// RT64
//

#include <algorithm>
#include <cassert>
#include <cmath>

#include "rt64_upscaler.h"

// Upscaler

RT64::Upscaler::QualityMode RT64::Upscaler::getQualityAuto(int displayWidth, int displayHeight) {
    assert(displayWidth > 0);
    assert(displayHeight > 0);

    // Get the most appropriate quality level for the target resolution.
    const uint64_t PixelsDisplay = displayWidth * displayHeight;
    const uint64_t Pixels720p = 1280 * 720;
    const uint64_t Pixels1080p = 1920 * 1080;
    const uint64_t Pixels1440p = 2560 * 1440;
    const uint64_t Pixels4K = 3840 * 2160;
    if (PixelsDisplay <= Pixels720p) {
        return QualityMode::UltraQuality;
    }
    else if (PixelsDisplay <= Pixels1080p) {
        return QualityMode::Quality;
    }
    else if (PixelsDisplay <= Pixels1440p) {
        return QualityMode::Balanced;
    }
    else if (PixelsDisplay <= Pixels4K) {
        return QualityMode::Performance;
    }
    else {
        return QualityMode::UltraPerformance;
    }
}

float RT64::Upscaler::getUpscaleRatio(QualityMode quality) {
    switch (quality) {
    case QualityMode::UltraPerformance:
        return 3.0f;
    case QualityMode::Performance:
        return 2.0f;
    case QualityMode::Balanced:
        return 1.7f;
    case QualityMode::Quality:
        return 1.5f;
    case QualityMode::UltraQuality:
        return 1.3f;
    case QualityMode::Native:
    default:
        return 1.0f;
    }
}

void RT64::Upscaler::getRenderSize(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight) {
    const float ratio = getUpscaleRatio(quality);
    renderWidth = std::max(int(std::lround(float(displayWidth) / ratio)), 1);
    renderHeight = std::max(int(std::lround(float(displayHeight) / ratio)), 1);
}

const char *RT64::Upscaler::getQualityName(QualityMode quality) {
    switch (quality) {
    case QualityMode::UltraPerformance:
        return "Ultra Performance";
    case QualityMode::Performance:
        return "Performance";
    case QualityMode::Balanced:
        return "Balanced";
    case QualityMode::Quality:
        return "Quality";
    case QualityMode::UltraQuality:
        return "Ultra Quality";
    case QualityMode::Native:
        return "Native AA";
    case QualityMode::Auto:
        return "Auto";
    default:
        return "unknown";
    }
}

void RT64::Upscaler::getJitterOffset(int index, int phaseCount, float &x, float &y) {
    const hlslpp::float2 jitter = HaltonJitter(index, std::max(phaseCount, 1));
    x = float(jitter.x);
    y = float(jitter.y);
}
