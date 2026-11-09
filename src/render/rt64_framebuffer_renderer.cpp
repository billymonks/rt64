//
// RT64
//

#include "rt64_framebuffer_renderer.h"

#include "../include/rt64_extended_gbi.h"

#include "common/rt64_elapsed_timer.h"
#include "common/rt64_math.h"
#include "hle/rt64_color_converter.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_blender.h"
#include "shared/rt64_framebuffer_params.h"
#include "shared/rt64_raster_params.h"

#include "rt64_descriptor_sets.h"
#include "rt64_render_worker.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cfloat>
#include <unordered_map>

// TODO: Move to shared.

namespace interop {
    struct BicubicCB {
        uint2 InputResolution;
        uint2 OutputResolution;
    };

    struct HistogramAverageCB {
        uint pixelCount;
        float minLuminance;
        float luminanceRange;
        float timeDelta;
        float tau;
    };

    struct HistogramSetCB {
        float luminanceValue;
    };

    struct LuminanceHistogramCB {
        uint inputWidth;
        uint inputHeight;
        float minLuminance;
        float oneOverLuminanceRange;
    };

    struct TextureCB {
        uint2 TextureSize;
        float2 TexelSize;
    };
};

namespace RT64 {
    // Helper functions.
    
    RenderRect convertFixedRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, int32_t horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        if (!rect.isNull()) {
            auto computeOrigin = [=](uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return std::lround(((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage));
                }
                else {
                    return fbWidth / 2L;
                }
            };

            auto correctMisalignment = [=](int32_t coord, uint16_t origin) {
                if (origin < G_EX_ORIGIN_NONE) {
                    return int32_t(coord - (coord % std::lround(resScale[1]))) - horizontalMisalignment;
                }
                else {
                    return coord;
                }
            };

            int32_t left = static_cast<int32_t>(std::floor((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x));
            int32_t right = static_cast<int32_t>(std::ceil((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x));
            int32_t top = lround(rect.top(true) * resScale.y);
            int32_t bottom = lround(rect.bottom(true) * resScale.y);
            left = correctMisalignment(left, leftOrigin);
            right = correctMisalignment(right, rightOrigin);
            return RenderRect(left, top, right, bottom);
        }
        else {
            return RenderRect(0, 0, 0, 0);
        }
    }
    
    RenderViewport convertViewportRect(FixedRect rect, hlslpp::float2 resScale, int32_t fbWidth, float aspectRatioScale, float extOriginPercentage, float horizontalMisalignment, uint16_t leftOrigin, uint16_t rightOrigin) {
        auto computeOrigin = [=](uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return ((fbWidth * origin) / G_EX_ORIGIN_RIGHT) * extOriginPercentage + (fbWidth / 2) * (1.0f - extOriginPercentage);
            }
            else {
                return float(fbWidth) / 2;
            }
        };
        
        auto correctMisalignment = [=](float coord, uint16_t origin) {
            if (origin < G_EX_ORIGIN_NONE) {
                return (coord - std::fmod(coord, resScale[1])) - horizontalMisalignment;
            }
            else {
                return coord;
            }
        };
        
        float left = std::round((computeOrigin(leftOrigin) + (rect.left(true) - computeOrigin(leftOrigin)) * aspectRatioScale) * resScale.x);
        float right = std::round((computeOrigin(rightOrigin) + (rect.right(true) - computeOrigin(rightOrigin)) * aspectRatioScale) * resScale.x);
        float top = std::round(rect.top(true) * resScale.y);
        float bottom = std::round(rect.bottom(true) * resScale.y);
        left = correctMisalignment(left, leftOrigin);
        right = correctMisalignment(right, rightOrigin);
        return RenderViewport(left, top, right - left, bottom - top);
    }

    FixedRect fixRect(FixedRect rect, const FixedRect &scissorRect, bool fixLR) {
        // There's a very common error in many games where the fill rectangles are incorrectly configured
        // with one less coordinate than what's required to fill out the screen because other rectangle methods
        // under different modes require adding an extra coordinate. Since this behavior often breaks detection
        // for widescreen hacks, an enhancement option to fix them is available given they're within the tolerance
        // of one pixel from the scissor coordinates.
        if (fixLR) {
            if ((abs(scissorRect.lrx - rect.lrx) <= 4) && (rect.ulx < scissorRect.lrx)) {
                rect.lrx = scissorRect.lrx;
            }

            if ((abs(scissorRect.lry - rect.lry) <= 4) && (rect.uly < scissorRect.lry)) {
                rect.lry = scissorRect.lry;
            }
        }

        return rect;
    }
    
    hlslpp::float3 viewPositionFrom(hlslpp::float4x4 viewI) {
        return viewI[3].xyz;
    }

    hlslpp::float3 viewDirectionFrom(hlslpp::float4x4 viewI) {
        return hlslpp::normalize(viewI[2].xyz);
    }

    RenderColor toRenderColor(hlslpp::float4 v) {
        return { v.x, v.y, v.z, v.w };
    }

    static RenderRect viewportScissorIntersection(const RenderViewport &viewport, const RenderRect &scissor) {
        return RenderRect{
            std::max(static_cast<int32_t>(std::floor(viewport.x)), scissor.left),
            std::max(static_cast<int32_t>(std::floor(viewport.y)), scissor.top),
            std::min(static_cast<int32_t>(std::ceil(viewport.x + viewport.width)), scissor.right),
            std::min(static_cast<int32_t>(std::ceil(viewport.y + viewport.height)), scissor.bottom)
        };
    }

    // RasterScene

    static constexpr uint32_t ShadowMapSize = 2048;

    static bool scenePosition(const DrawData &drawData, uint32_t vertex, float &x, float &y, float &z, const interop::float4x4 *&transform) {
        if ((vertex * 3 + 2 >= drawData.posShorts.size()) || (vertex >= drawData.worldIndices.size())) {
            return false;
        }

        const uint16_t worldIndex = drawData.worldIndices[vertex];
        if (worldIndex >= drawData.worldTransforms.size()) {
            return false;
        }

        transform = &drawData.worldTransforms[worldIndex];
        const float px = float(drawData.posShorts[vertex * 3 + 0]);
        const float py = float(drawData.posShorts[vertex * 3 + 1]);
        const float pz = float(drawData.posShorts[vertex * 3 + 2]);
        const interop::float4x4 &world = *transform;
        x = px * world.m[0][0] + py * world.m[1][0] + pz * world.m[2][0] + world.m[3][0];
        y = px * world.m[0][1] + py * world.m[1][1] + pz * world.m[2][1] + world.m[3][1];
        z = px * world.m[0][2] + py * world.m[1][2] + pz * world.m[2][2] + world.m[3][2];
        return true;
    }

    static int quantizeRotation(float value) {
        return int(std::floor(value * 50.0f + (value >= 0.0f ? 0.5f : -0.5f)));
    }

    static bool isCameraFacing(const DrawData &drawData, uint32_t indexStart, uint32_t indexCount) {
        if ((indexCount < 3) || (indexStart + indexCount > drawData.faceIndices.size())) {
            return false;
        }

        // For a perspective camera, 1 / (1 - z) grows with the distance from the camera, so corners at the same
        // distance have the same value.
        for (uint32_t i = 0; i + 2 < indexCount; i += 3) {
            float distance[3];
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t vertex = drawData.faceIndices[indexStart + i + k];
                if (vertex >= drawData.posScreen.size()) {
                    return false;
                }
                distance[k] = 1.0f / std::max(1.0f - float(drawData.posScreen[vertex][2]), 1e-7f);
            }

            const float nearest = std::min(distance[0], std::min(distance[1], distance[2]));
            const float farthest = std::max(distance[0], std::max(distance[1], distance[2]));
            if ((farthest - nearest) > (nearest * 0.003f)) {
                return false;
            }
        }

        return true;
    }

    struct ShadowVec3 {
        float x, y, z;

        ShadowVec3 operator+(const ShadowVec3 &o) const { return { x + o.x, y + o.y, z + o.z }; }
        ShadowVec3 operator-(const ShadowVec3 &o) const { return { x - o.x, y - o.y, z - o.z }; }
        ShadowVec3 operator*(float s) const { return { x * s, y * s, z * s }; }
        float dot(const ShadowVec3 &o) const { return x * o.x + y * o.y + z * o.z; }
        ShadowVec3 cross(const ShadowVec3 &o) const { return { y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x }; }
        float length() const { return std::sqrt(dot(*this)); }
    };

    struct ShadowVertex {
        uint32_t index;
        ShadowVec3 position;
    };

    struct ShadowFitRange {
        uint32_t indexStart = 0;
        uint32_t indexCount = 0;
    };

    struct ShadowCamera {
        bool valid = false;
        ShadowVec3 position = { 0.0f, 0.0f, 0.0f };
        ShadowVec3 forward = { 0.0f, 0.0f, 1.0f };
    };

    // Finds the camera in the same space as scenePosition from the view projection used by most of the draws.
    static ShadowCamera findShadowCamera(const DrawData &drawData, const std::vector<ShadowFitRange> &ranges) {
        ShadowCamera camera;
        thread_local std::vector<std::pair<uint16_t, uint32_t>> counts;
        counts.clear();
        uint32_t sampleVertex = UINT32_MAX;
        for (const ShadowFitRange &range : ranges) {
            for (uint32_t i = 0; i < range.indexCount; i += 7) {
                const uint32_t vertex = drawData.faceIndices[range.indexStart + i];
                if ((vertex >= drawData.viewProjIndices.size()) || (vertex >= drawData.posTransformed.size())) {
                    continue;
                }
                const uint16_t viewProjIndex = drawData.viewProjIndices[vertex];
                bool found = false;
                for (auto &count : counts) {
                    if (count.first == viewProjIndex) {
                        count.second++;
                        found = true;
                    }
                }
                if (!found) {
                    counts.emplace_back(viewProjIndex, 1);
                }
                if (sampleVertex == UINT32_MAX) {
                    sampleVertex = vertex;
                }
            }
        }
        if (counts.empty()) {
            return camera;
        }

        uint16_t viewProjIndex = counts[0].first;
        uint32_t bestCount = 0;
        for (const auto &count : counts) {
            if (count.second > bestCount) {
                viewProjIndex = count.first;
                bestCount = count.second;
            }
        }
        if (viewProjIndex >= drawData.viewProjTransforms.size()) {
            return camera;
        }

        // Clip = [scene, 1] * viewProj. The camera sits where the clip x, y and w are all zero.
        const interop::float4x4 &vp = drawData.viewProjTransforms[viewProjIndex];
        const int rows[3] = { 0, 1, 3 };
        float a[3][3], b[3];
        for (int r = 0; r < 3; r++) {
            for (int i = 0; i < 3; i++) {
                a[r][i] = vp.m[i][rows[r]];
            }
            b[r] = -vp.m[3][rows[r]];
        }
        const float det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if (std::fabs(det) < 1e-12f) {
            return camera;
        }
        auto solve = [&](int column) {
            float m[3][3];
            for (int r = 0; r < 3; r++) {
                for (int i = 0; i < 3; i++) {
                    m[r][i] = (i == column) ? b[r] : a[r][i];
                }
            }
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
        };
        camera.position = { solve(0), solve(1), solve(2) };
        camera.forward = { vp.m[0][3], vp.m[1][3], vp.m[2][3] };
        const float forwardLength = camera.forward.length();
        if (forwardLength < 1e-8f) {
            return camera;
        }
        camera.forward = camera.forward * (1.0f / forwardLength);

        // Only trust it if it reproduces what the RSP computed for a real vertex.
        const interop::float4x4 *transform = nullptr;
        ShadowVec3 p;
        if ((sampleVertex == UINT32_MAX) || !scenePosition(drawData, sampleVertex, p.x, p.y, p.z, transform)) {
            return camera;
        }
        const float w = p.x * vp.m[0][3] + p.y * vp.m[1][3] + p.z * vp.m[2][3] + vp.m[3][3];
        const float expectedW = drawData.posTransformed[sampleVertex][3];
        camera.valid = std::fabs(w - expectedW) <= (std::fabs(expectedW) * 0.01f + 0.01f);
        return camera;
    }

    // Splits a draw into pieces of triangles connected by shared vertices or shared positions, since a single
    // draw can hold several separate sprites.
    static void splitShadowComponents(const DrawData &drawData, uint32_t indexStart, uint32_t indexCount, std::vector<uint32_t> &triangleOrder, std::vector<uint32_t> &componentStarts) {
        const uint32_t triangleCount = indexCount / 3;
        thread_local std::vector<uint32_t> parent;
        thread_local std::unordered_map<uint32_t, uint32_t> byVertex;
        thread_local std::unordered_map<uint64_t, uint32_t> byPosition;
        parent.resize(triangleCount);
        byVertex.clear();
        byPosition.clear();
        for (uint32_t t = 0; t < triangleCount; t++) {
            parent[t] = t;
        }

        auto find = [&](uint32_t t) {
            while (parent[t] != t) {
                parent[t] = parent[parent[t]];
                t = parent[t];
            }
            return t;
        };
        auto unite = [&](uint32_t a, uint32_t b) {
            a = find(a);
            b = find(b);
            if (a != b) {
                parent[std::max(a, b)] = std::min(a, b);
            }
        };

        for (uint32_t t = 0; t < triangleCount; t++) {
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t vertex = drawData.faceIndices[indexStart + t * 3 + k];
                auto vertexIt = byVertex.emplace(vertex, t);
                if (!vertexIt.second) {
                    unite(vertexIt.first->second, t);
                }

                const interop::float4x4 *transform = nullptr;
                ShadowVec3 p;
                if (scenePosition(drawData, vertex, p.x, p.y, p.z, transform)) {
                    const uint64_t qx = uint64_t(int64_t(std::lround(p.x * 4.0f)) & 0x1FFFFF);
                    const uint64_t qy = uint64_t(int64_t(std::lround(p.y * 4.0f)) & 0x1FFFFF);
                    const uint64_t qz = uint64_t(int64_t(std::lround(p.z * 4.0f)) & 0x1FFFFF);
                    auto positionIt = byPosition.emplace((qx << 42) | (qy << 21) | qz, t);
                    if (!positionIt.second) {
                        unite(positionIt.first->second, t);
                    }
                }
            }
        }

        triangleOrder.resize(triangleCount);
        for (uint32_t t = 0; t < triangleCount; t++) {
            triangleOrder[t] = t;
        }
        for (uint32_t t = 0; t < triangleCount; t++) {
            parent[t] = find(t);
        }
        std::stable_sort(triangleOrder.begin(), triangleOrder.end(), [&](uint32_t l, uint32_t r) { return parent[l] < parent[r]; });

        componentStarts.clear();
        for (uint32_t i = 0; i < triangleCount; i++) {
            if ((i == 0) || (parent[triangleOrder[i]] != parent[triangleOrder[i - 1]])) {
                componentStarts.push_back(i);
            }
        }
        componentStarts.push_back(triangleCount);
    }

    // Detects a small flat sprite that turns to face the camera, either around the vertical axis or fully. When
    // it is one, its vertices are rebuilt turned around the vertical axis to face the light instead, so its
    // shadow keeps the same shape from every camera angle.
    static bool reorientBillboard(const DrawData &drawData, uint32_t indexStart, const uint32_t *triangles, uint32_t triangleCount, const ShadowCamera &camera, const interop::float3 &worldUp, const interop::float3 &lightDirection, std::vector<ShadowVertex> &vertices, bool *uprightOut = nullptr) {
        vertices.clear();
        if ((triangleCount == 0) || (triangleCount > 8)) {
            return false;
        }

        thread_local std::vector<float> distances;
        distances.clear();
        const uint32_t indexCount = triangleCount * 3;
        auto faceVertex = [&](uint32_t i) {
            return drawData.faceIndices[indexStart + triangles[i / 3] * 3 + (i % 3)];
        };
        for (uint32_t i = 0; i < indexCount; i++) {
            const uint32_t vertex = faceVertex(i);
            bool known = false;
            for (const ShadowVertex &existing : vertices) {
                known = known || (existing.index == vertex);
            }
            if (known) {
                continue;
            }

            const interop::float4x4 *transform = nullptr;
            ShadowVertex shadowVertex;
            shadowVertex.index = vertex;
            if ((vertex >= drawData.posTransformed.size()) || !scenePosition(drawData, vertex, shadowVertex.position.x, shadowVertex.position.y, shadowVertex.position.z, transform)) {
                return false;
            }

            vertices.push_back(shadowVertex);
            distances.push_back(float(drawData.posTransformed[vertex][3]));
        }

        // Use the largest triangle for the plane so slivers don't decide it.
        ShadowVec3 normal = { 0.0f, 0.0f, 0.0f };
        float normalLength = 0.0f;
        for (uint32_t i = 0; i + 2 < indexCount; i += 3) {
            ShadowVec3 p[3];
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t vertex = faceVertex(i + k);
                for (const ShadowVertex &shadowVertex : vertices) {
                    if (shadowVertex.index == vertex) {
                        p[k] = shadowVertex.position;
                    }
                }
            }

            const ShadowVec3 faceNormal = (p[1] - p[0]).cross(p[2] - p[0]);
            const float faceLength = faceNormal.length();
            if (faceLength > normalLength) {
                normal = faceNormal;
                normalLength = faceLength;
            }
        }
        if (normalLength < 1e-6f) {
            return false;
        }
        normal = normal * (1.0f / normalLength);

        ShadowVec3 center = { 0.0f, 0.0f, 0.0f };
        for (const ShadowVertex &shadowVertex : vertices) {
            center = center + shadowVertex.position;
        }
        center = center * (1.0f / float(vertices.size()));

        float extent = 0.0f;
        for (const ShadowVertex &shadowVertex : vertices) {
            extent = std::max(extent, (shadowVertex.position - center).length());
        }
        for (const ShadowVertex &shadowVertex : vertices) {
            if (std::fabs((shadowVertex.position - center).dot(normal)) > (extent * 0.02f)) {
                return false;
            }
        }

        const ShadowVec3 up = { worldUp.x, worldUp.y, worldUp.z };
        ShadowVec3 across = up.cross(normal);
        const float acrossLength = across.length();
        if (acrossLength < 0.3f) {
            return false;
        }
        across = across * (1.0f / acrossLength);

        ShadowVec3 along = normal.cross(across);
        if (along.dot(up) < 0.0f) {
            along = along * -1.0f;
        }

        // Upright sprites must face the camera horizontally, anything tilted must face it fully. Sloped level
        // geometry would otherwise get stood up whenever the camera happens to line up with it.
        const bool upright = std::fabs(normal.dot(up)) < 0.2f;
        if (uprightOut != nullptr) {
            *uprightOut = upright;
        }

        if (camera.valid) {
            const float FacingCos = 0.985f;
            ShadowVec3 toCamera = camera.position - center;
            ShadowVec3 viewAxis = camera.forward;
            ShadowVec3 facing = normal;
            if (upright) {
                toCamera = toCamera - up * toCamera.dot(up);
                viewAxis = viewAxis - up * viewAxis.dot(up);
                facing = facing - up * facing.dot(up);
            }

            const float toCameraLength = toCamera.length();
            const float viewAxisLength = viewAxis.length();
            const float facingLength = facing.length();
            if (facingLength < 1e-4f) {
                return false;
            }
            const bool facesPosition = (toCameraLength > 1e-4f) && (std::fabs(toCamera.dot(facing)) >= (FacingCos * toCameraLength * facingLength));
            const bool facesPlane = (viewAxisLength > 1e-4f) && (std::fabs(viewAxis.dot(facing)) >= (FacingCos * viewAxisLength * facingLength));
            if (!facesPosition && !facesPlane) {
                return false;
            }
        }
        else if (!upright) {
            return false;
        }

        // Without a camera, an upright sprite faces it horizontally when the distance doesn't change across its width.
        float meanA = 0.0f, meanB = 0.0f, meanD = 0.0f, minD = FLT_MAX, minA = FLT_MAX, maxA = -FLT_MAX;
        thread_local std::vector<float> coordsA;
        thread_local std::vector<float> coordsB;
        coordsA.clear();
        coordsB.clear();
        for (uint32_t i = 0; i < vertices.size(); i++) {
            const ShadowVec3 offset = vertices[i].position - center;
            coordsA.push_back(offset.dot(across));
            coordsB.push_back(offset.dot(along));
            meanA += coordsA.back();
            meanB += coordsB.back();
            meanD += distances[i];
            minD = std::min(minD, distances[i]);
            minA = std::min(minA, coordsA.back());
            maxA = std::max(maxA, coordsA.back());
        }
        const float invCount = 1.0f / float(vertices.size());
        meanA *= invCount;
        meanB *= invCount;
        meanD *= invCount;

        float sAA = 0.0f, sAB = 0.0f, sBB = 0.0f, sAD = 0.0f, sBD = 0.0f;
        for (uint32_t i = 0; i < vertices.size(); i++) {
            const float a = coordsA[i] - meanA;
            const float b = coordsB[i] - meanB;
            const float d = distances[i] - meanD;
            sAA += a * a;
            sAB += a * b;
            sBB += b * b;
            sAD += a * d;
            sBD += b * d;
        }
        if (!camera.valid) {
            const float determinant = sAA * sBB - sAB * sAB;
            if (determinant <= (sAA * sBB * 1e-4f)) {
                return false;
            }
            const float gradientA = (sAD * sBB - sBD * sAB) / determinant;
            if ((std::fabs(gradientA) * (maxA - minA)) > (std::fabs(minD) * 0.004f)) {
                return false;
            }
        }

        ShadowVec3 lightAcross = up.cross({ lightDirection.x, lightDirection.y, lightDirection.z });
        const float lightAcrossLength = lightAcross.length();
        if (lightAcrossLength < 1e-3f) {
            lightAcross = across;
        }
        else {
            lightAcross = lightAcross * (1.0f / lightAcrossLength);
            if (lightAcross.dot(across) < 0.0f) {
                lightAcross = lightAcross * -1.0f;
            }
        }

        for (uint32_t i = 0; i < vertices.size(); i++) {
            vertices[i].position = center + lightAcross * coordsA[i] + up * coordsB[i];
        }

        return true;
    }

    // Camera position and forward axis from a view projection in row-vector form, like findShadowCamera.
    static ShadowCamera cameraFromViewProj(const interop::float4x4 &vp) {
        ShadowCamera camera;
        const int rows[3] = { 0, 1, 3 };
        float a[3][3], b[3];
        for (int r = 0; r < 3; r++) {
            for (int i = 0; i < 3; i++) {
                a[r][i] = vp.m[i][rows[r]];
            }
            b[r] = -vp.m[3][rows[r]];
        }
        const float det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if (std::fabs(det) < 1e-12f) {
            return camera;
        }
        auto solve = [&](int column) {
            float m[3][3];
            for (int r = 0; r < 3; r++) {
                for (int i = 0; i < 3; i++) {
                    m[r][i] = (i == column) ? b[r] : a[r][i];
                }
            }
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) / det;
        };
        camera.position = { solve(0), solve(1), solve(2) };
        camera.forward = { vp.m[0][3], vp.m[1][3], vp.m[2][3] };
        const float forwardLength = camera.forward.length();
        if (forwardLength < 1e-8f) {
            return camera;
        }
        camera.forward = camera.forward * (1.0f / forwardLength);
        camera.valid = true;
        return camera;
    }

    // Shadow caster copy of a cutout draw for the ray tracer: every piece that reorientBillboard recognizes as a
    // camera facing sprite is stood up facing the light; other pieces keep their positions. The copy has one vertex
    // per index in the draw's own triangle order, so the shadow any-hit shader finds the same texels by primitive
    // index and barycentrics. Returns false when the draw has no billboard pieces.
    static bool buildBillboardProxy(const DrawData &drawData, uint32_t indexStart, uint32_t indexCount, const ShadowCamera &camera, const interop::float3 &lightDirection,
        std::vector<float> &positions, uint32_t &cylindricalCount, uint32_t &sphericalCount)
    {
        const uint32_t MaxProxyDrawTriangles = 512;
        const uint32_t triangleCount = indexCount / 3;
        if ((triangleCount == 0) || (triangleCount > MaxProxyDrawTriangles) || (indexStart + indexCount > drawData.faceIndices.size())) {
            return false;
        }

        thread_local std::vector<ShadowVec3> corners;
        thread_local std::vector<uint32_t> componentTriangles;
        thread_local std::vector<uint32_t> componentStarts;
        thread_local std::vector<ShadowVertex> billboardVertices;
        corners.resize(indexCount);
        for (uint32_t i = 0; i < indexCount; i++) {
            const interop::float4x4 *transform = nullptr;
            ShadowVec3 &p = corners[i];
            if (!scenePosition(drawData, drawData.faceIndices[indexStart + i], p.x, p.y, p.z, transform)) {
                return false;
            }
        }

        const interop::float3 worldUp = { 0.0f, 1.0f, 0.0f };
        splitShadowComponents(drawData, indexStart, indexCount, componentTriangles, componentStarts);
        bool anyBillboard = false;
        for (size_t c = 0; c + 1 < componentStarts.size(); c++) {
            const uint32_t *triangles = &componentTriangles[componentStarts[c]];
            const uint32_t count = componentStarts[c + 1] - componentStarts[c];
            bool upright = false;
            if (!reorientBillboard(drawData, indexStart, triangles, count, camera, worldUp, lightDirection, billboardVertices, &upright)) {
                continue;
            }

            anyBillboard = true;
            if (upright) {
                cylindricalCount++;
            }
            else {
                sphericalCount++;
            }

            for (uint32_t t = 0; t < count; t++) {
                for (uint32_t k = 0; k < 3; k++) {
                    const uint32_t slot = triangles[t] * 3 + k;
                    const uint32_t vertex = drawData.faceIndices[indexStart + slot];
                    for (const ShadowVertex &shadowVertex : billboardVertices) {
                        if (shadowVertex.index == vertex) {
                            corners[slot] = shadowVertex.position;
                            break;
                        }
                    }
                }
            }
        }

        if (!anyBillboard) {
            return false;
        }

        positions.resize(size_t(indexCount) * 4);
        for (uint32_t i = 0; i < indexCount; i++) {
            positions[i * 4 + 0] = corners[i].x;
            positions[i * 4 + 1] = corners[i].y;
            positions[i * 4 + 2] = corners[i].z;
            positions[i * 4 + 3] = 1.0f;
        }

        return true;
    }

    static bool buildDirectionalShadow(const DrawData &drawData, float nativeWidth, float nativeHeight, const std::vector<ShadowFitRange> &ranges, const float localDirection[3], interop::float4x4 &lightMatrix, interop::float3 &lightDirection, interop::float3 &worldUp) {
        struct Point {
            float x, y, z, w;
        };
        struct Bucket {
            int quantized[9] = {};
            uint32_t weight = 0;
            float m[3][3] = {};
        };
        thread_local std::vector<Point> visible;
        thread_local std::vector<Bucket> buckets;
        thread_local std::vector<float> distances;
        visible.clear();
        buckets.clear();

        const float marginX = nativeWidth * 0.4f;
        const float marginY = nativeHeight * 0.4f;
        for (const ShadowFitRange &range : ranges) {
            if (range.indexStart + range.indexCount > drawData.faceIndices.size()) {
                continue;
            }
            for (uint32_t i = 0; i < range.indexCount; i++) {
                const uint32_t vertex = drawData.faceIndices[range.indexStart + i];
                const interop::float4x4 *transform = nullptr;
                Point point;
                if (!scenePosition(drawData, vertex, point.x, point.y, point.z, transform)) {
                    continue;
                }
                point.w = (vertex < drawData.posTransformed.size()) ? std::max(float(drawData.posTransformed[vertex][3]), 0.0f) : 0.0f;

                const int quantized[9] = {
                    quantizeRotation(transform->m[0][0]), quantizeRotation(transform->m[1][0]), quantizeRotation(transform->m[2][0]),
                    quantizeRotation(transform->m[0][1]), quantizeRotation(transform->m[1][1]), quantizeRotation(transform->m[2][1]),
                    quantizeRotation(transform->m[0][2]), quantizeRotation(transform->m[1][2]), quantizeRotation(transform->m[2][2])
                };
                int bucketIndex = -1;
                for (uint32_t b = 0; b < buckets.size(); b++) {
                    bool same = true;
                    for (int k = 0; k < 9; k++) {
                        same = same && (buckets[b].quantized[k] == quantized[k]);
                    }
                    if (same) {
                        bucketIndex = int(b);
                        break;
                    }
                }
                if (bucketIndex < 0) {
                    Bucket bucket;
                    for (int k = 0; k < 9; k++) {
                        bucket.quantized[k] = quantized[k];
                    }
                    bucket.m[0][0] = transform->m[0][0]; bucket.m[1][0] = transform->m[1][0]; bucket.m[2][0] = transform->m[2][0];
                    bucket.m[0][1] = transform->m[0][1]; bucket.m[1][1] = transform->m[1][1]; bucket.m[2][1] = transform->m[2][1];
                    bucket.m[0][2] = transform->m[0][2]; bucket.m[1][2] = transform->m[1][2]; bucket.m[2][2] = transform->m[2][2];
                    buckets.push_back(bucket);
                    bucketIndex = int(buckets.size() - 1);
                }
                buckets[bucketIndex].weight++;

                bool onScreen = true;
                if (vertex < drawData.posScreen.size()) {
                    const float sx = drawData.posScreen[vertex][0];
                    const float sy = drawData.posScreen[vertex][1];
                    const float sz = drawData.posScreen[vertex][2];
                    onScreen = (sx >= -marginX) && (sx <= nativeWidth + marginX) && (sy >= -marginY) && (sy <= nativeHeight + marginY) && (sz > -0.05f) && (sz < 1.02f);
                }
                if (onScreen) {
                    visible.push_back(point);
                }
            }
        }

        if ((visible.size() < 3) || buckets.empty()) {
            return false;
        }

        const Bucket *best = &buckets[0];
        for (const Bucket &bucket : buckets) {
            if (bucket.weight > best->weight) {
                best = &bucket;
            }
        }
        const float localX = localDirection[0];
        const float localY = localDirection[1];
        const float localZ = localDirection[2];
        float dirX = localX * best->m[0][0] + localY * best->m[1][0] + localZ * best->m[2][0];
        float dirY = localX * best->m[0][1] + localY * best->m[1][1] + localZ * best->m[2][1];
        float dirZ = localX * best->m[0][2] + localY * best->m[1][2] + localZ * best->m[2][2];
        float dirLength = std::sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
        if (dirLength < 1e-4f) {
            dirX = localX;
            dirY = localY;
            dirZ = localZ;
            dirLength = std::sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
            if (dirLength < 1e-4f) {
                return false;
            }
        }
        dirX /= dirLength;
        dirY /= dirLength;
        dirZ /= dirLength;

        // Fit to everything up to a distance from the camera, so nearby geometry is never dropped because
        // far away terrain pulls the fit elsewhere.
        distances.clear();
        distances.reserve(visible.size());
        for (const Point &point : visible) {
            distances.push_back(point.w);
        }
        std::sort(distances.begin(), distances.end());
        const float keepDistance = std::max(distances[distances.size() / 2] * 2.5f, 64.0f);

        float centerX = 0.0f;
        float centerY = 0.0f;
        float centerZ = 0.0f;
        uint32_t centerCount = 0;
        for (const Point &point : visible) {
            if (point.w > keepDistance) {
                continue;
            }
            centerX += point.x;
            centerY += point.y;
            centerZ += point.z;
            centerCount++;
        }
        if (centerCount < 3) {
            return false;
        }
        const float invCount = 1.0f / float(centerCount);
        centerX *= invCount;
        centerY *= invCount;
        centerZ *= invCount;

        const float axisX = (std::fabs(dirY) > 0.85f) ? 1.0f : 0.0f;
        const float axisY = (std::fabs(dirY) > 0.85f) ? 0.0f : 1.0f;
        const float axisZ = 0.0f;
        const float forwardX = -dirX;
        const float forwardY = -dirY;
        const float forwardZ = -dirZ;
        float rightX = axisY * forwardZ - axisZ * forwardY;
        float rightY = axisZ * forwardX - axisX * forwardZ;
        float rightZ = axisX * forwardY - axisY * forwardX;
        const float rightLength = std::sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
        if (rightLength < 1e-4f) {
            return false;
        }
        rightX /= rightLength;
        rightY /= rightLength;
        rightZ /= rightLength;
        const float upX = forwardY * rightZ - forwardZ * rightY;
        const float upY = forwardZ * rightX - forwardX * rightZ;
        const float upZ = forwardX * rightY - forwardY * rightX;

        float minLX = FLT_MAX, maxLX = -FLT_MAX;
        float minLY = FLT_MAX, maxLY = -FLT_MAX;
        float minLZ = FLT_MAX, maxLZ = -FLT_MAX;
        uint32_t kept = 0;
        for (const Point &point : visible) {
            const float ox = point.x - centerX;
            const float oy = point.y - centerY;
            const float oz = point.z - centerZ;
            if (point.w > keepDistance) {
                continue;
            }
            const float lightX = ox * rightX + oy * rightY + oz * rightZ;
            const float lightY = ox * upX + oy * upY + oz * upZ;
            const float lightZ = ox * forwardX + oy * forwardY + oz * forwardZ;
            minLX = std::min(minLX, lightX);
            maxLX = std::max(maxLX, lightX);
            minLY = std::min(minLY, lightY);
            maxLY = std::max(maxLY, lightY);
            minLZ = std::min(minLZ, lightZ);
            maxLZ = std::max(maxLZ, lightZ);
            kept++;
        }
        if (kept < 3) {
            return false;
        }

        const float padX = std::max((maxLX - minLX) * 0.12f, 8.0f);
        const float padY = std::max((maxLY - minLY) * 0.12f, 8.0f);
        const float padZ = std::max((maxLZ - minLZ) * 0.15f, 8.0f);
        minLX -= padX;
        maxLX += padX;
        minLY -= padY;
        maxLY += padY;
        minLZ -= padZ;
        maxLZ += padZ;
        if (maxLX - minLX < 1.0f) {
            minLX -= 0.5f;
            maxLX += 0.5f;
        }
        if (maxLY - minLY < 1.0f) {
            minLY -= 0.5f;
            maxLY += 0.5f;
        }
        if (maxLZ - minLZ < 1.0f) {
            minLZ -= 0.5f;
            maxLZ += 0.5f;
        }

        const float shadowSize = float(ShadowMapSize);
        const float texelX = (maxLX - minLX) / shadowSize;
        const float texelY = (maxLY - minLY) / shadowSize;
        minLX = std::floor(minLX / texelX) * texelX;
        maxLX = std::ceil(maxLX / texelX) * texelX;
        minLY = std::floor(minLY / texelY) * texelY;
        maxLY = std::ceil(maxLY / texelY) * texelY;

        const float view[4][4] = {
            { rightX, rightY, rightZ, -(centerX * rightX + centerY * rightY + centerZ * rightZ) },
            { upX, upY, upZ, -(centerX * upX + centerY * upY + centerZ * upZ) },
            { forwardX, forwardY, forwardZ, -(centerX * forwardX + centerY * forwardY + centerZ * forwardZ) },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        };
        const float left = minLX;
        const float right = maxLX;
        const float bottom = minLY;
        const float top = maxLY;
        const float nearPlane = minLZ;
        const float farPlane = maxLZ;
        const float projection[4][4] = {
            { 2.0f / (right - left), 0.0f, 0.0f, -(right + left) / (right - left) },
            { 0.0f, 2.0f / (top - bottom), 0.0f, -(top + bottom) / (top - bottom) },
            { 0.0f, 0.0f, 1.0f / (farPlane - nearPlane), -nearPlane / (farPlane - nearPlane) },
            { 0.0f, 0.0f, 0.0f, 1.0f }
        };

        float combined[4][4];
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                combined[row][col] = projection[row][0] * view[0][col] + projection[row][1] * view[1][col] + projection[row][2] * view[2][col] + projection[row][3] * view[3][col];
            }
        }
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                lightMatrix.m[col][row] = combined[row][col];
            }
        }
        lightDirection = interop::float3(dirX, dirY, dirZ);

        const float upLength = std::sqrt(best->m[1][0] * best->m[1][0] + best->m[1][1] * best->m[1][1] + best->m[1][2] * best->m[1][2]);
        if (upLength > 1e-4f) {
            worldUp = interop::float3(best->m[1][0] / upLength, best->m[1][1] / upLength, best->m[1][2] / upLength);
        }
        else {
            worldUp = interop::float3(0.0f, 1.0f, 0.0f);
        }

        return true;
    }


    RasterScene::RasterScene() { }

    // FramebufferRenderer
    
    FramebufferRenderer::FramebufferRenderer(RenderWorker *worker, bool rtSupport, UserConfiguration::GraphicsAPI graphicsAPI, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);

        this->shaderLibrary = shaderLibrary;

        frameParams.frameCount = 0;
        frameParams.viewUbershaders = false;
        frameParams.ditherNoiseStrength = 1.0f;

        shaderUploader = std::make_unique<BufferUploader>(worker->device);
        descCommonSet = std::make_unique<FramebufferRendererDescriptorCommonSet>(shaderLibrary->samplerLibrary, worker->device->getCapabilities().raytracing, worker->device);

#   if RT_ENABLED
        if (rtSupport) {
            this->rtSupport = rtSupport;
            rtResources = std::make_unique<RaytracingResources>(worker, graphicsAPI);
    }
#   endif
    }

    FramebufferRenderer::~FramebufferRenderer() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
    }
    
    void FramebufferRenderer::resetFramebuffers(RenderWorker *worker, bool ubershadersVisible, float ditherNoiseStrength, const RenderMultisampling &multisampling) {
        instanceDrawCallVector.clear();
        hitGroupVector.clear();
        renderIndicesVector.clear();
        rspSmoothNormalVector.clear();
        frameParams.viewUbershaders = ubershadersVisible;
        frameParams.ditherNoiseStrength = ditherNoiseStrength;
        framebufferCount = 0;
        shadowCutoutChunkCursor = 0;
        shadowMapSetCursor = 0;

        // Create dummy color target if it hasn't been created yet.
        if (dummyColorTarget == nullptr) {
            RenderTextureDesc dummyColorDesc = RenderTextureDesc::ColorTarget(4, 4, RenderTarget::colorBufferFormat(shaderLibrary->usesHDR), multisampling);
            dummyColorTarget = worker->device->createTexture(dummyColorDesc);
            dummyColorTarget->setName("Framebuffer Renderer Color Dummy");
            dummyColorTargetView = dummyColorTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyColorDesc.format));
            dummyColorTargetTransitioned = false;
        }

        // Create dummy depth target if it hasn't been created yet.
        if (dummyDepthTarget == nullptr) {
            RenderTextureDesc dummyDepthDesc = RenderTextureDesc::DepthTarget(4, 4, RenderFormat::D32_FLOAT, multisampling);
            dummyDepthTarget = worker->device->createTexture(dummyDepthDesc);
            dummyDepthTarget->setName("Framebuffer Renderer Depth Dummy");
            dummyDepthTargetView = dummyDepthTarget->createTextureView(RenderTextureViewDesc::Texture2D(dummyDepthDesc.format));
            dummyDepthTargetTransitioned = false;
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::resetRaytracing(RaytracingShaderCache *rtShaderCache, const RenderTexture *blueNoiseTexture) {
        assert(rtResources != nullptr);
        assert(rtShaderCache != nullptr);
        assert(blueNoiseTexture != nullptr);

        const int32_t stateIndex = rtShaderCache->getActiveState();
        rtState = &rtShaderCache->states[stateIndex];
        rtPipelineLayout = rtShaderCache->pipelineLayout.get();
        rtResources->resetBottomLevelAS();

        this->blueNoiseTexture = blueNoiseTexture;
    }
#endif

    void FramebufferRenderer::updateTextureCache(TextureCache *textureCache) {
        const std::unique_lock<std::mutex> textureMapLock(textureCache->textureMapMutex);
        textureCacheVersions = textureCache->textureMap.versions;
        textureCacheTextures = textureCache->textureMap.textures;
        textureCacheTextureReplacements = textureCache->textureMap.textureReplacements;
        textureCacheFreeSpaces = textureCache->textureMap.freeSpaces;
        textureCacheSize = static_cast<uint32_t>(textureCacheTextures.size());
        textureCacheGlobalVersion = textureCache->textureMap.globalVersion;
        textureCacheReplacementMapEnabled = textureCache->textureMap.replacementMapEnabled;
        dynamicTextureViewVector.clear();
        dynamicTextureBarrierVector.clear();
    }

    void FramebufferRenderer::createGPUTiles(const DrawCallTile *callTiles, uint32_t callTileCount, interop::GPUTile *dstGPUTiles, const FramebufferManager *fbManager, 
        TextureCache *textureCache, uint64_t submissionFrame)
    {
        for (uint32_t i = 0; i < callTileCount; i++) {
            const DrawCallTile &callTile = callTiles[i];
            if (!callTile.valid) {
                continue;
            }
            
            interop::GPUTile &gpuTile = dstGPUTiles[i];
            if (callTile.tileCopyUsed) {
                const auto &it = fbManager->tileCopies.find(callTile.tmemHashOrID);
                if (it != fbManager->tileCopies.end()) {
                    const FramebufferManager::TileCopy &tileCopy = it->second;
                    gpuTile.tcScale.x = static_cast<float>(tileCopy.usedWidth) / static_cast<float>(callTile.tileCopyWidth);
                    gpuTile.tcScale.y = static_cast<float>(tileCopy.usedHeight) / static_cast<float>(callTile.tileCopyHeight);
                    gpuTile.ulScale.x = tileCopy.ulScaleS ? gpuTile.tcScale.x : 1.0f;
                    gpuTile.ulScale.y = tileCopy.ulScaleT ? gpuTile.tcScale.y : 1.0f;
                    gpuTile.texelShift = tileCopy.texelShift;
                    gpuTile.texelMask = tileCopy.texelMask;
                    gpuTile.textureIndex = getTextureIndex(tileCopy);
                    gpuTile.textureDimensions = interop::float3(float(tileCopy.textureWidth), float(tileCopy.textureHeight), 1.0f);
                    gpuTile.flags.alphaIsCvg = !callTile.reinterpretTile;
                    gpuTile.flags.highRes = true;
                    gpuTile.flags.fromCopy = true;
                    gpuTile.flags.rawTMEM = false;
                    gpuTile.flags.hasMipmaps = false;
                }
            }
            else {
                // Retrieve the texture from the cache or use a blank texture if not found.
                uint32_t textureIndex = 0;
                bool textureReplaced = false;
                bool hasMipmaps = false;
                bool shiftedByHalf = false;
                textureCache->useTexture(callTile.tmemHashOrID, submissionFrame, textureIndex, gpuTile.tcScale, gpuTile.textureDimensions, textureReplaced, hasMipmaps, shiftedByHalf);

                // Describe the GPU tile for a regular texture.
                gpuTile.ulScale.x = gpuTile.tcScale.x;
                gpuTile.ulScale.y = gpuTile.tcScale.y;
                gpuTile.texelShift = { 0, 0 };
                gpuTile.texelMask = { UINT_MAX, UINT_MAX };
                gpuTile.textureIndex = textureIndex;
                gpuTile.flags.alphaIsCvg = false;
                gpuTile.flags.highRes = textureReplaced;
                gpuTile.flags.fromCopy = false;
                gpuTile.flags.rawTMEM = !textureReplaced && callTile.rawTMEM;
                gpuTile.flags.hasMipmaps = hasMipmaps;
                gpuTile.flags.shiftedByHalf = shiftedByHalf;
            }
        }
    }

    uint32_t FramebufferRenderer::getDestinationIndex() {
        uint32_t dstIndex;
        if (textureCacheFreeSpaces.empty()) {
            dstIndex = textureCacheSize++;
        }
        else {
            dstIndex = textureCacheFreeSpaces.back();
            textureCacheFreeSpaces.pop_back();
        }

        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(RenderTarget *renderTarget) {
        assert(renderTarget != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ renderTarget->getResolvedTexture(), dstIndex, renderTarget->getResolvedTextureView()});
        return dstIndex;
    }
    
    uint32_t FramebufferRenderer::getTextureIndex(const FramebufferManager::TileCopy &tileCopy) {
        assert(tileCopy.texture != nullptr);

        uint32_t dstIndex = getDestinationIndex();
        dynamicTextureViewVector.emplace_back(DynamicTextureView{ tileCopy.texture.get(), dstIndex, nullptr });
        dynamicTextureBarrierVector.emplace_back(RenderTextureBarrier(tileCopy.texture.get(), RenderTextureLayout::SHADER_READ));
        return dstIndex;
    }
    
    void FramebufferRenderer::updateShaderDescriptorSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        assert(worker != nullptr);
        assert(drawBuffers != nullptr);
        
        const bool createSet = (descTextureSet == nullptr) || (descTextureSet->textureCacheSize < (textureCacheSize + 1));
        if (createSet) {
            descTextureSet = std::make_unique<FramebufferRendererDescriptorTextureSet>(worker->device, ((textureCacheSize + 1) * 3) / 2);
        }

        if (createSet || (descriptorTextureReplacementMapEnabled != textureCacheReplacementMapEnabled)) {
            descriptorTextureVersions.clear();
            descriptorTextureGlobalVersion = 0;
            descriptorTextureReplacementMapEnabled = textureCacheReplacementMapEnabled;
        }

#   if RT_ENABLED
        if (raytracingEnabled) {
            descCommonSet->setBuffer(descCommonSet->posBuffer, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->normBuffer, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->velBuffer, outputBuffers->worldVelBuffer.buffer.get(), outputBuffers->worldVelBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->genTexCoordBuffer, outputBuffers->genTexCoordBuffer.buffer.get(), outputBuffers->genTexCoordBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->shadedColBuffer, outputBuffers->shadedColBuffer.buffer.get(), outputBuffers->shadedColBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcFogIndices, drawBuffers->fogIndicesBuffer.get(), drawBuffers->fogIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightIndices, drawBuffers->lightIndicesBuffer.get(), drawBuffers->lightIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->srcLightCounts, drawBuffers->lightCountsBuffer.get(), drawBuffers->lightCountsBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->indexBuffer, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize);
            descCommonSet->setBuffer(descCommonSet->RSPFogVector, drawBuffers->rspFogBuffer.get(), drawBuffers->rspFogBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPFog)));
            descCommonSet->setBuffer(descCommonSet->RSPLightVector, drawBuffers->rspLightsBuffer.get(), drawBuffers->rspLightsBuffer.allocatedSize, RenderBufferStructuredView(sizeof(interop::RSPLight)));
            descCommonSet->setTexture(descCommonSet->gViewDirection, rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingPosition, rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingNormal, rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gShadingSpecular, rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDiffuse, rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gInstanceId, rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReflection, rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gRefraction, rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gTransparent, rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFlow, rtResources->flowTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gReactiveMask, rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gLockMask, rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gDepth, rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevNormalRoughness, rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDepth, rtResources->depthTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevDirectLightAccum, rtResources->directLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gPrevIndirectLightAccum, rtResources->indirectLightTexture[rtResources->swapBuffers ? 0 : 1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredDirectLight, rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL);
            descCommonSet->setTexture(descCommonSet->gFilteredIndirectLight, rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL);

            const uint32_t hitBufferBytes = 256;
            descCommonSet->setBuffer(descCommonSet->gHitVelocityDistance, rtResources->hitVelocityDistanceBuffer.get(), hitBufferBytes, rtResources->hitVelocityDistanceBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitColor, rtResources->hitColorBuffer.get(), hitBufferBytes, rtResources->hitColorBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitNormalFog, rtResources->hitNormalFogBuffer.get(), hitBufferBytes, rtResources->hitNormalFogBufferView.get());
            descCommonSet->setBuffer(descCommonSet->gHitInstanceId, rtResources->hitInstanceIdBuffer.get(), hitBufferBytes, rtResources->hitInstanceIdBufferView.get());
            descCommonSet->setAccelerationStructure(descCommonSet->SceneBVH, rtResources->topLevelAS.get());
            descCommonSet->setBuffer(descCommonSet->SceneLights, rtResources->lightsBuffer.get(), sizeof(interop::PointLight) * std::max(rtResources->rtParams.lightsCount, 1U), RenderBufferStructuredView(sizeof(interop::PointLight)));
            descCommonSet->setBuffer(descCommonSet->interleavedRasters, interleavedRastersBuffer.get(), sizeof(interop::InterleavedRaster) * std::max(interleavedRastersCount, 1U), RenderBufferStructuredView(sizeof(interop::InterleavedRaster)));
            descCommonSet->setTexture(descCommonSet->gBlueNoise, blueNoiseTexture, RenderTextureLayout::SHADER_READ);
            descCommonSet->setBuffer(descCommonSet->instanceExtraParams, drawBuffers->extraParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::ExtraParams)));
            descCommonSet->setBuffer(descCommonSet->RtParams, rtResources->rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
#   endif

        descCommonSet->setBuffer(descCommonSet->FrParams, frameParamsBuffer.get(), sizeof(interop::FrameParams));
        descCommonSet->setBuffer(descCommonSet->instanceRenderIndices, renderIndicesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderIndices)));
        descCommonSet->setBuffer(descCommonSet->instanceRDPParams, drawBuffers->rdpParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPParams)));
        descCommonSet->setBuffer(descCommonSet->RDPTiles, drawBuffers->rdpTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::RDPTile)));
        descCommonSet->setBuffer(descCommonSet->GPUTiles, drawBuffers->gpuTilesBuffer.get(), RenderBufferStructuredView(sizeof(interop::GPUTile)));
        descCommonSet->setBuffer(descCommonSet->DynamicRenderParams, drawBuffers->renderParamsBuffer.get(), RenderBufferStructuredView(sizeof(interop::RenderParams)));

        // Make sure the versions vector matches the texture cache size.
        descriptorTextureVersions.resize(textureCacheSize, 0);

        // Update texture vector with static textures from the cache and dynamic resource views.
        if (descriptorTextureGlobalVersion != textureCacheGlobalVersion) {
            const uint32_t textureVersionSize = static_cast<uint32_t>(textureCacheVersions.size());
            for (uint32_t i = 0; i < textureVersionSize; i++) {
                if (textureCacheVersions[i] == descriptorTextureVersions[i]) {
                    continue;
                }

                descriptorTextureVersions[i] = textureCacheVersions[i];
                if (textureCacheTextures[i] == nullptr) {
                    continue;
                }

                if (textureCacheReplacementMapEnabled && (textureCacheTextureReplacements[i] != nullptr)) {
                    descTextureSet->setTexture(i, textureCacheTextureReplacements[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else if (textureCacheTextures[i]->texture != nullptr) {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->texture.get(), RenderTextureLayout::SHADER_READ);
                }
                else {
                    descTextureSet->setTexture(i, textureCacheTextures[i]->tmem.get(), RenderTextureLayout::SHADER_READ);
                }
            }

            descriptorTextureGlobalVersion = textureCacheGlobalVersion;
        }

        for (const DynamicTextureView &dynamicView : dynamicTextureViewVector) {
            descTextureSet->setTexture(dynamicView.dstIndex, dynamicView.texture, RenderTextureLayout::SHADER_READ, dynamicView.textureView);
            descriptorTextureVersions[dynamicView.dstIndex] = 0;
        }
    }

    void FramebufferRenderer::updateRSPSmoothNormalSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (smoothDescSet == nullptr) {
            smoothDescSet = std::make_unique<RSPSmoothNormalDescriptorSet>(worker->device);
        }

        smoothDescSet->setBuffer(smoothDescSet->srcWorldPos, outputBuffers->worldPosBuffer.buffer.get(), outputBuffers->worldPosBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcCol, drawBuffers->normalColorBuffer.get(), drawBuffers->normalColorBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint8_t) * 4));
        smoothDescSet->setBuffer(smoothDescSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        smoothDescSet->setBuffer(smoothDescSet->dstWorldNorm, outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->worldNormBuffer.allocatedSize, RenderBufferStructuredView(sizeof(float) * 4));
    }

    void FramebufferRenderer::updateRSPVertexTestZSet(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers) {
        assert(worker != nullptr);

        if (vertexTestZSet == nullptr) {
            vertexTestZSet = std::make_unique<RSPVertexTestZDescriptorSet>(worker->device);
        }
        
        vertexTestZSet->setBuffer(vertexTestZSet->screenPos, outputBuffers->screenPosBuffer.buffer.get(), RenderBufferStructuredView(sizeof(float) * 4));
        vertexTestZSet->setBuffer(vertexTestZSet->srcFaceIndices, drawBuffers->faceIndicesBuffer.get(), drawBuffers->faceIndicesBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
        vertexTestZSet->setBuffer(vertexTestZSet->dstFaceIndices, outputBuffers->testZIndexBuffer.buffer.get(), outputBuffers->testZIndexBuffer.allocatedSize, RenderBufferStructuredView(sizeof(uint32_t)));
    }

    void FramebufferRenderer::updateShaderViews(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, const bool raytracingEnabled) {
        updateShaderDescriptorSet(worker, drawBuffers, outputBuffers, raytracingEnabled);
        updateRSPVertexTestZSet(worker, drawBuffers, outputBuffers);

#   if RT_ENABLED
        if (raytracingEnabled) {
            updateRSPSmoothNormalSet(worker, drawBuffers, outputBuffers);
            rtResources->updateShaderSets(worker, shaderLibrary);
        }
#   endif
    }

    bool FramebufferRenderer::submitDepthAccess(RenderWorker *worker, RenderFramebufferStorage *fbStorage, bool readOnly, bool &depthState) {
        if (depthState == readOnly) {
            return false;
        }
        
        RenderFramebuffer *renderFramebuffer = readOnly ? fbStorage->colorWriteDepthRead.get() : fbStorage->colorDepthWrite.get();
        const RenderTextureLayout depthReadState = RenderTextureLayout::DEPTH_READ;
        const RenderTextureLayout depthWriteState = RenderTextureLayout::DEPTH_WRITE;
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(fbStorage->depthTarget->texture.get(), readOnly ? depthReadState : RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(renderFramebuffer);
        depthState = readOnly;
        return true;
    }

    void FramebufferRenderer::submitSceneLighting(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState) {
        static_assert(sizeof(ShadowCasterCB) == 64, "Shadow caster push constant must match the shader.");
        static_assert(sizeof(ShadowApplyCB) == 128, "Shadow apply push constant must match the shader.");
        static_assert(offsetof(ShadowApplyCB, screenScaleOffset) == 64, "Shadow apply layout must match the shader.");
        static_assert(offsetof(ShadowApplyCB, lightDirectionBias) == 96, "Shadow apply layout must match the shader.");

        RenderTarget *colorTarget = fbStorage->colorTarget;
        RenderTarget *depthTarget = fbStorage->depthTarget;
        if ((colorTarget == nullptr) || (depthTarget == nullptr) || (colorTarget->texture == nullptr) || (depthTarget->texture == nullptr)) {
            return;
        }
        if ((shaderLibrary == nullptr) || (shaderLibrary->shadowCaster.pipeline == nullptr) || (shaderLibrary->shadowApply.pipeline == nullptr)) {
            return;
        }
        if ((lightingWorldPosBuffer == nullptr) || (lightingWorldPosSize < sizeof(float) * 4) || (lightingDrawData == nullptr)) {
            return;
        }
        if ((rasterScene.viewportWidth < 1.0f) || (rasterScene.viewportHeight < 1.0f) || (rasterScene.nativeWidth < 1.0f) || (rasterScene.nativeHeight < 1.0f)) {
            return;
        }
        if (!lightingSceneLight.enabled) {
            return;
        }

        struct LitDraw {
            uint32_t indexStart;
            uint32_t indexCount;
            interop::float2 screenScale;
            interop::float2 screenOffset;
            RenderRect scissor;
            uint32_t instanceIndex;
            bool caster;
            bool receiver;
            bool vertexTestZ;
            bool cutout;
            bool cameraFacing;
        };
        thread_local std::vector<LitDraw> litDraws;
        thread_local std::vector<ShadowFitRange> fitRanges;
        litDraws.clear();
        fitRanges.clear();

        for (uint32_t instanceIndex : rasterScene.instanceIndices) {
            if (instanceIndex >= instanceDrawCallVector.size()) {
                continue;
            }
            const InstanceDrawCall &drawCall = instanceDrawCallVector[instanceIndex];
            if (drawCall.type != InstanceDrawCall::Type::IndexedTriangles) {
                continue;
            }
            if ((drawCall.triangles.faceCount == 0) || drawCall.triangles.scissor.isEmpty()) {
                continue;
            }

            LitDraw litDraw;
            litDraw.indexStart = drawCall.triangles.indexStart;
            litDraw.indexCount = drawCall.triangles.faceCount * 3;
            litDraw.screenScale = drawCall.triangles.screenScale;
            litDraw.screenOffset = drawCall.triangles.screenOffset;
            litDraw.scissor = drawCall.triangles.scissor;
            litDraw.vertexTestZ = drawCall.triangles.vertexTestZ;
            const interop::OtherMode &otherMode = drawCall.triangles.shaderDesc.otherMode;
            const bool copyMode = (otherMode.cycleType() == G_CYC_COPY);
            const bool solidDepth = (otherMode.zMode() == ZMODE_OPA) || (otherMode.zMode() == ZMODE_INTER);
            const bool alphaCutout = otherMode.cvgXAlpha() || otherMode.alphaCvgSel() || (otherMode.alphaCompare() != G_AC_NONE);
            if (copyMode || !otherMode.zUpd() || !solidDepth) {
                continue;
            }

            litDraw.instanceIndex = instanceIndex;
            litDraw.cutout = alphaCutout;
            litDraw.cameraFacing = alphaCutout && isCameraFacing(*lightingDrawData, litDraw.indexStart, litDraw.indexCount);
            litDraw.caster = !drawCall.triangles.vertexTestZ;
            litDraw.receiver = !litDraw.cameraFacing;

            litDraws.push_back(litDraw);
            if (!litDraw.vertexTestZ && !litDraw.cameraFacing) {
                fitRanges.push_back({ litDraw.indexStart, litDraw.indexCount });
            }
        }
        if (litDraws.empty()) {
            return;
        }

        interop::float4x4 lightMatrix;
        interop::float3 lightDirection;
        interop::float3 worldUp;
        if (!buildDirectionalShadow(*lightingDrawData, rasterScene.nativeWidth, rasterScene.nativeHeight, fitRanges, lightingSceneLight.direction, lightMatrix, lightDirection, worldUp)) {
            return;
        }

        // Cutout casters are drawn from light space positions computed here, so billboards can be turned to
        // face the light and their transparent texels can be discarded.
        const uint32_t vertexCount = lightingDrawData->vertexCount();
        const uint64_t cutoutPositionsSize = uint64_t(vertexCount) * sizeof(float) * 4;
        const bool cutoutSupported = (shaderLibrary->shadowCutoutVertexShader != nullptr) && (shaderLibrary->shadowCutoutPixelShader != nullptr) &&
            (rendererPipelineLayout != nullptr) && (descCommonSet != nullptr) && (descTextureSet != nullptr) && (vertexCount > 0);
        const uint32_t MaxBillboardDrawTriangles = 512;
        thread_local std::vector<ShadowVertex> billboardVertices;
        thread_local std::vector<uint32_t> componentTriangles;
        thread_local std::vector<uint32_t> componentStarts;
        ShadowCamera shadowCamera;
        bool cameraSearched = false;
        float *cutoutPositions = nullptr;
        RenderBuffer *cutoutBuffer = nullptr;
        auto writeLightPosition = [&](uint32_t vertex, const ShadowVec3 &p) {
            float *dst = &cutoutPositions[vertex * 4];
            for (int row = 0; row < 4; row++) {
                dst[row] = lightMatrix.m[0][row] * p.x + lightMatrix.m[1][row] * p.y + lightMatrix.m[2][row] * p.z + lightMatrix.m[3][row];
            }
        };

        uint32_t cutoutCasterCount = 0;
        for (LitDraw &litDraw : litDraws) {
            if (!litDraw.caster || !litDraw.cutout) {
                continue;
            }

            if (!cutoutSupported) {
                litDraw.caster = false;
                continue;
            }

            const uint32_t triangleCount = litDraw.indexCount / 3;
            const bool splittable = (triangleCount <= MaxBillboardDrawTriangles) && (litDraw.indexStart + litDraw.indexCount <= lightingDrawData->faceIndices.size());
            if (!splittable && litDraw.cameraFacing) {
                litDraw.caster = false;
                continue;
            }

            if (cutoutPositions == nullptr) {
                if (shadowCutoutChunkCursor >= shadowCutoutChunks.size()) {
                    shadowCutoutChunks.emplace_back();
                }

                ShadowCutoutChunk &chunk = shadowCutoutChunks[shadowCutoutChunkCursor++];
                if (chunk.capacity < cutoutPositionsSize) {
                    chunk.capacity = cutoutPositionsSize + cutoutPositionsSize / 2;
                    chunk.buffer = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(chunk.capacity, RenderBufferFlag::VERTEX));
                }

                cutoutBuffer = chunk.buffer.get();
                cutoutPositions = static_cast<float *>(cutoutBuffer->map());
            }

            auto writeTriangle = [&](uint32_t triangle, bool hidden) {
                for (uint32_t k = 0; k < 3; k++) {
                    const uint32_t vertex = lightingDrawData->faceIndices[litDraw.indexStart + triangle * 3 + k];
                    if (vertex >= vertexCount) {
                        continue;
                    }

                    if (hidden) {
                        // Off the side of the shadow map. Depth is clamped rather than clipped, so it can't go there.
                        float *dst = &cutoutPositions[vertex * 4];
                        dst[0] = 4.0f;
                        dst[1] = 4.0f;
                        dst[2] = 0.5f;
                        dst[3] = 1.0f;
                    }
                    else {
                        const interop::float4x4 *transform = nullptr;
                        ShadowVec3 p = { 0.0f, 0.0f, 0.0f };
                        scenePosition(*lightingDrawData, vertex, p.x, p.y, p.z, transform);
                        writeLightPosition(vertex, p);
                    }
                }
            };

            if (!splittable) {
                for (uint32_t t = 0; t < triangleCount; t++) {
                    writeTriangle(t, false);
                }
                cutoutCasterCount++;
                continue;
            }

            if (!cameraSearched) {
                shadowCamera = findShadowCamera(*lightingDrawData, fitRanges);
                cameraSearched = true;
            }

            splitShadowComponents(*lightingDrawData, litDraw.indexStart, litDraw.indexCount, componentTriangles, componentStarts);
            bool allBillboards = true;
            for (size_t c = 0; c + 1 < componentStarts.size(); c++) {
                const uint32_t *triangles = &componentTriangles[componentStarts[c]];
                const uint32_t count = componentStarts[c + 1] - componentStarts[c];
                if (reorientBillboard(*lightingDrawData, litDraw.indexStart, triangles, count, shadowCamera, worldUp, lightDirection, billboardVertices)) {
                    for (const ShadowVertex &shadowVertex : billboardVertices) {
                        writeLightPosition(shadowVertex.index, shadowVertex.position);
                    }
                }
                else {
                    allBillboards = false;
                    for (uint32_t t = 0; t < count; t++) {
                        writeTriangle(triangles[t], litDraw.cameraFacing);
                    }
                }
            }

            // Billboards don't receive, since their camera facing surface doesn't match the light facing caster.
            if (allBillboards) {
                litDraw.receiver = false;
            }

            cutoutCasterCount++;
        }

        if (cutoutBuffer != nullptr) {
            cutoutBuffer->unmap();
        }

        if ((cutoutCasterCount > 0) && ((shadowCutoutPipeline == nullptr) || (shadowCutoutPipelineLayout != rendererPipelineLayout))) {
            static const RenderInputElement cutoutInputElements[] = {
                RenderInputElement("POSITION", 0, 0, RenderFormat::R32G32B32A32_FLOAT, 0, 0),
                RenderInputElement("TEXCOORD", 0, 1, RenderFormat::R32G32_FLOAT, 1, 0),
                RenderInputElement("COLOR", 0, 2, RenderFormat::R32G32B32A32_FLOAT, 2, 0)
            };

            RenderGraphicsPipelineDesc pipelineDesc;
            pipelineDesc.pipelineLayout = rendererPipelineLayout;
            pipelineDesc.vertexShader = shaderLibrary->shadowCutoutVertexShader.get();
            pipelineDesc.pixelShader = shaderLibrary->shadowCutoutPixelShader.get();
            pipelineDesc.renderTargetBlend[0] = RenderBlendDesc::Copy();
            pipelineDesc.renderTargetBlend[0].renderTargetWriteMask = uint8_t(RenderColorWriteEnable::RED);
            pipelineDesc.renderTargetFormat[0] = RenderFormat::R32_FLOAT;
            pipelineDesc.renderTargetCount = 1;
            pipelineDesc.depthEnabled = true;
            pipelineDesc.depthWriteEnabled = true;
            pipelineDesc.depthFunction = RenderComparisonFunction::LESS;
            pipelineDesc.depthBias = 0;
            pipelineDesc.slopeScaledDepthBias = 1.0f;
            pipelineDesc.depthClipEnabled = false;
            pipelineDesc.depthTargetFormat = RenderFormat::D32_FLOAT;
            pipelineDesc.cullMode = RenderCullMode::NONE;
            pipelineDesc.inputSlots = vertexInputSlots.data();
            pipelineDesc.inputSlotsCount = uint32_t(vertexInputSlots.size());
            pipelineDesc.inputElements = cutoutInputElements;
            pipelineDesc.inputElementsCount = uint32_t(std::size(cutoutInputElements));
            shadowCutoutPipeline = worker->device->createGraphicsPipeline(pipelineDesc);
            shadowCutoutPipelineLayout = rendererPipelineLayout;
        }

        if (shadowDepth == nullptr) {
            RenderClearValue depthClear = RenderClearValue::Depth(RenderDepth(), RenderFormat::D32_FLOAT);
            shadowColor = worker->device->createTexture(RenderTextureDesc::ColorTarget(ShadowMapSize, ShadowMapSize, RenderFormat::R32_FLOAT));
            shadowColor->setName("Scene Shadow Map");
            shadowColorView = shadowColor->createTextureView(RenderTextureViewDesc::Texture2D(RenderFormat::R32_FLOAT));
            shadowDepth = worker->device->createTexture(RenderTextureDesc::DepthTarget(ShadowMapSize, ShadowMapSize, RenderFormat::D32_FLOAT, RenderMultisampling(), &depthClear));
            shadowDepth->setName("Scene Shadow Depth");
            const RenderTexture *shadowColorAttachment = shadowColor.get();
            shadowFramebuffer = worker->device->createFramebuffer(RenderFramebufferDesc(&shadowColorAttachment, 1, shadowDepth.get()));
        }

        if (shadowMapSetCursor >= shadowMapSets.size()) {
            shadowMapSets.emplace_back(std::make_unique<ShadowMapDescriptorSet>(worker->device));
        }
        ShadowMapDescriptorSet *shadowMapSet = shadowMapSets[shadowMapSetCursor++].get();

        ShadowApplyCB applyCB;
        applyCB.lightViewProj = lightMatrix;
        applyCB.resolutionShadow = interop::float4(rasterScene.nativeWidth, rasterScene.nativeHeight, float(ShadowMapSize), float(ShadowMapSize));
        applyCB.lightDirectionBias = interop::float4(lightDirection.x, lightDirection.y, lightDirection.z, 0.0001f);
        applyCB.shadowTint = interop::float4(lightingSceneLight.shadowTint[0], lightingSceneLight.shadowTint[1], lightingSceneLight.shadowTint[2], lightingSceneLight.facingFloor);

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(const_cast<RenderBuffer *>(lightingWorldPosBuffer), RenderBufferAccess::READ));
        RenderTextureBarrier shadowWrite[] = {
            RenderTextureBarrier(shadowColor.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(shadowDepth.get(), RenderTextureLayout::DEPTH_WRITE)
        };
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, shadowWrite, uint32_t(std::size(shadowWrite)));
        worker->commandList->setFramebuffer(shadowFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(ShadowMapSize), float(ShadowMapSize)));
        worker->commandList->setScissors(RenderRect(0, 0, ShadowMapSize, ShadowMapSize));
        worker->commandList->clearColor(0, RenderColor(1.0f, 1.0f, 1.0f, 1.0f));
        worker->commandList->clearDepth();

        const RenderInputSlot casterSlot(0, sizeof(float) * 4);
        const RenderVertexBufferView worldView(RenderBufferReference(lightingWorldPosBuffer), lightingWorldPosSize);
        worker->commandList->setPipeline(shaderLibrary->shadowCaster.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(shaderLibrary->shadowCaster.pipelineLayout.get());
        worker->commandList->setGraphicsPushConstants(0, &applyCB.lightViewProj);
        worker->commandList->setVertexBuffers(0, &worldView, 1, &casterSlot);
        worker->commandList->setIndexBuffer(&indexBufferView);
        for (const LitDraw &litDraw : litDraws) {
            if (!litDraw.caster || litDraw.cutout) {
                continue;
            }
            worker->commandList->drawIndexedInstanced(litDraw.indexCount, 1, litDraw.indexStart, 0, 0);
        }

        if ((cutoutCasterCount > 0) && (shadowCutoutPipeline != nullptr)) {
            worker->commandList->setPipeline(shadowCutoutPipeline.get());
            worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(framebuffer.descDummyFbSet->get(), 3);

            const RenderVertexBufferView cutoutViews[] = {
                RenderVertexBufferView(RenderBufferReference(cutoutBuffer), uint32_t(cutoutPositionsSize)),
                indexedVertexViews[1],
                indexedVertexViews[2]
            };
            worker->commandList->setVertexBuffers(0, cutoutViews, uint32_t(std::size(cutoutViews)), vertexInputSlots.data());
            worker->commandList->setIndexBuffer(&indexBufferView);

            interop::RasterParams rasterParams;
            rasterParams.screenScale = interop::float2(1.0f, 1.0f);
            rasterParams.screenOffset = interop::float2(0.0f, 0.0f);
            for (const LitDraw &litDraw : litDraws) {
                if (!litDraw.caster || !litDraw.cutout) {
                    continue;
                }
                rasterParams.renderIndex = litDraw.instanceIndex;
                worker->commandList->setGraphicsPushConstants(0, &rasterParams);
                worker->commandList->drawIndexedInstanced(litDraw.indexCount, 1, litDraw.indexStart, 0, 0);
            }
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(shadowColor.get(), RenderTextureLayout::SHADER_READ));
        shadowMapSet->setTexture(shadowMapSet->gShadowMap, shadowColor.get(), RenderTextureLayout::SHADER_READ, shadowColorView.get());
        shadowMapSet->setTexture(shadowMapSet->gSceneDepth, depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, depthTarget->textureView.get());

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ));
        worker->commandList->setFramebuffer(fbStorage->colorWriteDepthRead.get());
        depthState = true;
        worker->commandList->setViewports(RenderViewport(rasterScene.viewportX, rasterScene.viewportY, rasterScene.viewportWidth, rasterScene.viewportHeight));
        worker->commandList->setPipeline(shaderLibrary->shadowApply.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(shaderLibrary->shadowApply.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(shadowMapSet->get(), 0);

        const RenderInputSlot applySlots[] = {
            RenderInputSlot(0, sizeof(float) * 4),
            RenderInputSlot(1, sizeof(float) * 4)
        };
        const RenderVertexBufferView applyViews[] = { indexedVertexViews[0], worldView };
        worker->commandList->setVertexBuffers(0, applyViews, 2, applySlots);
        for (const LitDraw &litDraw : litDraws) {
            if (!litDraw.receiver) {
                continue;
            }
            applyCB.screenScaleOffset = interop::float4(litDraw.screenScale.x, litDraw.screenScale.y, litDraw.screenOffset.x, litDraw.screenOffset.y);
            worker->commandList->setGraphicsPushConstants(0, &applyCB);
            worker->commandList->setScissors(litDraw.scissor);
            worker->commandList->setIndexBuffer(litDraw.vertexTestZ ? &testZIndexBufferView : &indexBufferView);
            worker->commandList->drawIndexedInstanced(litDraw.indexCount, 1, litDraw.indexStart, 0, 0);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->setFramebuffer(fbStorage->colorDepthWrite.get());
        depthState = false;
    }
    
    void FramebufferRenderer::submitRasterScene(RenderWorker *worker, const Framebuffer &framebuffer, RenderFramebufferStorage *fbStorage, const RasterScene &rasterScene, bool &depthState) {
        InstanceDrawCall::Type previousCallType = InstanceDrawCall::Type::Unknown;
        bool previousVertexTestZ = false;
        const RenderPipeline *previousPipeline = nullptr;
        RenderRect previousScissor;
        interop::RasterParams rasterParams;
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descDummyFbSet = framebuffer.descDummyFbSet->get();

        auto switchToGraphicsPipeline = [&]() {
            previousCallType = InstanceDrawCall::Type::Unknown;
            previousVertexTestZ = false;
            previousPipeline = nullptr;
            previousScissor = RenderRect();
            worker->commandList->setGraphicsPipelineLayout(rendererPipelineLayout);
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(depthState ? descRealFbSet : descDummyFbSet, 3);
            worker->commandList->setViewports(framebuffer.viewport);
        };

        auto switchToDepthRead = [&]() {
            if (submitDepthAccess(worker, fbStorage, true, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            }
        };

        auto switchToDepthWrite = [&]() {
            if (submitDepthAccess(worker, fbStorage, false, depthState)) {
                worker->commandList->setGraphicsDescriptorSet(descDummyFbSet, 3);
            }
        };

        auto drawCallTriangles = [&](const InstanceDrawCall &drawCall) {
            if (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) {
                worker->commandList->drawIndexedInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0, 0);
            }
            else {
                worker->commandList->drawInstanced(drawCall.triangles.faceCount * 3, 1, drawCall.triangles.indexStart, 0);
            }
        };

        if (fbStorage->colorTarget != nullptr) {
            switchToGraphicsPipeline();
        }
        
        for (uint32_t i : rasterScene.instanceIndices) {
            const InstanceDrawCall &drawCall = instanceDrawCallVector[i];
            switch (drawCall.type) {
            case InstanceDrawCall::Type::IndexedTriangles: 
            case InstanceDrawCall::Type::RawTriangles:
            case InstanceDrawCall::Type::RegularRect: {
                assert(fbStorage->colorTarget != nullptr);

                const bool typeDifferent = (drawCall.type != previousCallType);
                const bool testZDifferent = (drawCall.type == InstanceDrawCall::Type::IndexedTriangles) && (drawCall.triangles.vertexTestZ != previousVertexTestZ);
                if (typeDifferent || testZDifferent) {
                    switch (drawCall.type) {
                    case InstanceDrawCall::Type::IndexedTriangles:
                        worker->commandList->setVertexBuffers(0, indexedVertexViews.data(), uint32_t(indexedVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(drawCall.triangles.vertexTestZ ? &testZIndexBufferView : &indexBufferView);
                        previousVertexTestZ = drawCall.triangles.vertexTestZ;
                        break;
                    case InstanceDrawCall::Type::RawTriangles:
                    case InstanceDrawCall::Type::RegularRect:
                        worker->commandList->setVertexBuffers(0, rawVertexViews.data(), uint32_t(rawVertexViews.size()), vertexInputSlots.data());
                        worker->commandList->setIndexBuffer(nullptr);
                        break;
                    default:
                        assert(false && "Unknown draw call type.");
                        break;
                    };

                    previousCallType = drawCall.type;
                }

                const auto &triangles = drawCall.triangles;
                assert(triangles.pipeline != nullptr);

                // Draw calls can sometimes end up with empty scissors and cause validation errors. We just skip them.
                if (triangles.scissor.isEmpty()) {
                    continue;
                }

                // A new pass must be started if decals are required and something wrote to the depth buffer before this call.
                const interop::OtherMode otherMode = triangles.shaderDesc.otherMode;
                bool depthDecal = (otherMode.zMode() == ZMODE_DEC);
                bool depthWrite = otherMode.zUpd();
                if (depthDecal) {
                    switchToDepthRead();
                }
                else if (!depthDecal && depthWrite) {
                    switchToDepthWrite();
                }

                if (previousScissor != triangles.scissor) {
                    worker->commandList->setScissors(triangles.scissor);
                    previousScissor = triangles.scissor;
                }

                if (previousPipeline != triangles.pipeline) {
                    worker->commandList->setPipeline(triangles.pipeline);
                    previousPipeline = triangles.pipeline;
                }
                
                rasterParams.renderIndex = i;
                rasterParams.screenScale = triangles.screenScale;
                rasterParams.screenOffset = triangles.screenOffset;
                worker->commandList->setGraphicsPushConstants(0, &rasterParams);

                drawCallTriangles(drawCall);

                // Simulate dither noise.
                if (triangles.postBlendDitherNoise) {
                    if (triangles.postBlendDitherNoiseNegative) {
                        worker->commandList->setPipeline(postBlendDitherNoiseSubNegativePipeline);
                    }
                    else {
                        worker->commandList->setPipeline(postBlendDitherNoiseAddPipeline);
                        drawCallTriangles(drawCall);

                        worker->commandList->setPipeline(postBlendDitherNoiseSubPipeline);
                    }

                    drawCallTriangles(drawCall);
                    previousPipeline = nullptr;
                }

                break;
            };
            case InstanceDrawCall::Type::FillRect: {
                const auto &clearRect = drawCall.clearRect;
                RenderTarget *chosenTarget = (fbStorage->colorTarget != nullptr) ? fbStorage->colorTarget : fbStorage->depthTarget;
                bool rectCoversWholeTarget = (clearRect.rect.left == 0) && (clearRect.rect.top == 0) && (uint32_t(clearRect.rect.right) == chosenTarget->width) && (uint32_t(clearRect.rect.bottom) == chosenTarget->height);
                const RenderRect *clearRects = rectCoversWholeTarget ? nullptr : &clearRect.rect;
                uint32_t clearRectCount = rectCoversWholeTarget ? 0 : 1;
                if (fbStorage->colorTarget != nullptr) {
                    worker->commandList->clearColor(0, clearRect.color, clearRects, clearRectCount);
                }
                else {
                    worker->commandList->clearDepth(true, clearRect.depth, clearRects, clearRectCount);
                }

                break;
            };
            case InstanceDrawCall::Type::VertexTestZ: {
                assert(testZIndexBuffer != nullptr);
                assert(fbStorage->colorTarget != nullptr);

                const interop::RSPVertexTestZCB &testZCB = drawCall.vertexTestZ;
                switchToDepthRead();

                const bool useMSAA = (fbStorage->colorTarget->multisampling.sampleCount > 0);
                const auto &rspVertexTestZ = useMSAA ? shaderLibrary->rspVertexTestZMS : shaderLibrary->rspVertexTestZ;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::WRITE));
                worker->commandList->setPipeline(rspVertexTestZ.pipeline.get());
                worker->commandList->setComputePipelineLayout(rspVertexTestZ.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &testZCB);
                worker->commandList->setComputeDescriptorSet(vertexTestZSet->get(), 0);
                worker->commandList->setComputeDescriptorSet(descRealFbSet, 1);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderBufferBarrier(testZIndexBuffer, RenderBufferAccess::READ));

                switchToGraphicsPipeline();
                break;
            };
            default:
                // Do nothing.
                break;
            }
        }

        // Mark targets for resolve.
        if (fbStorage->colorTarget != nullptr) {
            fbStorage->colorTarget->markForResolve();
        }

        if (fbStorage->depthTarget != nullptr) {
            fbStorage->depthTarget->markForResolve();
        }
    }

    void FramebufferRenderer::updateMultisampling() {
        dummyColorTargetView.reset();
        dummyDepthTargetView.reset();
        dummyColorTarget.reset();
        dummyDepthTarget.reset();
#   if RT_ENABLED
        if (rtResources != nullptr) {
            rtResources->updateMultisampling();
        }
#   endif
    }

#if RT_ENABLED
    void FramebufferRenderer::updateRaytracingScene(RenderWorker *worker, const RaytracingScene &rtScene) {
        auto &rtParams = rtResources->rtParams;
        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;

        // Only use jitter when an upscaler is active.
        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        bool jitterActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (jitterActive) {
            const int phaseCount = upscaler->getJitterPhaseCount(rtResources->textureWidth, rtScene.screenWidth);
            rtParams.pixelJitter = HaltonJitter(frameParams.frameCount, phaseCount);
        }
        else {
            rtParams.pixelJitter = { 0.0f, 0.0f };
        }

        rtParams.viewport.x = rtScene.viewport.x;
        rtParams.viewport.y = rtScene.viewport.y;
        rtParams.viewport.z = rtScene.viewport.width;
        rtParams.viewport.w = rtScene.viewport.height;

        const auto &preset = rtScene.presetScene;
        rtParams.ambientBaseColor = hlslpp::float4(preset.ambientBaseColor, 0.0f);
        rtParams.ambientNoGIColor = hlslpp::float4(preset.ambientNoGIColor, 0.0f);
        rtParams.eyeLightDiffuseColor = hlslpp::float4(preset.eyeLightDiffuseColor, 0.0f);
        rtParams.eyeLightSpecularColor = hlslpp::float4(preset.eyeLightSpecularColor, 0.0f);
        rtParams.giDiffuseStrength = preset.giDiffuseStrength;
        rtParams.giBackgroundStrength = preset.giBackgroundStrength;
        rtParams.tonemapExposure = preset.tonemapExposure;
        rtParams.tonemapWhite = preset.tonemapWhite;
        rtParams.tonemapBlack = preset.tonemapBlack;

        const auto &proj = rtScene.curProjMatrix;
        rtParams.fovRadians = fovFromProj(proj);
        rtParams.nearDist = nearPlaneFromProj(proj);
        rtParams.farDist = farPlaneFromProj(proj);

        if (isnan(rtParams.fovRadians)) {
            rtParams.fovRadians = 0.75f;
        }

        if (isnan(rtParams.nearDist)) {
            rtParams.nearDist = 1.0f;
        }

        if (isnan(rtParams.farDist)) {
            rtParams.farDist = 1000.0f;
        }

        rtParams.view = rtScene.curViewMatrix;
        rtParams.projection = rtScene.curProjMatrix;
        rtParams.viewProj = hlslpp::mul(rtParams.view, rtParams.projection);
        // viewI is the inverse of the same view-projection the rasterizer uses, so rays match its pixels.
        rtParams.viewI = hlslpp::inverse(rtParams.viewProj);
        rtParams.projectionI = hlslpp::inverse(rtParams.projection);
        rtParams.prevViewProj = hlslpp::mul(rtScene.prevViewMatrix, rtScene.prevProjMatrix);
        // +1 when the viewport's Y scale points down the screen. The ray shader uses this to map pixels to clip Y.
        rtParams.nearDist = (rtScene.viewportScaleY >= 0.0f) ? 1.0f : -1.0f;
        rtParams.farDist = 1.0e7f;

        // Angle between the camera rays of two vertically neighboring pixels at the center of the viewport, using the
        // same unprojection as the ray generation shader.
        {
            const hlslpp::float4x4 viewProjInverse = rtParams.viewI;
            const float viewportH = std::max(float(rtParams.viewport.w), 1.0f);
            auto rayDirection = [&](float pixelY) {
                const float vpY = pixelY / viewportH;
                const float clipY = -((-rtParams.nearDist) + (2.0f * rtParams.nearDist) * vpY);
                const hlslpp::float4 nearH = hlslpp::mul(hlslpp::float4(0.0f, clipY, -1.0f, 1.0f), viewProjInverse);
                const hlslpp::float4 midH = hlslpp::mul(hlslpp::float4(0.0f, clipY, 0.0f, 1.0f), viewProjInverse);
                const hlslpp::float3 nearPos = nearH.xyz / float(nearH.w);
                const hlslpp::float3 midPos = midH.xyz / float(midH.w);
                return hlslpp::float3(midPos - nearPos);
            };

            const hlslpp::float3 dirA = rayDirection(viewportH * 0.5f);
            const hlslpp::float3 dirB = rayDirection(viewportH * 0.5f + 1.0f);
            const float lenA = float(hlslpp::length(dirA));
            const float lenB = float(hlslpp::length(dirB));
            float spread = 0.001f;
            if ((lenA > 1e-12f) && (lenB > 1e-12f)) {
                const float cosAngle = std::clamp(float(hlslpp::dot(dirA, dirB)) / (lenA * lenB), -1.0f, 1.0f);
                spread = std::acos(cosAngle);
            }

            rtParams.pixelSpreadAngle = (std::isfinite(spread) && (spread > 1e-6f)) ? spread : 0.001f;
        }

        // cameraW carries the shadow tuning of the scene light shared with the shadow map pass: the color
        // multiplier of a fully shadowed surface (xyz) and of a surface facing away from the light (w).
        if (lightingSceneLight.enabled) {
            rtParams.cameraW = { lightingSceneLight.shadowTint[0], lightingSceneLight.shadowTint[1], lightingSceneLight.shadowTint[2], lightingSceneLight.facingFloor };
        }
        else {
            rtParams.cameraW = { 1.0f, 1.0f, 1.0f, 1.0f };
        }

        // cameraU/V are unused by the ray shaders. They carry the diagnostics probe: a world vertex of
        // the first ray traced draw (w = vertex index) and the CPU camera origin (w = 1 when valid).
        rtParams.cameraU = { rtScene.probeWorld[0], rtScene.probeWorld[1], rtScene.probeWorld[2], rtScene.probeValid ? float(rtScene.probeVertex) : -1.0f };
        {
            const hlslpp::float4 originH = hlslpp::mul(hlslpp::float4(0.0f, 0.0f, 1.0f, 0.0f), hlslpp::float4x4(rtParams.viewI));
            const float originW = originH[3];
            if (std::fabs(originW) > 1e-12f) {
                rtParams.cameraV = { originH[0] / originW, originH[1] / originW, originH[2] / originW, 1.0f };
            }
            else {
                rtParams.cameraV = { 0.0f, 0.0f, 0.0f, 0.0f };
            }
        }

        // Enable light reprojection if denoising is enabled.
#   ifdef DI_REPROJECTION_SUPPORT
        globalParamsBufferData.diReproject = !rtResources->skipReprojection && denoiserEnabled && (globalParamsBufferData.diSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
#   else
        rtParams.diReproject = 0;
#   endif

        rtParams.giReproject = !rtResources->skipReprojection && rtResources->denoiserEnabled && (rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS) ? 1 : 0;
        rtParams.binaryLockMask = (rtResources->upscalerMode != UpscaleMode::FSR);
        rtParams.interleavedRastersCount = interleavedRastersCount;
        
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealDepthSet = framebuffer.descRealFbSet->get();
        RenderDescriptorSet *descriptorSets[] = { descCommonSet->get(), descTextureSet->get(), descTextureSet->get(), descRealDepthSet };
        rtResources->updateTopLevelASResources(worker, rtScene.instances);
        rtResources->createShaderBindingTable(worker, rtState, descriptorSets, uint32_t(std::size(descriptorSets)), hitGroupVector);
        rtResources->updateLightsBuffer(worker, rtScene);

        rtResources->readDiagnosticBuffers();
        rtResources->diagFrame++;
        const uint32_t diagFrame = rtResources->diagFrame;
        rtResources->diagCaptureThisFrame = (diagFrame == 60) || (diagFrame == 600) || (diagFrame == 3000);
        if (rtResources->diagCaptureThisFrame) {
            // Tells the any-hit shaders to count rejected texels this frame.
            rtParams.cameraV.w = 2.0f;
            rtResources->logSceneDiagnostics(rtScene);
        }
    }
    
    void FramebufferRenderer::submitRaytracingScene(RenderWorker *worker, RenderTarget *colorTarget, const RaytracingScene &rtScene) {
        // Unbind any render targets.
        worker->commandList->setFramebuffer(nullptr);

        // Resolve the color target if necessary before using it as the RT scene background.
        colorTarget->resolveTarget(worker, shaderLibrary);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(colorTarget->getResolvedTexture(), RenderTextureLayout::SHADER_READ));

        if (rtResources->transitionOutputBuffers) {
            RenderTextureBarrier afterCreationBarriers[] = {
                RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->directLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->indirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->normalRoughnessTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCreationBarriers, uint32_t(std::size(afterCreationBarriers)));
            rtResources->transitionOutputBuffers = false;
        }

        // Make sure all these buffers are usable as UAVs.
        RenderTextureBarrier preDispatchBarriers[] = {
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[1].get(), RenderTextureLayout::GENERAL)
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, preDispatchBarriers, uint32_t(std::size(preDispatchBarriers)));
        
        rtResources->prepareForDispatch(worker);

        if (rtResources->topLevelAS == nullptr) {
            static uint32_t missingCount = 0;
            if ((missingCount++ % 600) == 0) {
                rtDiagLog("Skipping ray dispatch because the acceleration structure is missing (%u times).", missingCount);
            }
            return;
        }

        const bool diagCapture = rtResources->diagCaptureThisFrame;
        if (diagCapture) {
            rtResources->clearDiagnosticBuffers(worker);
        }

        // Bind pipeline and dispatch primary rays.
        RT64_LOG_PRINTF("Dispatching primary rays");
        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        RenderDescriptorSet *descRealFbSet = framebuffer.descRealFbSet->get();
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 0;
        worker->commandList->setPipeline(rtState->pipeline.get());
        worker->commandList->setRaytracingPipelineLayout(rtPipelineLayout);
        worker->commandList->setRaytracingDescriptorSet(descCommonSet->get(), 0);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 1);
        worker->commandList->setRaytracingDescriptorSet(descTextureSet->get(), 2);
        worker->commandList->setRaytracingDescriptorSet(descRealFbSet, 3);
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Barriers for shading buffers before dispatching secondary rays.
        RenderTextureBarrier shadingBarriers[] = {
            RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->shadingSpecularTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL),
        };

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, shadingBarriers, uint32_t(std::size(shadingBarriers)));

        // Dispatch rays for direct light.
        RT64_LOG_PRINTF("Dispatching direct light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 1;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Dispatch rays for indirect light.
        RT64_LOG_PRINTF("Dispatching indirect light rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 2;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        if (diagCapture) {
            rtResources->copyDiagnosticBuffers(worker);
            rtResources->diagCaptureThisFrame = false;
        }

        // Wait until indirect light is done before dispatching reflection or refraction rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get(), RenderTextureLayout::GENERAL));

        // Dispatch rays for refraction.
        RT64_LOG_PRINTF("Dispatching refraction rays");
        rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 4;
        worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);

        // Wait until refraction is done before dispatching reflection rays.
        // TODO: This is only required to prevent simultaneous usage of the anyhit buffers.
        // This barrier can be removed if this no longer happens, resulting in less serialization of the commands.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::GENERAL));

        // Reflection passes.
        int reflections = rtResources->maxReflections;
        while (reflections > 0) {
            // Dispatch rays for reflection.
            RT64_LOG_PRINTF("Dispatching reflection rays");
            rtResources->shaderBindingTableInfo.groups.rayGen.startIndex = 3;
            worker->commandList->traceRays(rtResources->textureWidth, rtResources->textureHeight, 1, rtResources->shaderBindingTableBuffer->at(0), rtResources->shaderBindingTableInfo.groups);
            reflections--;

            // Add a barrier to wait for the input UAVs to be finished if there's more passes left to be done.
            if (reflections > 0) {
                RenderTextureBarrier newInputBarriers[] = {
                    RenderTextureBarrier(rtResources->viewDirectionTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->instanceIdTexture.get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::GENERAL)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, newInputBarriers, uint32_t(std::size(newInputBarriers)));
            }
        }

        // Copy direct light raw buffer to the first direct filtered buffer.
        {
            RenderTexture *source = rtResources->directLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredDirectLightTexture[1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Copy indirect light raw buffer to the first indirect filtered buffer.
        bool denoiseGI = rtResources->denoiserEnabled && (rtResources->rtParams.giSamples > 0) && (rtResources->upscalerMode != UpscaleMode::DLSS);
        {
            RenderTexture *source = rtResources->indirectLightTexture[rtResources->swapBuffers ? 1 : 0].get();
            RenderTexture *dest = rtResources->filteredIndirectLightTexture[denoiseGI ? 0 : 1].get();

            RenderTextureBarrier beforeCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(dest, RenderTextureLayout::COPY_DEST)
            };

            worker->commandList->barriers(RenderBarrierStage::COPY, beforeCopyBarriers, uint32_t(std::size(beforeCopyBarriers)));
            worker->commandList->copyTexture(dest, source);

            RenderTextureBarrier afterCopyBarriers[] = {
                RenderTextureBarrier(source, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(dest, RenderTextureLayout::SHADER_READ)
            };

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterCopyBarriers, uint32_t(std::size(afterCopyBarriers)));
        }

        // Apply a gaussian filter to the indirect light with a compute shader.
        if (denoiseGI) {
            for (int i = 0; i < 5; i++) {
                const uint32_t ThreadGroupWorkCount = 8;
                uint32_t dispatchX = (rtResources->textureWidth + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                uint32_t dispatchY = (rtResources->textureHeight + ThreadGroupWorkCount - 1) / ThreadGroupWorkCount;
                const ShaderRecord &gaussianFilter = shaderLibrary->gaussianFilterRGB3x3;
                interop::TextureCB textureCB;
                textureCB.TextureSize = { rtResources->textureWidth, rtResources->textureHeight };
                textureCB.TexelSize = { 1.0f / rtResources->textureWidth, 1.0f / rtResources->textureHeight };

                worker->commandList->setPipeline(gaussianFilter.pipeline.get());
                worker->commandList->setComputePipelineLayout(gaussianFilter.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &textureCB);
                worker->commandList->setComputeDescriptorSet(rtResources->indirectFilterSets[i % 2]->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);

                RenderTextureBarrier afterBlurBarriers[] = {
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 1 : 0].get(), RenderTextureLayout::GENERAL),
                    RenderTextureBarrier(rtResources->filteredIndirectLightTexture[(i % 2) ? 0 : 1].get(), RenderTextureLayout::SHADER_READ)
                };

                worker->commandList->barriers(RenderBarrierStage::COMPUTE, afterBlurBarriers, uint32_t(std::size(afterBlurBarriers)));
            }
        }

        // Compose the output buffer.
        RenderTexture *rtOutputCur = rtResources->outputTexture[rtResources->swapBuffers ? 1 : 0].get();

        // Barriers for shading buffers after rays are finished.
        RenderTextureBarrier afterDispatchBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
            RenderTextureBarrier(rtResources->diffuseTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reflectionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->refractionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->transparentTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterDispatchBarriers, uint32_t(std::size(afterDispatchBarriers)));

        // Set the output as the current render target.
        worker->commandList->setFramebuffer(rtResources->outputFramebuffer[rtResources->swapBuffers ? 1 : 0].get());

        // Apply the scissor and viewport to the size of the output texture.
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(rtResources->textureWidth), float(rtResources->textureHeight)));
        worker->commandList->setScissors(RenderRect(0, 0, rtResources->textureWidth, rtResources->textureHeight));

        // Draw the raytracing output.
        RT64_LOG_PRINTF("Composing the raytracing output");
        const ShaderRecord &composeShader = shaderLibrary->compose;
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);
        worker->commandList->setPipeline(composeShader.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(composeShader.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->composeSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Switch resources to the correct states after composing the image
        RenderTextureBarrier afterComposeBarriers[] = {
            RenderTextureBarrier(rtOutputCur, RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rtResources->flowTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->reactiveMaskTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(rtResources->lockMaskTexture.get(), RenderTextureLayout::SHADER_READ)
        };

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, afterComposeBarriers, uint32_t(std::size(afterComposeBarriers)));

        const bool lumaActive = rtScene.presetScene.luminanceRange > 0.0f;
        if (lumaActive) {
            const uint32_t ThreadGroupWorkRegionDim = 8;
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the downscaling shader");
            {
                // Execute the compute shader for downscaling the image.
                const ShaderRecord &bicubicScaling = shaderLibrary->bicubicScaling;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::BicubicCB bicubicCB;
                bicubicCB.InputResolution = { rtResources->textureWidth, rtResources->textureHeight };
                bicubicCB.OutputResolution = { rtResources->textureWidth / 8, rtResources->textureHeight / 8 };
                worker->commandList->setPipeline(bicubicScaling.pipeline.get());
                worker->commandList->setComputePipelineLayout(bicubicScaling.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &bicubicCB);
                worker->commandList->setComputeDescriptorSet(rtResources->downscaleSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }


            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

            RT64_LOG_PRINTF("Do the luminance histogram shader");
            {
                // Execute the compute shader for the luminance histogram.
                const ShaderRecord &luminanceHistogram = shaderLibrary->luminanceHistogram;
                uint32_t dispatchX = ((rtResources->textureWidth / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                uint32_t dispatchY = ((rtResources->textureHeight / 8) + ThreadGroupWorkRegionDim - 1) / ThreadGroupWorkRegionDim;
                interop::LuminanceHistogramCB histogramCB;
                histogramCB.inputWidth = rtResources->textureWidth / 8;
                histogramCB.inputHeight = rtResources->textureHeight / 8;
                histogramCB.minLuminance = rtScene.presetScene.minLuminance;
                histogramCB.oneOverLuminanceRange = 1.0f / rtScene.presetScene.luminanceRange;
                worker->commandList->setPipeline(luminanceHistogram.pipeline.get());
                worker->commandList->setComputePipelineLayout(luminanceHistogram.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &histogramCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSet->get(), 0);
                worker->commandList->dispatch(dispatchX, dispatchY, 1);
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->downscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));

            RT64_LOG_PRINTF("Do the luminance average shader");
            {
                // Execute the compute shader for the luminance histogram average.
                const ShaderRecord &histogramAverage = shaderLibrary->histogramAverage;
                interop::HistogramAverageCB averageCB;
                averageCB.pixelCount = (rtResources->textureWidth / 8) * (rtResources->textureHeight / 8);
                averageCB.minLuminance = rtScene.presetScene.minLuminance;
                averageCB.luminanceRange = rtScene.presetScene.luminanceRange;
                averageCB.timeDelta = rtScene.deltaTime;
                averageCB.tau = rtScene.presetScene.lumaUpdateTime;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramAverage.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramAverage.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &averageCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaAvgSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }

            RT64_LOG_PRINTF("Do the histogram clear shader");
            {
                // Execute the compute shader for clearing the luminance histogram.
                const ShaderRecord &histogramClear = shaderLibrary->histogramClear;
                worker->commandList->setPipeline(histogramClear.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramClear.pipelineLayout.get());
                worker->commandList->setComputeDescriptorSet(rtResources->lumaClearSet->get(), 0);
                worker->commandList->dispatch(ThreadGroupWorkRegionDim, ThreadGroupWorkRegionDim, 1);
            }
        }
        else {
            RT64_LOG_PRINTF("Do the histogram set shader");
            {
                // Execute the compute shader for setting the luminance value.
                const ShaderRecord &histogramSet = shaderLibrary->histogramSet;
                interop::HistogramSetCB setCB;
                setCB.luminanceValue = rtScene.presetScene.minLuminance;
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::GENERAL));
                worker->commandList->setPipeline(histogramSet.pipeline.get());
                worker->commandList->setComputePipelineLayout(histogramSet.pipelineLayout.get());
                worker->commandList->setComputePushConstants(0, &setCB);
                worker->commandList->setComputeDescriptorSet(rtResources->lumaSetSet->get(), 0);
                worker->commandList->dispatch(1, 1, 1);
                worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(rtResources->lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ));
            }
        }

        Upscaler *upscaler = rtResources->getUpscaler(rtResources->upscalerMode);
        const bool upscalerActive = rtResources->upscaleActive && (upscaler != nullptr);
        if (upscalerActive) {
            thread_local std::vector<RenderTextureBarrier> beforeBarriers;
            thread_local std::vector<RenderTextureBarrier> afterBarriers;
            beforeBarriers.clear();
            afterBarriers.clear();

            beforeBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::GENERAL));
            afterBarriers.push_back(RenderTextureBarrier(rtResources->upscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));
            RenderTexture *rtDepthCur = rtResources->depthTexture[rtResources->swapBuffers ? 1 : 0].get();
            if (upscaler->requiresNonShaderResourceInputs()) {
                for (RenderTexture *res : { rtOutputCur, rtResources->flowTexture.get(), rtResources->reactiveMaskTexture.get(), rtResources->lockMaskTexture.get(), rtDepthCur }) {
                    beforeBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                    afterBarriers.push_back(RenderTextureBarrier(res, RenderTextureLayout::SHADER_READ));
                }
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, beforeBarriers);

            Upscaler::UpscaleParameters params;
            params.inRect = { 0, 0, static_cast<int>(rtResources->textureWidth), static_cast<int>(rtResources->textureHeight) };
            params.inDiffuseAlbedo = rtResources->diffuseTexture.get();
            params.inSpecularAlbedo = rtResources->shadingSpecularTexture.get();
            params.inNormalRoughness = rtResources->normalRoughnessTexture[rtResources->swapBuffers ? 1 : 0].get();
            params.inColor = rtOutputCur;
            params.inFlow = rtResources->flowTexture.get();
            params.inReactiveMask = rtResources->upscalerReactiveMask ? rtResources->reactiveMaskTexture.get() : nullptr;
            params.inLockMask = rtResources->upscalerLockMask ? rtResources->lockMaskTexture.get() : nullptr;
            params.inDepth = rtDepthCur;
            params.outColor = rtResources->upscaledOutputTexture.get();
            params.jitterX = -rtResources->rtParams.pixelJitter.x;
            params.jitterY = -rtResources->rtParams.pixelJitter.y;
            params.deltaTime = rtScene.deltaTime * 1000.0f;
            params.nearPlane = rtResources->rtParams.nearDist;
            params.farPlane = rtResources->rtParams.farDist;
            params.fovY = rtResources->rtParams.fovRadians;
            params.resetAccumulation = false; // TODO: Make this configurable via the API.
            upscaler->upscale(worker, params);

            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, afterBarriers);
        }

        // Set the final render target. The ray traced image is in screen pixels, so it covers the target one to one and
        // only the 3D viewport and its scissor are written.
        worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        worker->commandList->setViewports(RenderViewport(0.0f, 0.0f, float(rtResources->textureWidth), float(rtResources->textureHeight)));
        worker->commandList->setScissors(viewportScissorIntersection(rtScene.viewport, rtScene.scissor));
        worker->commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        // Draw final output.
        const ShaderRecord &postProcess = shaderLibrary->postProcess;
        worker->commandList->setPipeline(postProcess.pipeline.get());
        worker->commandList->setGraphicsPipelineLayout(postProcess.pipelineLayout.get());
        worker->commandList->setGraphicsDescriptorSet(rtResources->postProcessSet->get(), 0);
        worker->commandList->drawInstanced(3, 1, 0, 0);

        // Draw debug view on top.
        if (rtResources->rtParams.visualizationMode != interop::VisualizationMode::Final) {
            const ShaderRecord &debugShader = shaderLibrary->debug;
            worker->commandList->setPipeline(debugShader.pipeline.get());
            worker->commandList->setGraphicsPipelineLayout(debugShader.pipelineLayout.get());
            worker->commandList->setGraphicsDescriptorSet(descCommonSet->get(), 0);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 1);
            worker->commandList->setGraphicsDescriptorSet(descTextureSet->get(), 2);
            worker->commandList->setGraphicsDescriptorSet(descRealFbSet, 3);
            worker->commandList->drawInstanced(3, 1, 0, 0);
        }

        if (diagCapture) {
            RenderTexture *finalColor = colorTarget->texture.get();
            RenderTextureLayout finalColorLayout = RenderTextureLayout::COLOR_WRITE;
            if (colorTarget->usesResolve()) {
                colorTarget->markForResolve();
                colorTarget->resolveTarget(worker, shaderLibrary);
                finalColor = colorTarget->resolvedTexture.get();
                finalColorLayout = RenderTextureLayout::SHADER_READ;
            }

            rtResources->copyDiagnosticTextures(worker, finalColor, colorTarget->format, finalColorLayout, colorTarget->width / 2, colorTarget->height / 2);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
            worker->commandList->setFramebuffer(colorTarget->textureFramebuffer.get());
        }

        // Mark targets for resolve.
        colorTarget->markForResolve();
    }
#endif

    void FramebufferRenderer::submitRSPSmoothNormalCompute(RenderWorker *worker, const OutputBuffers *outputBuffers) {
        if (rspSmoothNormalVector.empty()) {
            return;
        }
        
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(outputBuffers->worldNormBuffer.buffer.get(), RenderBufferAccess::WRITE));

        const int ThreadGroupSize = 64;
        const auto &rspSmoothNormal = shaderLibrary->rspSmoothNormal;
        for (const RSPSmoothNormalGenerationCB &cb : rspSmoothNormalVector) {
            const uint32_t triangleCount = cb.indexCount / 3;
            const uint32_t dispatchCount = (triangleCount + ThreadGroupSize - 1) / ThreadGroupSize;
            worker->commandList->setPipeline(rspSmoothNormal.pipeline.get());
            worker->commandList->setComputePipelineLayout(rspSmoothNormal.pipelineLayout.get());
            worker->commandList->setComputePushConstants(0, &cb);
            worker->commandList->setComputeDescriptorSet(smoothDescSet->get(), 0);
            worker->commandList->dispatch(dispatchCount, 1, 1);
        }
    }

    void FramebufferRenderer::recordSetup(RenderWorker *worker, std::vector<BufferUploader *> bufferUploaders, RSPProcessor *rspProcessor,
        VertexProcessor *vertexProcessor, const OutputBuffers *outputBuffers, bool rtEnabled) {
        if (!dummyColorTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyColorTarget.get(), RenderTextureLayout::SHADER_READ));
            dummyColorTargetTransitioned = true;
        }

        if (!dummyDepthTargetTransitioned) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ));
            dummyDepthTargetTransitioned = true;
        }

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListBeforeBarriers(worker);
        }

        shaderUploader->commandListBeforeBarriers(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListCopyResources(worker);
        }

        shaderUploader->commandListCopyResources(worker);

        for (BufferUploader *uploader : bufferUploaders) {
            uploader->commandListAfterBarriers(worker);
        }

        if (rspProcessor != nullptr) {
            rspProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

        if (vertexProcessor != nullptr) {
            vertexProcessor->recordCommandList(worker, shaderLibrary, outputBuffers);
        }

#   if RT_ENABLED
        if (rtEnabled) {
            assert(rtResources != nullptr);
            if (!rtResources->bottomLevelASVector.empty()) {
                if (!rspSmoothNormalVector.empty()) {
                    submitRSPSmoothNormalCompute(worker, outputBuffers);
                }

                rtResources->submitBottomLevelASCreation(worker);
                rtResources->submitTopLevelASCreation(worker);
            }

            if (outputBuffers != nullptr) {
                rtResources->setShaderReadBuffers(outputBuffers->worldNormBuffer.buffer.get(), outputBuffers->shadedColBuffer.buffer.get());
            }
        }
#   endif

        shaderUploader->commandListAfterBarriers(worker);
        pendingUploaders = bufferUploaders;

        if (!dynamicTextureBarrierVector.empty()) {
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, dynamicTextureBarrierVector);
        }
    }

    void FramebufferRenderer::recordFramebuffer(RenderWorker *worker, uint32_t framebufferIndex) {
        // Submit all transition barriers first.
        thread_local std::vector<RenderTextureBarrier> startBarriers;
        startBarriers.clear();

        const Framebuffer &framebuffer = framebufferVector[framebufferIndex];
        for (RenderTarget *target : framebuffer.transitionRenderTargetSet) {
            startBarriers.emplace_back(RenderTextureBarrier(target->getResolvedTexture(), RenderTextureLayout::SHADER_READ));
        }
        
        const RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        RenderTarget *colorTarget = targetDrawCall.fbStorage->colorTarget;
        RenderTarget *depthTarget = targetDrawCall.fbStorage->depthTarget;
        if (colorTarget != nullptr) {
            startBarriers.emplace_back(RenderTextureBarrier(colorTarget->texture.get(), RenderTextureLayout::COLOR_WRITE));
        }

        startBarriers.emplace_back(RenderTextureBarrier(depthTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE));
        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, startBarriers);

        bool depthState = false;
        worker->commandList->setFramebuffer(targetDrawCall.fbStorage->colorDepthWrite.get());
        for (const auto &pair : targetDrawCall.sceneIndices) {
#       if RT_ENABLED
            if (pair.second) {
                const auto &rtScene = targetDrawCall.rtScenes[pair.first];

                // Draw all the interleaved rasterized buffers that will be used in the render target.
                thread_local std::vector<RenderTextureBarrier> interleavedBarriers;
                interleavedBarriers.clear();
                for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                    bool interleavedDepthState = false;
                    const uint32_t sceneIndex = rtScene.interleavedRasters[i].rasterSceneIndex;
                    RenderTarget *colorRenderTarget = rtResources->interleavedColorTargetVector[i].get();
                    RenderTarget *depthRenderTarget = rtResources->interleavedDepthTargetVector[i].get();
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, {
                        RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::COLOR_WRITE),
                        RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_WRITE)
                    });

                    RenderFramebufferStorage *fbStorage = rtResources->interleavedFramebufferStorageVector[i].get();
                    worker->commandList->setFramebuffer(fbStorage->colorDepthWrite.get());
                    worker->commandList->clearColor();
                    worker->commandList->clearDepth();

                    submitDepthAccess(worker, fbStorage, false, interleavedDepthState);
                    submitRasterScene(worker, framebuffer, fbStorage, targetDrawCall.rasterScenes[sceneIndex], interleavedDepthState);

                    // Resolve the interleaved scene.
                    // TODO: Depth textures need to be thrown into a separate view vector for multisampled textures.
                    fbStorage->colorTarget->resolveTarget(worker, shaderLibrary);

                    interleavedBarriers.emplace_back(RenderTextureBarrier(colorRenderTarget->texture.get(), RenderTextureLayout::SHADER_READ));
                    interleavedBarriers.emplace_back(RenderTextureBarrier(depthRenderTarget->texture.get(), RenderTextureLayout::DEPTH_READ));
                }

                if (!interleavedBarriers.empty()) {
                    worker->commandList->barriers(RenderBarrierStage::COMPUTE, interleavedBarriers);
                }

                submitDepthAccess(worker, targetDrawCall.fbStorage, true, depthState);
                submitRaytracingScene(worker, targetDrawCall.fbStorage->colorTarget, rtScene);
            }
            else
#       endif
            {
                const RasterScene &rasterScene = targetDrawCall.rasterScenes[pair.first];
                submitDepthAccess(worker, targetDrawCall.fbStorage, false, depthState);
                submitRasterScene(worker, framebuffer, targetDrawCall.fbStorage, rasterScene, depthState);
                if (rasterScene.lightable && !rasterScene.rtLit) {
                    submitSceneLighting(worker, framebuffer, targetDrawCall.fbStorage, rasterScene, depthState);
                }
            }
        }
    }

    void FramebufferRenderer::waitForUploaders() {
        shaderUploader->wait();

        for (BufferUploader *uploader : pendingUploaders) {
            uploader->wait();
        }
    }

#if RT_ENABLED
    void FramebufferRenderer::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        assert(rtResources != nullptr);
        rtResources->setRaytracingConfig(rtConfig, resolutionChanged);
    }
#endif

    void FramebufferRenderer::addFramebuffer(const DrawParams &p) {
        assert(p.fbStorage != nullptr);
        
        // Setup framebuffer pair data and descriptor set.
        const FramebufferPair &fbPair = p.curWorkload->fbPairs[p.fbPairIndex];
        interop::FramebufferParams fbParams;
        fbParams.resolution = { p.targetWidth / p.resolutionScale.x, p.targetHeight / p.resolutionScale.y };
        fbParams.resolutionScale = p.resolutionScale;
        fbParams.horizontalMisalignment = p.horizontalMisalignment;
        framebufferCount++;

        while (framebufferCount > framebufferVector.size()) {
            framebufferVector.emplace_back();
            framebufferVector.back().paramsBuffer = p.worker->device->createBuffer(RenderBufferDesc::UploadBuffer(256, RenderBufferFlag::CONSTANT));
            framebufferVector.back().descRealFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
            framebufferVector.back().descDummyFbSet = std::make_unique<FramebufferRendererDescriptorFramebufferSet>(p.worker->device);
        }

        Framebuffer &framebuffer = framebufferVector[framebufferCount - 1];
        void *paramBufferBytes = framebuffer.paramsBuffer->map();
        if (paramBufferBytes != nullptr) {
            memcpy(paramBufferBytes, &fbParams, sizeof(interop::FramebufferParams));
            framebuffer.paramsBuffer->unmap();
        }

        RenderTexture *backgroundColorTexture = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTexture() : dummyColorTarget.get();
        RenderTextureView *backgroundColorTextureView = (p.fbStorage->colorTarget != nullptr) ? p.fbStorage->colorTarget->getResolvedTextureView() : dummyColorTargetView.get();
        framebuffer.descRealFbSet->setBuffer(framebuffer.descRealFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descRealFbSet->setTexture(framebuffer.descRealFbSet->gBackgroundDepth, p.fbStorage->depthTarget->texture.get(), RenderTextureLayout::DEPTH_READ, p.fbStorage->depthTarget->textureView.get());
        framebuffer.descDummyFbSet->setBuffer(framebuffer.descDummyFbSet->FbParams, framebuffer.paramsBuffer.get(), sizeof(interop::FramebufferParams));
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundColor, backgroundColorTexture, RenderTextureLayout::SHADER_READ, backgroundColorTextureView);
        framebuffer.descDummyFbSet->setTexture(framebuffer.descDummyFbSet->gBackgroundDepth, dummyDepthTarget.get(), RenderTextureLayout::DEPTH_READ, dummyDepthTargetView.get());

        // Store ubershader and other effect pipelines.
        const RasterShaderUber *rasterShaderUber = p.rasterShaderCache->getGPUShaderUber();
        rendererPipelineLayout = rasterShaderUber->pipelineLayout.get();
        postBlendDitherNoiseAddPipeline = rasterShaderUber->postBlendDitherNoiseAddPipeline.get();
        postBlendDitherNoiseSubPipeline = rasterShaderUber->postBlendDitherNoiseSubPipeline.get();
        postBlendDitherNoiseSubNegativePipeline = rasterShaderUber->postBlendDitherNoiseSubNegativePipeline.get();

        // Make a new render target draw call.
        RenderTargetDrawCall &targetDrawCall = framebuffer.renderTargetDrawCall;
        const DrawData &drawData = p.curWorkload->drawData;
        const DrawBuffers &drawBuffers = p.curWorkload->drawBuffers;
        const OutputBuffers &outputBuffers = p.curWorkload->outputBuffers;
        const RenderBuffer *screenPosRes = outputBuffers.screenPosBuffer.buffer.get();
        const RenderBuffer *tcRes = outputBuffers.genTexCoordBuffer.buffer.get();
        const RenderBuffer *indexRes = drawBuffers.faceIndicesBuffer.get();
        const RenderBuffer *shadedColRes = outputBuffers.shadedColBuffer.buffer.get();
        const RenderBuffer *worldPosRes = outputBuffers.worldPosBuffer.buffer.get();
        const RenderBuffer *worldNormRes = outputBuffers.worldNormBuffer.buffer.get();
        const RenderBuffer *worldVelRes = outputBuffers.worldVelBuffer.buffer.get();
        const RenderBuffer *triPosRes = drawBuffers.triPosBuffer.get();
        const RenderBuffer *triTcRes = drawBuffers.triTcBuffer.get();
        const RenderBuffer *triColRes = drawBuffers.triColorBuffer.get();
        const uint32_t PosStride = sizeof(float) * 4;
        const uint32_t ColStride = sizeof(float) * 4;
        const uint32_t WorldNormStride = sizeof(float) * 4;
        const uint32_t WorldVelStride = sizeof(float) * 4;
        const uint32_t TcStride = sizeof(float) * 2;
        const uint32_t IndexStride = sizeof(uint32_t);
        const uint32_t vertexCount = drawData.vertexCount();
        const uint32_t indexCount = uint32_t(drawData.faceIndices.size());
        const uint32_t rawTriVertexCount = drawData.rawTriVertexCount();
        vertexInputSlots[0] = RenderInputSlot(0, PosStride);
        vertexInputSlots[1] = RenderInputSlot(1, TcStride);
        vertexInputSlots[2] = RenderInputSlot(2, ColStride);
        indexedVertexViews[0] = RenderVertexBufferView(RenderBufferReference(screenPosRes), PosStride * vertexCount);
        indexedVertexViews[1] = RenderVertexBufferView(RenderBufferReference(tcRes), TcStride * vertexCount);
        indexedVertexViews[2] = RenderVertexBufferView(RenderBufferReference(shadedColRes), ColStride * vertexCount);
        indexBufferView = RenderIndexBufferView(RenderBufferReference(indexRes), IndexStride * indexCount, RenderFormat::R32_UINT);
        rawVertexViews[0] = RenderVertexBufferView(RenderBufferReference(triPosRes), PosStride * rawTriVertexCount);
        rawVertexViews[1] = RenderVertexBufferView(RenderBufferReference(triTcRes), TcStride * rawTriVertexCount);
        rawVertexViews[2] = RenderVertexBufferView(RenderBufferReference(triColRes), ColStride * rawTriVertexCount);
        testZIndexBuffer = outputBuffers.testZIndexBuffer.buffer.get();
        testZIndexBufferView = RenderIndexBufferView(testZIndexBuffer, uint32_t(outputBuffers.testZIndexBuffer.allocatedSize), RenderFormat::R32_UINT);
        lightingDrawData = &drawData;
        lightingSceneLight = p.curWorkload->extended.sceneLight;
        lightingWorldPosBuffer = worldPosRes;
        lightingWorldPosSize = uint32_t(outputBuffers.worldPosBuffer.computedSize);

        RasterScene rasterScene;
        auto checkRasterScene = [&](RasterScene &rasterScene) {
            if (!rasterScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rasterScenes.size());
                targetDrawCall.rasterScenes.push_back(rasterScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, false });
                rasterScene.instanceIndices.clear();

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rasterScenes.clear();

#   if RT_ENABLED
        RaytracingScene rtScene;
        auto checkRtScene = [&](RaytracingScene &rtScene) {
            if (!rtScene.instanceIndices.empty()) {
                uint32_t sceneIndex = static_cast<uint32_t>(targetDrawCall.rtScenes.size());
                targetDrawCall.rtScenes.push_back(rtScene);
                targetDrawCall.sceneIndices.push_back({ sceneIndex, true });
                rtScene.instanceIndices.clear();
                rtScene.instances.clear();
                rtScene.skippedDraws = 0;
                rtScene.rasterAfterDraws = 0;
                rtScene.rasterOnlyOpaqueDraws = 0;
                rtScene.drawsTwoCycle = 0;
                rtScene.drawsTextureLOD = 0;
                rtScene.drawsTexture1 = 0;
                rtScene.drawsUntextured = 0;
                rtScene.drawsFlatShade = 0;
                rtScene.drawsTexturePerspectiveOff = 0;
                rtScene.drawsCulling = 0;
                rtScene.incompatibleProjections = 0;
                rtScene.skippedLayeredDraws = 0;
                rtScene.billboardsCylindrical = 0;
                rtScene.billboardsSpherical = 0;
                rtScene.billboardProxies = 0;

                return true;
            }
            else {
                return false;
            }
        };

        targetDrawCall.rtScenes.clear();
#   endif

        // Submits the raster scene. In a lightable 3D view with path traced draws the order is: the raster scene
        // (sky, background and the path traced draws, which fill the depth buffer), the path traced pass that
        // replaces the pixels its rays hit, then the draws that are not path traced, rasterized on top.
        RasterScene rasterSceneAfter;
        auto flushRasterScene = [&]() {
#       if RT_ENABLED
            const bool rtLit = rasterScene.lightable && !rasterScene.instanceIndices.empty() && !rtScene.instanceIndices.empty();
            rasterScene.rtLit = rtLit;
            checkRasterScene(rasterScene);
            rasterScene.rtLit = false;
            if (rtLit) {
                checkRtScene(rtScene);
            }

            rasterSceneAfter.lightable = false;
            checkRasterScene(rasterSceneAfter);
#       else
            checkRasterScene(rasterScene);
#       endif
        };

        targetDrawCall.fbStorage = p.fbStorage;
        targetDrawCall.sceneIndices.clear();

        const float SimilarityPercentage = 0.1f; // TODO: Make more strict once VI ratios are in.
        const float scissorRatio = static_cast<float>(fbPair.scissorRect.width(false, true)) / static_cast<float>(fbPair.scissorRect.height(false, true));
        const bool adjustRatio = (abs((scissorRatio / p.aspectRatioSource) - 1.0f) < SimilarityPercentage);
        const float aspectRatioScale = adjustRatio ? (p.aspectRatioTarget / p.aspectRatioSource) : 1.0f;
        InstanceDrawCall instanceDrawCall;
        interop::RenderIndices renderIndices;
        FixedRect fixedRect;
        uint32_t globalCallIndex = 0;
        const float wideWidth = p.fbWidth * p.resolutionScale.x;
        const float originalWidth = p.fbWidth * p.resolutionScale.y;
        const float commonHeight = float(p.targetHeight);
        framebuffer.viewport = RenderViewport(0.0f, 0.0f, wideWidth, commonHeight);
        
        const interop::float2 halfViewportSize = { framebuffer.viewport.width / 2.0f, framebuffer.viewport.height / 2.0f };
        const interop::float2 halfPixelOffset = { 1.0f / framebuffer.viewport.width, -1.0f / framebuffer.viewport.height };
        const float middleViewport = (wideWidth / 2.0f) - (originalWidth / 2.0f);
        const float extOriginPercentage = p.extAspectPercentage;
        uint32_t vertexTestZFaceIndicesStart = 0;
        int32_t vertexTestZCallIndex = -1;
        RenderViewport viewportClip;
        for (uint32_t pr = 0; (pr < fbPair.projectionCount) && (globalCallIndex < p.maxGameCall); pr++) {
            const Projection &proj = fbPair.projections[pr];
            if (proj.scissorRect.isNull()) {
                continue;
            }

            // Keep the 3D view in its own scene so lighting runs before the HUD is drawn on top.
            const bool projLightable = p.advancedLighting && ((proj.type == Projection::Type::Perspective) || (proj.type == Projection::Type::Orthographic));
            if (!rasterScene.instanceIndices.empty() && (rasterScene.lightable != projLightable)) {
                flushRasterScene();
                rasterScene.lightable = false;
            }
            if (projLightable) {
                rasterScene.lightable = true;
                rasterScene.nativeWidth = fbParams.resolution.x;
                rasterScene.nativeHeight = fbParams.resolution.y;
                rasterScene.viewportX = framebuffer.viewport.x;
                rasterScene.viewportY = framebuffer.viewport.y;
                rasterScene.viewportWidth = framebuffer.viewport.width;
                rasterScene.viewportHeight = framebuffer.viewport.height;
            }

#       if RT_ENABLED
            // TODO: Move heuristics of RT proj elsewhere?
            // TODO: Use detected scenes logic instead.
            const bool perspProj = (proj.type == Projection::Type::Perspective);
            bool rtProj = p.rtEnabled && perspProj && fbPair.depthWrite && targetDrawCall.rtScenes.empty(); // TODO: Remove the last condition once multiple heaps per RT scene are supported.

            // Make sure the matrices are compatible if we're switching to a new projection.
            // The acceleration structure holds world space positions, so draws only need to be seen through the same
            // camera as the scene. Compare relative to the matrix size: an absolute threshold of 1e-6 let float noise
            // in reloaded or interpolated camera matrices flip draws between path traced and rasterized every frame.
            bool rtProjCompatible = true;
            if (rtProj && (!rtScene.instanceIndices.empty())) {
                const float RelativeThreshold = 1e-3f;
                const hlslpp::float4x4 zero(0.0f);
                const float viewScale = std::max(matrixDifference(rtScene.curViewMatrix, zero), 1.0f);
                const float projScale = std::max(matrixDifference(rtScene.curProjMatrix, zero), 1.0f);
                const float viewMatrixDiff = matrixDifference(drawData.modViewTransforms[proj.transformsIndex], rtScene.curViewMatrix);
                const float projMatrixDiff = matrixDifference(drawData.modProjTransforms[proj.transformsIndex], rtScene.curProjMatrix);
                rtProjCompatible = (viewMatrixDiff < (RelativeThreshold * viewScale)) && (projMatrixDiff < (RelativeThreshold * projScale));
            }

            // TODO: Remove this condition once multiple heaps per RT scene are supported.
            if (rtProj && !rtProjCompatible) {
                rtProj = false;
                rtScene.incompatibleProjections++;
            }
#       endif
            
            auto &triangles = instanceDrawCall.triangles;
            triangles.screenScale = { 1.0f, 1.0f };
            triangles.screenOffset = halfPixelOffset;

            float projInvRatioScale = 1.0f / aspectRatioScale;
            const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            if (proj.usesViewport()) {
                // The call's scissor spans the whole width of the framebuffer pair scissor. Custom origin must not be in use to be able to use the stretched viewport.
                const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                FixedRect intersectionRect = proj.scissorRect.intersection(viewport.rect(viewportClipRatios));
                bool coversWholeWidth = !intersectionRect.isEmpty() && (intersectionRect.ulx <= fbPair.scissorRect.ulx) && (intersectionRect.lrx >= fbPair.scissorRect.lrx);
                bool horizontalRatio = !intersectionRect.isEmpty() && (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                bool useWideViewport = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
                if (useWideViewport) {
                    projInvRatioScale = 1.0f;
                }
                else {
                    triangles.screenScale.x = originalWidth / wideWidth;

                    if (viewportOrigin < G_EX_ORIGIN_NONE) {
                        const float centerOffset = ((middleViewport * viewportOrigin) / G_EX_ORIGIN_CENTER) * extOriginPercentage + middleViewport * (1.0f - extOriginPercentage);
                        triangles.screenOffset.x = halfPixelOffset.x + ((centerOffset - middleViewport) / halfViewportSize.x);
                    }
                }

                viewportClip = convertViewportRect(viewport.rect(viewportClipRatios), p.resolutionScale, p.fbWidth, projInvRatioScale, extOriginPercentage, 0.0f, viewportOrigin, viewportOrigin);
            }

            for (uint32_t d = 0; (d < proj.gameCallCount) && (globalCallIndex < p.maxGameCall); d++) {
                const GameCall &call = proj.gameCalls[d];
                renderIndices.instanceIndex = call.callDesc.callIndex;
                renderIndices.faceIndicesStart = call.meshDesc.faceIndicesStart;
                renderIndices.rdpTileIndex = call.callDesc.tileIndex;
                renderIndices.rdpTileCount = call.callDesc.tileCount;
                renderIndices.highlightColor = call.debuggerDesc.highlightColor;
                renderIndicesVector.push_back(renderIndices);
#           if RT_ENABLED
                bool rtInstanceAdded = false;
                RaytracingInstance rtInstance;
                bool rtProxyAdded = false;
                RaytracingInstance rtProxyInstance;
#           endif

                uint32_t cycleType = call.callDesc.otherMode.cycleType();
                if (cycleType == G_CYC_FILL) {
                    instanceDrawCall.type = InstanceDrawCall::Type::FillRect;

                    auto &clearRect = instanceDrawCall.clearRect;
                    if (call.debuggerDesc.highlightColor > 0) {
                        clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.debuggerDesc.highlightColor));
                    }
                    else {
                        if (p.fbStorage->colorTarget == nullptr) {
                            clearRect.depth = ColorConverter::D16::toF(call.callDesc.fillColor & 0xFFFF);
                        }
                        else if (fbPair.colorImage.siz == G_IM_SIZ_32b) {
                            clearRect.color = toRenderColor(ColorConverter::RGBA32::toRGBAF(call.callDesc.fillColor));
                        }
                        else {
                            clearRect.color = toRenderColor(ColorConverter::RGBA16::toRGBAF(call.callDesc.fillColor & 0xFFFF));
                        }
                    }

                    float invRatioScale = 1.0f / aspectRatioScale;
                    int32_t horizontalMisalignment = 0;
                    fixedRect = fixRect(call.callDesc.rect, fbPair.scissorRect, p.fixRectLR);

                    // A rect that spans the whole width of the scissor.
                    if ((fixedRect.ulx <= fbPair.scissorRect.ulx) && (fixedRect.lrx >= fbPair.scissorRect.lrx)) {
                        invRatioScale = 1.0f;
                    }
                    // A regular rectangle that should correct its misalignment.
                    else {
                        horizontalMisalignment = int32_t(p.horizontalMisalignment);
                    }

                    clearRect.rect = convertFixedRect(fixedRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                }
                else if (call.callDesc.extendedType != DrawExtendedType::None) {
                    switch (call.callDesc.extendedType) {
                    case DrawExtendedType::VertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::VertexTestZ;
                        instanceDrawCall.vertexTestZ.vertexIndex = call.callDesc.extendedData.vertexTestZ.vertexIndex;
                        instanceDrawCall.vertexTestZ.resolutionScale = p.resolutionScale;
                        instanceDrawCall.vertexTestZ.srcIndexStart = call.meshDesc.faceIndicesStart + 3;
                        instanceDrawCall.vertexTestZ.dstIndexStart = vertexTestZFaceIndicesStart;
                        instanceDrawCall.vertexTestZ.indexCount = 0;
                        vertexTestZCallIndex = int32_t(instanceDrawCallVector.size());
                        break;
                    case DrawExtendedType::EndVertexTestZ:
                        instanceDrawCall.type = InstanceDrawCall::Type::Unknown;
                        vertexTestZCallIndex = -1;
                        break;
                    default:
                        assert(false && "Unknown extended type.");
                        break;
                    }
                }
                else {
#               if RT_ENABLED
                    // Path traced materials: draws with depth test and write in the 3D view go into the acceleration
                    // structure and are shaded by the ray tracer. They are still rasterized first to fill the depth
                    // buffer, so the draws that come after (decals, blended effects, flat shadows) can be rasterized
                    // on top of the path traced image with correct occlusion.
                    if (rtProj && projLightable && (rtState != nullptr) && rtState->valid) {
                        const interop::OtherMode &otherMode = call.shaderDesc.otherMode;
                        // Decal and translucent depth modes are layers over other surfaces. The raster draws them on
                        // top by depth mode and draw order, so they stay rasterized after the path traced pass.
                        const bool layeredDepthMode = (otherMode.zMode() == ZMODE_DEC) || (otherMode.zMode() == ZMODE_XLU);
                        if (layeredDepthMode) {
                            rtScene.skippedLayeredDraws++;
                        }

                        if (otherMode.zCmp() && otherMode.zUpd() && !layeredDepthMode && (call.callDesc.triangleCount > 0)) {
                            if (hitGroupVector.empty()) {
                                const RaytracingShaderPrograms &shaderPrograms = rtState->shaderProgramsMap.find(UberShaderHash)->second;
                                hitGroupVector.emplace_back(shaderPrograms.surface);
                                hitGroupVector.emplace_back(shaderPrograms.shadow);
                            }

                            // Alpha tested and blended materials run the any-hit shaders, which test the texture alpha.
                            // ALPHA_CVG_SEL alone does not make a cutout: the standard opaque render modes set it too.
                            const bool alphaTested = (otherMode.alphaCompare() != G_AC_NONE) || otherMode.cvgXAlpha() || interop::Blender::usesAlphaBlend(otherMode);
                            const interop::RenderFlags &flags = call.shaderDesc.flags;
                            // Camera rays must skip the faces the raster culls, which the any-hit shader tests.
                            rtInstance.alphaTested = alphaTested;
                            rtInstance.culling = flags.culling && (otherMode.cycleType() != G_CYC_COPY);
                            rtInstance.opaque = !rtInstance.alphaTested && !rtInstance.culling;
                            rtScene.drawsCulling += rtInstance.culling ? 1 : 0;

                            rtScene.drawsTwoCycle += (otherMode.cycleType() == G_CYC_2CYCLE) ? 1 : 0;
                            rtScene.drawsTextureLOD += (otherMode.textLOD() == G_TL_LOD) ? 1 : 0;
                            rtScene.drawsTexture1 += flags.usesTexture1 ? 1 : 0;
                            rtScene.drawsUntextured += (!flags.usesTexture0 && !flags.usesTexture1) ? 1 : 0;
                            rtScene.drawsFlatShade += flags.smoothShade ? 0 : 1;
                            rtScene.drawsTexturePerspectiveOff += (otherMode.textPersp() == G_TP_NONE) ? 1 : 0;
                            rtInstance.renderIndex = uint32_t(renderIndicesVector.size() - 1);
                            const RenderBottomLevelASMesh asMesh(indexRes->at(call.meshDesc.faceIndicesStart * IndexStride), worldPosRes->at(0), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, call.callDesc.triangleCount * 3, vertexCount, PosStride, rtInstance.opaque);
                            rtInstance.bottomLevelIndex = uint32_t(rtResources->pendingMeshes.size());
                            rtInstance.queryMask = RtCameraRayMask | RtShadowRayMask;
                            // N64 winding does not match the ray tracer's front face, so both sides must be hit.
                            rtInstance.cullDisable = true;
                            rtResources->addBottomLevelASMesh(asMesh);
                            rtInstanceAdded = true;

                            // Camera facing cutout sprites (tree canopies and the like) would cast thin shadows when the
                            // camera looks across the light. Shadow rays see a copy of the draw with those sprites turned
                            // to face the light instead, and the drawn sprites only take camera and bounce rays.
                            if (alphaTested && (proj.pointLightCount > 0)) {
                                const ShadowCamera camera = cameraFromViewProj(drawData.modViewProjTransforms[proj.transformsIndex]);
                                const interop::PointLight &light = proj.pointLights[0];
                                const float lightLength = std::sqrt(light.position.x * light.position.x + light.position.y * light.position.y + light.position.z * light.position.z);
                                thread_local std::vector<float> proxyPositions;
                                if (camera.valid && (lightLength > 1e-4f)) {
                                    const interop::float3 lightDirection = { light.position.x / lightLength, light.position.y / lightLength, light.position.z / lightLength };
                                    if (buildBillboardProxy(drawData, call.meshDesc.faceIndicesStart, call.callDesc.triangleCount * 3, camera, lightDirection,
                                        proxyPositions, rtScene.billboardsCylindrical, rtScene.billboardsSpherical))
                                    {
                                        rtProxyInstance = rtInstance;
                                        rtProxyInstance.culling = false;
                                        rtProxyInstance.proxy = true;
                                        rtProxyInstance.opaque = false;
                                        rtProxyInstance.queryMask = RtShadowRayMask;
                                        rtProxyInstance.bottomLevelIndex = rtResources->addBillboardProxyMesh(proxyPositions);
                                        rtInstance.queryMask = RtCameraRayMask;
                                        rtProxyAdded = true;
                                        rtScene.billboardProxies++;
                                    }
                                }
                            }
                        }
                        else {
                            rtScene.skippedDraws++;
                        }
                    }
#               endif
                    {
                        triangles.shaderDesc = call.shaderDesc;

                        RasterShader *gpuShader = p.ubershadersOnly ? nullptr : p.rasterShaderCache->getGPUShader(call.shaderDesc);
                        if (gpuShader != nullptr) {
                            triangles.pipeline = gpuShader->pipeline.get();
                        }
                        else {
                            const bool copyMode = (call.shaderDesc.otherMode.cycleType() == G_CYC_COPY);
                            triangles.pipeline = rasterShaderUber->getPipeline(
                                !copyMode && call.shaderDesc.otherMode.zCmp() && (call.shaderDesc.otherMode.zMode() != ZMODE_DEC),
                                !copyMode && call.shaderDesc.otherMode.zUpd(),
                                (call.shaderDesc.otherMode.cvgDst() == CVG_DST_WRAP) || (call.shaderDesc.otherMode.cvgDst() == CVG_DST_SAVE));
                        }
                        
                        triangles.faceCount = call.callDesc.triangleCount;
                        triangles.vertexTestZ = (vertexTestZCallIndex >= 0);
                        triangles.postBlendDitherNoise = false;

                        float invRatioScale = 1.0f / aspectRatioScale;
                        float horizontalMisalignment = 0.0f;
                        switch (proj.type) {
                        case Projection::Type::Perspective:
                        case Projection::Type::Orthographic: {
                            instanceDrawCall.type = InstanceDrawCall::Type::IndexedTriangles;
                            triangles.indexStart = triangles.vertexTestZ ? vertexTestZFaceIndicesStart : call.meshDesc.faceIndicesStart;
                            invRatioScale = projInvRatioScale;
                            break;
                        }
                        case Projection::Type::Rectangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RegularRect;
                            triangles.indexStart = call.meshDesc.rawVertexStart;
                            fixedRect = fixRect(call.callDesc.rect, fbPair.scissorRect, p.fixRectLR);

                            bool tileCopiesUsed = false;
                            for (uint32_t t = 0; (t < call.callDesc.tileCount) && !tileCopiesUsed; t++) {
                                tileCopiesUsed = drawData.callTiles[call.callDesc.tileIndex + t].tileCopyUsed;
                            }

                            // The call's scissor spans the whole width of the framebuffer pair scissor. The rect must not be using extended origins.
                            const bool regularOrigins = (call.callDesc.rectLeftOrigin == G_EX_ORIGIN_NONE) && (call.callDesc.rectRightOrigin == G_EX_ORIGIN_NONE);
                            const bool coversScissorWidth = regularOrigins && (fixedRect.ulx <= fbPair.scissorRect.ulx) && (fixedRect.lrx >= fbPair.scissorRect.lrx);
                            if (tileCopiesUsed || coversScissorWidth) {
                                invRatioScale = 1.0f;
                            }
                            else {
                                horizontalMisalignment = p.horizontalMisalignment;
                            }

                            RenderViewport viewportRect = convertViewportRect(fixedRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, horizontalMisalignment, call.callDesc.rectLeftOrigin, call.callDesc.rectRightOrigin);
                            triangles.screenScale = { viewportRect.width / framebuffer.viewport.width, viewportRect.height / framebuffer.viewport.height };
                            triangles.screenOffset.x = halfPixelOffset.x + ((viewportRect.x + viewportRect.width / 2.0f) - halfViewportSize.x) / halfViewportSize.x;
                            triangles.screenOffset.y = halfPixelOffset.y + (halfViewportSize.y - (viewportRect.y + viewportRect.height / 2.0f)) / halfViewportSize.y;

                            if (p.postBlendNoise) {
                                // Indicate if post blend dither noise should be applied.
                                bool rgbDitherNoise = (call.shaderDesc.otherMode.rgbDither() == G_CD_NOISE);
                                triangles.postBlendDitherNoise = rgbDitherNoise && !call.shaderDesc.otherMode.zCmp() && !call.shaderDesc.otherMode.zUpd();
                                triangles.postBlendDitherNoiseNegative = p.postBlendNoiseNegative;
                            }

                            break;
                        }
                        case Projection::Type::Triangle: {
                            instanceDrawCall.type = InstanceDrawCall::Type::RawTriangles;
                            triangles.indexStart = call.meshDesc.rawVertexStart;
                            break;
                        }
                        case Projection::Type::None:
                        default:
                            break;
                        }

                        triangles.scissor = convertFixedRect(call.callDesc.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, int32_t(horizontalMisalignment), call.callDesc.scissorLeftOrigin, call.callDesc.scissorRightOrigin);

                        bool usesViewport = (proj.type == Projection::Type::Perspective) || (proj.type == Projection::Type::Orthographic);
                        if (usesViewport) {
                            triangles.scissor = viewportScissorIntersection(viewportClip, triangles.scissor);
                        }
                        
                        if (triangles.vertexTestZ && usesViewport) {
                            instanceDrawCallVector[vertexTestZCallIndex].vertexTestZ.indexCount += call.callDesc.triangleCount * 3;
                            vertexTestZFaceIndicesStart += call.callDesc.triangleCount * 3;
                        }
                    }
                }

                // Every draw is rasterized. Shadow casting draws are also recorded in the ray traced lighting scene.
                const uint32_t instanceIndex = static_cast<uint32_t>(instanceDrawCallVector.size());
#           if RT_ENABLED
                if (rtInstanceAdded) {
                    if (rtScene.instanceIndices.empty()) {
                        float projRatioScale = 1.0f / aspectRatioScale;
                        float invRatioScale = 1.0f / aspectRatioScale;
                        const bool coversScissorWidth = (proj.scissorRect.ulx <= fbPair.scissorRect.ulx) && (proj.scissorRect.lrx >= fbPair.scissorRect.lrx);
                        if (coversScissorWidth) {
                            invRatioScale = 1.0f;
                        }
                        else {
                            projRatioScale = 1.0f;
                        }

                        rtScene.curViewMatrix = drawData.modViewTransforms[proj.transformsIndex];
                        rtScene.curProjMatrix = drawData.modProjTransforms[proj.transformsIndex];
                        rtScene.prevViewMatrix = drawData.prevViewTransforms[proj.transformsIndex];
                        rtScene.prevProjMatrix = drawData.prevProjTransforms[proj.transformsIndex];

                        // viewportClipRatios describe the guard band used for clipping (3x the viewport in this game),
                        // not the area the image covers. The ray image must cover exactly the viewport.
                        const auto &viewport = drawData.rspViewports[proj.transformsIndex];
                        const int16_t unitClipRatios[4] = { 1, 1, -1, -1 };
                        rtScene.viewport = convertViewportRect(viewport.rect(unitClipRatios), p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0.0f, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);
                        rtScene.viewportScaleY = viewport.scale.y;
                        rtScene.scissor = convertFixedRect(proj.scissorRect, p.resolutionScale, p.fbWidth, invRatioScale, extOriginPercentage, 0, G_EX_ORIGIN_NONE, G_EX_ORIGIN_NONE);

                        rtScene.presetScene = p.presetScene;
                        rtScene.deltaTime = p.deltaTimeMs * 0.001f;
                        rtScene.screenWidth = lround(static_cast<float>(p.fbWidth) * p.resolutionScale.x);
                        rtScene.screenHeight = lround(static_cast<float>(p.fbHeight) * p.resolutionScale.y);

                        if (proj.pointLightCount > 0) {
                            rtScene.pointLights = proj.pointLights.data();
                            rtScene.lightCount = proj.pointLightCount;
                        }
                        else {
                            rtScene.pointLights = nullptr;
                            rtScene.lightCount = 0;
                        }

                        rtScene.probeValid = false;
                        if (call.meshDesc.faceIndicesStart < drawData.faceIndices.size()) {
                            const uint32_t probeVertex = drawData.faceIndices[call.meshDesc.faceIndicesStart];
                            const interop::float4x4 *probeTransform = nullptr;
                            float px, py, pz;
                            if (scenePosition(drawData, probeVertex, px, py, pz, probeTransform)) {
                                rtScene.probeValid = true;
                                rtScene.probeVertex = probeVertex;
                                rtScene.probeWorld[0] = px;
                                rtScene.probeWorld[1] = py;
                                rtScene.probeWorld[2] = pz;
                                for (int c = 0; c < 4; c++) {
                                    rtScene.probeClipRSP[c] = (probeVertex < drawData.posTransformed.size()) ? float(drawData.posTransformed[probeVertex][c]) : 0.0f;
                                }
                            }
                        }
                    }

                    rtScene.instanceIndices.push_back(instanceIndex);
                    rtScene.instances.push_back(rtInstance);
                    if (rtProxyAdded) {
                        rtScene.instanceIndices.push_back(instanceIndex);
                        rtScene.instances.push_back(rtProxyInstance);
                    }

                    rasterScene.instanceIndices.push_back(instanceIndex);
                }
                else if (rasterScene.lightable && !rtScene.instanceIndices.empty()) {
                    rasterSceneAfter.instanceIndices.push_back(instanceIndex);
                    rtScene.rasterAfterDraws++;
                    const interop::OtherMode &otherMode = call.shaderDesc.otherMode;
                    if ((instanceDrawCall.type == InstanceDrawCall::Type::IndexedTriangles) && otherMode.zCmp() && otherMode.zUpd()) {
                        rtScene.rasterOnlyOpaqueDraws++;
                    }
                }
                else
#           endif
                {
                    rasterScene.instanceIndices.push_back(instanceIndex);
                }

                instanceDrawCallVector.push_back(instanceDrawCall);
                globalCallIndex++;
            }
        }

        flushRasterScene();
    }

    void FramebufferRenderer::endFramebuffers(RenderWorker *worker, const DrawBuffers *drawBuffers, const OutputBuffers *outputBuffers, bool rtEnabled) {
        bool shaderViewRtEnabled = false;
        std::vector<BufferUploader::Upload> shaderUploads = {
            { renderIndicesVector.data(), { 0, renderIndicesVector.size() }, sizeof(interop::RenderIndices), RenderBufferFlag::STORAGE, { }, &renderIndicesBuffer},
            { &frameParams, { 0, 1 }, sizeof(interop::FrameParams), RenderBufferFlag::CONSTANT, { }, &frameParamsBuffer}
        };

#   if RT_ENABLED
        // FIXME: Add support for multiple raytracing scenes.
        Framebuffer *chosenFramebuffer = nullptr;
        RaytracingScene *chosenRtScene = nullptr;
        if (rtEnabled) {
            rtResources->updateBottomLevelASResources(worker);

            for (uint32_t i = 0; i < framebufferCount; i++) {
                RenderTargetDrawCall &targetDrawCall = framebufferVector[i].renderTargetDrawCall;
                if (!targetDrawCall.rtScenes.empty()) {
                    chosenFramebuffer = &framebufferVector[i];
                    chosenRtScene = &targetDrawCall.rtScenes[0];
                }
            }
        }

        if (chosenRtScene != nullptr) {
            if (rtResources->updateOutputBuffers) {
                rtResources->createOutputBuffers(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight);
                rtResources->updateOutputBuffers = false;
            }

            const RenderTarget *framebufferTarget = chosenFramebuffer->renderTargetDrawCall.fbStorage->colorTarget;
            interleavedRastersCount = static_cast<uint32_t>(chosenRtScene->interleavedRasters.size());
            rtResources->updateInterleavedRenderTargets(worker, chosenRtScene->screenWidth, chosenRtScene->screenHeight, interleavedRastersCount, framebufferTarget->multisampling, framebufferTarget->usesHDR);

            for (uint32_t i = 0; i < interleavedRastersCount; i++) {
                auto &intRaster = chosenRtScene->interleavedRasters[i];
                RenderTarget *colorTarget = rtResources->interleavedColorTargetVector[i].get();
                RenderTarget *depthTarget = rtResources->interleavedDepthTargetVector[i].get();
                intRaster.colorTextureIndex = getTextureIndex(colorTarget);
                intRaster.depthTextureIndex = getTextureIndex(depthTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(colorTarget);
                chosenFramebuffer->transitionRenderTargetSet.emplace(depthTarget);
            }

            // Must have at least one element in the vector.
            if (chosenRtScene->interleavedRasters.empty()) {
                chosenRtScene->interleavedRasters.emplace_back();
            }

            shaderUploads.push_back({ &rtResources->rtParams, { 0, 1 }, sizeof(interop::RaytracingParams), RenderBufferFlag::CONSTANT, { }, &rtResources->rtParamsBuffer });
            shaderUploads.push_back({ chosenRtScene->interleavedRasters.data(), { 0, chosenRtScene->interleavedRasters.size() }, sizeof(interop::InterleavedRaster), RenderBufferFlag::STORAGE, { }, &interleavedRastersBuffer });

            updateRaytracingScene(worker, *chosenRtScene);
            shaderViewRtEnabled = true;
        }
#   endif

        shaderUploader->submit(worker, shaderUploads);
        updateShaderViews(worker, drawBuffers, outputBuffers, shaderViewRtEnabled);
    }

    void FramebufferRenderer::advanceFrame(bool rtEnabled) {
        frameParams.frameCount++;

#   if RT_ENABLED
        if (rtEnabled) {
            rtResources->swapBuffers = !rtResources->swapBuffers;
            rtResources->skipReprojection = false;
        }
#   endif
    }
};

/*
void RT64::View::renderIm3d() {
    if (Im3d::GetDrawListCount() > 0) {
        commandList->SetGraphicsRootSignature(worker->device->getIm3dRootSignature());

        commandList->SetDescriptorHeaps(1, &descriptorHeap);
        commandList->SetGraphicsRootDescriptorTable(0, descriptorHeap->GetGPUDescriptorHandleForHeapStart());
        commandList->RSSetViewports(1, &rtViewport);
        commandList->RSSetScissorRects(1, &rtScissor);

        unsigned int totalVertexCount = 0;
        for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
            auto &drawList = Im3d::GetDrawLists()[i];
            totalVertexCount += drawList.m_vertexCount;
        }

        if (totalVertexCount > 0) {
            // Release the previous vertex buffer if it should be bigger.
            if (!im3dVertexBuffer.IsNull() && (totalVertexCount > im3dVertexCount)) {
                im3dVertexBuffer.Release();
            }

            // Create the vertex buffer if it's empty.
            const UINT vertexBufferSize = totalVertexCount * sizeof(Im3d::VertexData);
            if (im3dVertexBuffer.IsNull()) {
                CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
                im3dVertexBuffer = worker->device->allocateResource(D3D12_HEAP_TYPE_UPLOAD, &uploadBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr);
                im3dVertexCount = totalVertexCount;
                im3dVertexBufferView.BufferLocation = im3dVertexBuffer.Get()->GetGPUVirtualAddress();
                im3dVertexBufferView.StrideInBytes = sizeof(Im3d::VertexData);
                im3dVertexBufferView.SizeInBytes = vertexBufferSize;
            }

            // Copy data to vertex buffer.
            UINT8 *pDataBegin;
            CD3DX12_RANGE readRange(0, 0);
            D3D12_CHECK(im3dVertexBuffer.Get()->Map(0, &readRange, reinterpret_cast<void **>(&pDataBegin)));
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                size_t copySize = sizeof(Im3d::VertexData) * drawList.m_vertexCount;
                memcpy(pDataBegin, drawList.m_vertexData, copySize);
                pDataBegin += copySize;
            }

            im3dVertexBuffer.Get()->Unmap(0, nullptr);

            unsigned int vertexOffset = 0;
            for (Im3d::U32 i = 0, n = Im3d::GetDrawListCount(); i < n; ++i) {
                auto &drawList = Im3d::GetDrawLists()[i];
                commandList->IASetVertexBuffers(0, 1, &im3dVertexBufferView);
                switch (drawList.m_primType) {
                case Im3d::DrawPrimitive_Points:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStatePoint());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                    break;
                case Im3d::DrawPrimitive_Lines:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateLine());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
                    break;
                case Im3d::DrawPrimitive_Triangles:
                    commandList->SetPipelineState(worker->device->getIm3dPipelineStateTriangle());
                    commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    break;
                default:
                    break;
                }

                commandList->DrawInstanced(drawList.m_vertexCount, 1, vertexOffset, 0);
                vertexOffset += drawList.m_vertexCount;
            }
        }
    }
}
*/

/*
namespace RT64 {
    class View {
    private:
        // Im3D
        AllocatedResource im3dVertexBuffer;
        D3D12_VERTEX_BUFFER_VIEW im3dVertexBufferView;
        unsigned int im3dVertexCount;
    };
};
*/
