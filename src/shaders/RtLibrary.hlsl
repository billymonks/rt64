//
// RT64
//
// Path traced materials for Advanced Lighting.
//
// Hit shaders evaluate the game's real material with the raster shader's own code: the draw's RDP tiles are
// sampled from the texture cache (sets 1 and 2), and the color combiner and alpha compare run on the result.
// Alpha tested and blended materials are non-opaque in the acceleration structure; their any-hit shaders
// reject transparent texels for camera, bounce and shadow rays.
//
// Lighting: one distant point light (sun, moon or room light) with ray traced shadows, plus one diffuse bounce.
// The N64 vertex shade is kept inside the albedo (it is the game's baked lighting and painted color), so the
// light factor is normalized to about 1 on fully lit surfaces to stay close to the original brightness.

#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"

#include "BlueNoise.hlsli"
#include "Color.hlsli"
#include "FbRendererCommon.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"
#include "shaders/FbRendererRT.hlsli"

// Set to true to light the rasterized frame instead of shading path traced materials (the earlier hybrid mode).
static const bool kHybridLighting = false;

// Quality knobs.
// Interpolate the vertex normals of lit geometry for shading (characters look smooth; unlit architecture stays flat).
static const bool kSmoothNormals = true;
// Angular radius of the sun or moon, in radians. Wider gives softer shadows that harden near the caster.
static const float kSunAngularRadius = 0.035;
// How far past facing away (cosine of the face against the sun) a solid face still takes some sunlight, so the
// terminator of a smooth-shaded character stays soft. Beyond it the face only gets ambient and bounce light.
static const float kSunTerminatorBand = 0.25;
// A solid face loses the sun over this much of its (geometric) N.L below zero, and gets none past it.
static const float kSunFaceFade = 0.05;
// Smallest cosine between a smoothed vertex normal and its face's normal that is still trusted for shading.
static const float kMinShadingNormalAgreement = 0.3;
// Architecture (see shadeSurface): triangles at least this long count even outside scenery draws. Their vertex normals
// shade fully smooth while every one is within about 23 degrees of the face, and flat past about 35 degrees.
static const float kArchitectureTriangleEdge = 32.0;
static const float kArchitectureFlatAgreement = 0.82;
static const float kArchitectureSmoothAgreement = 0.92;
// How much sun the back of a two-sided surface (cutout cloth, leaves) passes through.
static const float kBackTranslucency = 0.35;
// A triangle whose edge cross product is below this fraction of its edge lengths' product counts as degenerate.
static const float kDegenerateFaceRatio = 1e-4;
// Specular materials (rt_materials.json). The sun disk is small, so very smooth surfaces would show a pinpoint
// highlight that sparkles from pixel to pixel; its highlight uses at least this roughness.
static const float kSpecularMinRoughness = 0.05;
static const float kSunSpecularMinRoughness = 0.12;
static const float kSunSpecularStrength = 1.0;
// Highest specular light added to a pixel, against fireflies.
static const float kSpecularMaxRadiance = 1.5;
// Surfaces smoother than this trace a mirror reflection, fading in from half of it.
static const float kGlossyReflectionRoughness = 0.4;
// Color of the sky seen in reflections that miss the scene, times the ambient level.
static const float3 kReflectionSkyColor = float3(0.85, 0.92, 1.05);
// How much of a metal's diffuse light turns into specular.
static const float kMetalDiffuseLoss = 0.5;
// Fraction of an emissive light's sprite size used as the radius of the light for soft shadows.
static const float kEmissiveLightRadiusScale = 0.5;
// The light passes write this frame's samples; RtTemporalCS blends them with history and RtDenoiseCS filters them.
// History is reused when the reprojected depth matches within this fraction and the normals agree (RtTemporalCS
// tests the same).
static const float kTemporalDepthTolerance = 0.05;
static const float kTemporalNormalTolerance = 0.8;
// Ray traced ambient occlusion: the length of the occlusion rays in world units. Its strength is in ComposePS.
static const float kAORadius = 30.0;

// Light balance. The light factor is direct + (1 - direct) * fill (see ComposePS): 1 on fully lit surfaces, and
// shadowed surfaces keep close to the scene's shadow tint, lifted a little by the colored light bounced off the
// surface a diffuse ray hits (or the sky).
static const float kFillBounce = 0.35;
static const float kSkyBounce = 0.55;
// Fill strength with the second bounce on. The second bounce adds roughly a third more fill outdoors (half the
// light again from surfaces of about 0.5 albedo, only the sunlit part above the shadow floor), so the strength drops
// to keep sunlit scenes as bright as with one bounce: 0.35 / 1.3.
static const float kFillBounceTwo = 0.27;
// Fill strength in rooms with the sun and moon off: lit only by manual and emissive lights, which the fill otherwise
// leaves thin.
static const float kFillBounceSunless = 0.45;
// A first bounce surface darker than this reflects too little for a second bounce to show.
static const float kSecondBounceMinAlbedo = 0.02;
static const float kLightColorTint = 0.25;
static const float kBounceConeSpread = 0.3;

// Closest hit output. px/py/pz is the world space velocity of the hit point (current minus previous frame
// position), used to reproject history; the hit position itself is origin + direction * hitT. n is the shading
// normal (smooth for lit geometry) and g the geometric normal, both facing the ray.
struct HitPayload {
    float px, py, pz;
    float nx, ny, nz;
    float ar, ag, ab;
    float hitT;
    float hit;
    float renderIndex;
    float fr, fg, fb, fogFactor;
    float gx, gy, gz;
    // Combiner alpha of the material, for the blended water surfaces of the water pass.
    float alpha;
};

// payload.hit of a closest hit on an alpha tested surface; plain hits are 1. Stays below the camera ray markers (2, 3).
static const float kHitTwoSided = 1.25;

// Added to payload.hit for a triangle without a reliable face normal (zero area or a sliver).
static const float kHitDegenerate = 0.125;

bool hitTwoSided(float hit) {
    return (hit - 1.0) > 0.2;
}

bool hitDegenerate(float hit) {
    const float flags = hit - 1.0;
    return (abs(flags - 0.125) < 0.03) || (abs(flags - 0.375) < 0.03);
}

struct ShadowPayload {
    float shadowed;
    // Render index of the surface being lit. Its own draw does not shadow it: the light facing proxy of a billboard
    // would otherwise cut through the sprite it stands in for.
    float selfRenderIndex;
    // 1 when the surface is a camera facing billboard: proxies near it don't shadow it (see kBillboardProxyReach).
    float selfBillboard;
};

// Instance masks, see RtCameraRayMask and RtShadowRayMask. Camera and bounce rays see the drawn geometry; shadow
// rays see the shadow casters, where camera facing billboards are replaced by light facing proxies.
static const uint kCameraRayMask = 0x1;
static const uint kShadowRayMask = 0x2;
// Water surfaces (Advanced Water Rendering), seen only by the water pass, see RtWaterRayMask.
static const uint kWaterRayMask = 0x4;
// Flags above the material class in a draw's gRtMaterials word, see RtMaterialWaterPassFlag: a path traced draw the
// water pass shades as water, and (water found by its scrolling texture) only where its color is blue or blue-green.
static const uint kMaterialWaterPassFlag = 1u << 28;
static const uint kMaterialWaterHueGateFlag = 1u << 29;
// A decal (decal depth mode: paths, windows, wall details), path traced on top of the surface it lies on; and an alpha
// blended one, whose color the camera ray blends over that surface's. See RtMaterialDecalFlag.
static const uint kMaterialDecalFlag = 1u << 30;
static const uint kMaterialDecalBlendFlag = 1u << 31;
// Blended decal texels fainter than this are left out; ones at least this opaque hide the surface below.
static const float kDecalMinAlpha = 0.02;
static const float kDecalOpaqueAlpha = 0.98;
// How far off its surface (along the surface's normal) a decal still counts as lying on it: at least this, or this
// fraction of the hit distance. Along the camera ray that becomes offset / cos(angle to the normal), with the cosine
// floored and the ray interval capped.
static const float kDecalTieMinOffset = 8.0;
static const float kDecalTieRelativeOffset = 0.01;
static const float kDecalTieMinCosine = 0.02;
// Largest offset a decal may learn (see RaytracingResources::decalOffsets), and the longest stretch along the ray that
// is searched, so a grazing ray can't reach decals on other ground far behind the base.
static const float kDecalMaxOffset = 16.0;
static const float kDecalTieMaxWindow = 96.0;
// Material word flags of path traced draws: large draws (scenery, not characters), which take decals drawn before
// them; and, for decals, a learned offset from the base in the metalness byte (bits 8-15, 0 for the default).
static const uint kMaterialSceneryFlag = 1u << 27;
static const uint kMaterialDecalOffsetShift = 8;
// Why a decal candidate wasn't taken, for the F10 decal view and the decal statistics.
static const uint kDecalRejectDistance = 2;
static const uint kDecalRejectOrder = 3;
static const uint kDecalRejectParallel = 4;
// Only shown by the F10 decal view, not counted in the decal statistics (whose slot 5 is kDecalStatBehind).
static const uint kDecalRejectOccluded = 5;

// How far along the base's normal a decal may lie from it: its learned offset, or the default.
float decalOffset(uint decalMaterial, float hitT) {
    const uint learned = (decalMaterial >> kMaterialDecalOffsetShift) & 0xFF;
    const float offset = (learned != 0) ? (float(learned) / 255.0 * kDecalMaxOffset) : kDecalTieMinOffset;
    return max(offset, hitT * kDecalTieRelativeOffset);
}
// Smallest cosine between a decal's face and its base's for the decal to count as lying on it.
static const float kDecalMinParallel = 0.7;
// Decals (see RtDecalRayMask): only the camera ray's decal query sees them.
static const uint kDecalRayMask = 0x8;

// Diagnostics counters in gHitInstanceId, read back by RaytracingResources::readDiagnosticBuffers.
static const uint kDiagInBounds = 0;
static const uint kDiagOutsideViewport = 1;
static const uint kDiagTraced = 2;
static const uint kDiagHits = 3;
static const uint kDiagBadDirection = 4;
static const uint kDiagBadOrigin = 5;
static const uint kDiagSurfaceIgnores = 6;
static const uint kDiagShadowIgnores = 7;
static const uint kDiagMarker = 8;
static const uint kDiagCompared = 9;
static const uint kDiagMismatched = 10;
static const uint kDiagRecordCount = 11;
static const uint kDiagMaxRecords = 5;
static const uint kDiagShadowRays = 12;
static const uint kDiagShadowBlocked = 13;
static const uint kDiagCulledHits = 14;
static const uint kDiagTieBreaks = 15;
static const uint kDiagStraddlingCulled = 16;
static const uint kDiagNearClipped = 17;
static const uint kDiagHistoryTested = 18;
static const uint kDiagHistoryValid = 19;
static const uint kDiagOcclusionSum = 20;
static const uint kDiagOcclusionPixels = 21;
static const uint kDiagSmoothNormalHits = 22;
// Camera hits on triangles without a reliable face normal.
static const uint kDiagDegenerateHits = 23;
// The three vertex positions (as float bits) of the triangle the center camera ray hit, then its degenerate flag.
static const uint kDiagCenterTriangle = 24;
// Camera hits whose shading normal disagrees with the face normal by more than kMinShadingNormalAgreement.
static const uint kDiagNormalDisagreement = 34;
// The center pixel's fill from bounce 1 (3 floats), from bounce 2 (3 floats), the fill strength and the bounce count.
static const uint kDiagBounceFill = 35;
// Water pass: pixels covered by water, then at the center pixel: whether it's water, the water draw's render index,
// its opacity, Fresnel reflectance and depth below the surface (float bits).
static const uint kDiagWaterPixels = 43;
static const uint kDiagWaterCenter = 44;
// Camera hits on decals, and on blended decals blended over the surface below.
static const uint kDiagDecalHits = 49;
static const uint kDiagDecalBlends = 50;
// Per decal slot (RtParams.diagDecal0-5), kDiagDecalStatsStride counters from kDiagDecalStats: camera hits won (0),
// candidates rejected for distance (2), draw order (3) and not lying parallel (4); of the distance rejections, the ones
// behind (5) and in front of (6) the base, the smallest normal distance (7, float bits) and a histogram of normal
// distances (8-15: below 2, 4, 8, 12, 16, 24, 32 and beyond). The smallest distance is stored complemented (~bits).
static const uint kDiagDecalStats = 64;
static const uint kDiagDecalStatsStride = 16;
static const uint kDecalStatHits = 0;
static const uint kDecalStatBehind = 5;
static const uint kDecalStatInFront = 6;
static const uint kDecalStatMinDistance = 7;
static const uint kDecalStatHistogram = 8;

// The center pixel's emission (see surfaceEmission): whether it's emissive, then the emitted color as float bits.
static const uint kDiagEmissiveCenter = 200;
// Pixels whose surface is emissive, this frame.
static const uint kDiagEmissivePixels = 204;
// Marks an emissive pixel in gRefraction's alpha for the upscaler's reactive mask (UpscalerInputsCS), without visibly
// changing the composed color: animated lava has no motion vectors for its texture.
static const float kEmissiveReactiveCoverage = 0.004;

// Light an emissive material (rt_materials.json "emission", lava) gives off at a surface of this albedo: its emission
// color (or the albedo itself) times its strength. Zero for every draw when no draw of the frame is emissive.
float3 surfaceEmission(uint renderIndex, float3 albedo) {
    if (RtParams.emissionOffset == 0) {
        return 0.0;
    }

    const uint packed = gRtMaterials[RtParams.emissionOffset + renderIndex];
    const float strength = float(packed >> 24) / 32.0;
    const uint colorBits = packed & 0xFFFFFF;
    const float3 color = (colorBits != 0) ? (float3(colorBits & 0xFF, (colorBits >> 8) & 0xFF, (colorBits >> 16) & 0xFF) / 255.0) : albedo;
    return color * strength;
}

// Foliage (see RtMaterialFoliageFlag): trees and bushes drawn as upright cutout planes, often two crossed planes.
// Their normal is a soft one around the tree's vertical axis, as if the tree were a round volume: away from the axis
// sideways, towards the viewer in the middle and a little upwards. Every plane of a tree then shades as one smooth
// gradient instead of each plane by its own facing. Light wraps around them (kFoliageWrap), their side away from the
// light still passes some of it (kFoliageTransmission), and a tree doesn't shadow itself.
static const uint kMaterialFoliageFlag = 1u << 26;
// A camera facing billboard with a light facing shadow proxy (see RtMaterialBillboardFlag). Shadow rays from it skip the
// proxies within kBillboardProxyReach: its own, and those of the other draws of the same tree. A proxy turned to the
// light crosses the camera facing sprite along a vertical line, and the half behind it would be shadowed by its own
// tree, a line that moves as the billboard turns.
static const uint kMaterialBillboardFlag = 1u << 25;
static const float kBillboardProxyReach = 48.0;
// The center pixel's vertex shade (float bits), and shadow ray hits on nearby proxies skipped for billboards this frame.
static const uint kDiagCenterShade = 210;
static const uint kDiagBillboardProxySkips = 214;
static const float kFoliageWrap = 0.5;
static const float kFoliageTransmission = 0.6;
static const float kFoliageUp = 0.4;
// The center pixel: whether it's foliage, then the soft normal used (float bits); and the foliage pixels this frame.
static const uint kDiagFoliageCenter = 205;
static const uint kDiagFoliagePixels = 209;

bool isFoliage(float renderIndex) {
    return (renderIndex >= 0.0) && ((gRtMaterials[uint(renderIndex)] & kMaterialFoliageFlag) != 0);
}

// The soft normal of a foliage hit at position, facing the ray. The tree's axis comes from the draw's center
// (RtParams.foliageOffset), or for draws spread wider than a tree, the center of the triangle's own bounds.
float3 foliageNormal(uint renderIndex, float3 position, float3 rayDirection, float3 p0, float3 p1, float3 p2) {
    float2 center = float2(0.0, 0.0);
    float radius = 0.0;
    if (RtParams.foliageOffset != 0) {
        const uint base = RtParams.foliageOffset + renderIndex * 3;
        center = float2(asfloat(gRtMaterials[base + 0]), asfloat(gRtMaterials[base + 1]));
        radius = asfloat(gRtMaterials[base + 2]);
    }

    // Camera facing billboards (radius -1) face the viewer as a whole: the same normal on every draw of a tree, which
    // may be split into a left and a right draw with centers of their own.
    if (radius < 0.0) {
        center = position.xz;
        radius = 1.0;
    }
    else if (!(radius > 0.0)) {
        const float2 minXZ = min(p0.xz, min(p1.xz, p2.xz));
        const float2 maxXZ = max(p0.xz, max(p1.xz, p2.xz));
        center = (minXZ + maxXZ) * 0.5;
        radius = max(max(maxXZ.x - minXZ.x, maxXZ.y - minXZ.y) * 0.5, 1.0);
    }

    float2 radial = (position.xz - center) / radius;
    const float radialLength = length(radial);
    if (radialLength > 1.0) {
        radial /= radialLength;
    }

    const float2 toViewer = -rayDirection.xz;
    const float toViewerLength = length(toViewer);
    const float2 viewerSide = (toViewerLength > 1e-4) ? (toViewer / toViewerLength) : float2(0.0, 0.0);
    const float depth = sqrt(saturate(1.0 - dot(radial, radial)));
    float3 normal = normalize(float3(radial.x + viewerSide.x * depth, kFoliageUp, radial.y + viewerSide.y * depth));
    // Facing the ray, like every other normal: a ray from above or below still sees the near side.
    if (dot(normal, rayDirection) > -0.1) {
        normal = normalize(normal - rayDirection * (dot(normal, rayDirection) + 0.1));
    }

    return normal;
}

int diagDecalSlot(uint renderIndex) {
    const uint id = renderIndex + 1;
    if (id == RtParams.diagDecal0) return 0;
    if (id == RtParams.diagDecal1) return 1;
    if (id == RtParams.diagDecal2) return 2;
    if (id == RtParams.diagDecal3) return 3;
    if (id == RtParams.diagDecal4) return 4;
    if (id == RtParams.diagDecal5) return 5;
    return -1;
}

// Surfaces closer than this to the camera ray's closest hit count as coplanar layers.
static const float kTieBreakMinWindow = 0.02;
static const float kTieBreakRelativeWindow = 5e-4;

// Instance ID layout, see RaytracingResources::updateTopLevelASResources.
static const uint kInstanceRenderIndexMask = 0x1FFFFF;
static const uint kInstanceProxyBit = 1u << 21;
static const uint kInstanceCullingBit = 1u << 22;
static const uint kInstanceAlphaTestBit = 1u << 23;

uint hitRenderIndex() {
    return InstanceID() & kInstanceRenderIndexMask;
}

uint2 rayPixel() {
    return DispatchRaysIndex().xy;
}

bool pixelInBounds(uint2 pix) {
    return (pix.x < uint(RtParams.resolution.x)) && (pix.y < uint(RtParams.resolution.y));
}

bool pixelInViewport(uint2 pix) {
    if ((RtParams.viewport.z < 1.0) || (RtParams.viewport.w < 1.0)) {
        return true;
    }

    float2 p = float2(pix) + 0.5;
    return (p.x >= RtParams.viewport.x) && (p.y >= RtParams.viewport.y) &&
        (p.x < (RtParams.viewport.x + RtParams.viewport.z)) &&
        (p.y < (RtParams.viewport.y + RtParams.viewport.w));
}

bool isDiagPixel(uint2 pix) {
    return all(pix == uint2(RtParams.resolution.xy * 0.5));
}

// cameraV.w is 2 on diagnostics capture frames.
bool diagFrame() {
    return RtParams.cameraV.w > 1.5;
}

void diagCount(uint slot, bool condition) {
    uint count = WaveActiveCountBits(condition);
    if (WaveIsFirstLane() && (count > 0)) {
        InterlockedAdd(gHitInstanceId[slot], count);
    }
}

bool isFinite3(float3 v) {
    return all(abs(v) < 1e30);
}

// The rasterized frame as displayed (sky, background and anything the ray tracer does not shade). It stays at the display
// resolution when the ray traced image is rendered smaller, so the pixel under the ray's sample position is read.
float3 rasterColor(uint2 pix) {
    const float2 samplePosition = (float2(pix) + 0.5 + RtParams.pixelJitter) * float2(RtParams.displayScaleX, RtParams.displayScaleY);
    return gBackgroundColor.Load(int3(uint2(samplePosition), 0)).rgb;
}

float3 loadPos(uint index) {
    uint byteOffset = index * 16;
    // Load3 crashes the D3D12 driver compiler when this shader is linked into a hit group.
    return float3(
        asfloat(posBuffer.Load(byteOffset)),
        asfloat(posBuffer.Load(byteOffset + 4)),
        asfloat(posBuffer.Load(byteOffset + 8)));
}

float3 loadNorm(uint index) {
    uint byteOffset = index * 16;
    return float3(
        asfloat(normBuffer.Load(byteOffset)),
        asfloat(normBuffer.Load(byteOffset + 4)),
        asfloat(normBuffer.Load(byteOffset + 8)));
}

float3 loadVelocity(uint index) {
    uint byteOffset = index * 16;
    return float3(
        asfloat(velBuffer.Load(byteOffset)),
        asfloat(velBuffer.Load(byteOffset + 4)),
        asfloat(velBuffer.Load(byteOffset + 8)));
}

// The RSP lights a vertex when it has lights assigned; only then does it carry a normal instead of a color.
bool vertexIsLit(uint index) {
    const uint word = srcLightCounts.Load(index & ~3u);
    return ((word >> ((index & 3u) * 8u)) & 0xFFu) != 0;
}

float2 loadTexcoord(uint index) {
    uint byteOffset = index * 8;
    return float2(asfloat(genTexCoordBuffer.Load(byteOffset)), asfloat(genTexCoordBuffer.Load(byteOffset + 4)));
}

float4 loadColor(uint index) {
    uint byteOffset = index * 16;
    return float4(
        asfloat(shadedColBuffer.Load(byteOffset)),
        asfloat(shadedColBuffer.Load(byteOffset + 4)),
        asfloat(shadedColBuffer.Load(byteOffset + 8)),
        asfloat(shadedColBuffer.Load(byteOffset + 12)));
}

struct Material {
    float4 color;
    float3 fogColor;
    float fogFactor;
    bool visible;
    // The N64 vertex shade the combiner used (the game's own lighting of lit vertices), for the F9 capture.
    float3 shade;
};

// The raster pixel shader's material path (RasterPS / ShadowCutoutPS) for one ray hit. There are no screen space
// derivatives in ray tracing, so the texture footprint comes from a ray cone: footprintWorld is the width of the
// cone at the hit, divided by the cosine to the surface by the caller.
Material evalMaterial(uint renderIndex, uint primitiveIndex, float2 barycentrics, float footprintWorld) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint instanceIndex = ri.instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    const RDPParams rdp = instanceRDPParams[instanceIndex];

    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const uint i0 = indexBuffer.Load(base * 4);
    const uint i1 = indexBuffer.Load((base + 1) * 4);
    const uint i2 = indexBuffer.Load((base + 2) * 4);
    const float w0 = saturate(1.0 - barycentrics.x - barycentrics.y);
    const float2 uv0 = loadTexcoord(i0);
    const float2 uv1 = loadTexcoord(i1);
    const float2 uv2 = loadTexcoord(i2);
    const float2 uv = uv0 * w0 + uv1 * barycentrics.x + uv2 * barycentrics.y;
    const float4 smoothColor = loadColor(i0) * w0 + loadColor(i1) * barycentrics.x + loadColor(i2) * barycentrics.y;
    // Flat shading takes the color of the first vertex, like the raster pipelines' provoking vertex.
    const float4 flatColor = loadColor(i0);
    const float4 shadeColor = renderFlagSmoothShade(rp.flags) ? smoothColor : float4(flatColor.rgb, smoothColor.a);

    // Texels per world unit on this triangle, from the ratio of its texture and world areas.
    const float3 p0 = loadPos(i0);
    const float worldArea = length(cross(loadPos(i1) - p0, loadPos(i2) - p0));
    const float2 duv1 = uv1 - uv0;
    const float2 duv2 = uv2 - uv0;
    const float texelArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);
    const float texelsPerWorld = sqrt(texelArea / max(worldArea, 1e-12));
    const float footprintTexels = max(footprintWorld * texelsPerWorld, 1.0 / 1024.0);

    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction;
    computeLOD(otherMode, ri.rdpTileCount, rdp.primLOD, 1.0, footprintTexels, footprintTexels, tileIndex0, tileIndex1, lodFraction);
    const float2 ddxUV = float2(footprintTexels, 0.0);
    const float2 ddyUV = float2(0.0, footprintTexels);

    float4 texVal0 = float4(0.0, 0.0, 0.0, 1.0);
    float4 texVal1 = float4(0.0, 0.0, 0.0, 1.0);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = ri.rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        texVal0 = sampleTexture(otherMode, rp.flags, uv, ddxUV, ddyUV, rdpTile, GPUTiles[globalTileIndex], false);
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = ri.rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }

        texVal1 = sampleTexture(otherMode, rp.flags, uv, ddxUV, ddyUV, rdpTile, GPUTiles[globalTileIndex], oneCycleHardwareBug);
    }

    uint randomSeed = initRand(FrParams.frameCount, renderIndex * 7919 + primitiveIndex, 16);
    float4 combinerColor;
    float alphaCompareValue;
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = rdp.primColor;
    ccInputs.shadeColor = shadeColor;
    ccInputs.envColor = rdp.envColor;
    ccInputs.keyCenter = rdp.keyCenter;
    ccInputs.keyScale = rdp.keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = rdp.primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (rdp.convertK[4] / 255.0);
    ccInputs.K5 = (rdp.convertK[5] / 255.0);
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    // Run the blender like the raster pipeline, with its standard fog cycle disabled. Fog is applied after
    // lighting in ComposePS so lit and shadowed surfaces fade to the same fog color.
    Blender::Inputs blInputs;
    blInputs.blendColor = rdp.blendColor;
    blInputs.fogColor = rdp.fogColor;
    blInputs.shadeAlpha = shadeColor.a;

    Material material;
    material.color = float4(Blender::run(otherMode, rp.flags, blInputs, combinerColor, true).rgb, combinerColor.a);
    material.visible = true;
    material.shade = shadeColor.rgb;
    if ((otherMode.alphaCompare() == G_AC_THRESHOLD) && (alphaCompareValue < rdp.blendColor.a)) {
        material.visible = false;
    }
    else if ((otherMode.alphaCompare() == G_AC_DITHER) && (alphaCompareValue < 0.5)) {
        material.visible = false;
    }
    else if (otherMode.cvgXAlpha() && (combinerColor.a < 0.5)) {
        material.visible = false;
    }
    // Blended surfaces in the acceleration structure are treated as cutouts.
    else if (Blender::usesAlphaBlend(otherMode) && (combinerColor.a < 0.5)) {
        material.visible = false;
    }

    material.fogColor = rdp.fogColor.rgb;
    material.fogFactor = Blender::usesStandardFogCycle(otherMode) ? saturate(shadeColor.a) : 0.0;
    return material;
}

// Ray cone inputs: before tracing, payload.hitT holds the cone width at the ray origin and payload.fogFactor its
// spread angle. The closest hit shader overwrites both.
float coneFootprint(float originWidth, float spreadAngle) {
    return originWidth + RayTCurrent() * spreadAngle;
}

// True when the rasterizer culls this triangle for a camera ray. RT64 culls the triangles of draws with any N64 cull
// mode that appear clockwise on screen. For a ray from the camera that is a sign test of the triangle's geometric
// normal against the ray direction, calibrated against the screen space rule on the CPU (cullWindingSign). Unlike a
// projected winding test, it stays defined for triangles that cross the camera plane, such as a wall the camera
// has clipped into.
bool rasterCulls(uint renderIndex, uint primitiveIndex, float3 rayDirection) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const float3 p0 = loadPos(indexBuffer.Load(base * 4));
    const float3 p1 = loadPos(indexBuffer.Load((base + 1) * 4));
    const float3 p2 = loadPos(indexBuffer.Load((base + 2) * 4));
    const bool culled = (dot(cross(p1 - p0, p2 - p0), rayDirection) * RtParams.cullWindingSign) > 0.0;
    if (culled && diagFrame()) {
        // Count culled triangles that cross the camera plane, which the earlier projected test could not handle.
        const float w0 = mul(RtParams.viewProj, float4(p0, 1.0)).w;
        const float w1 = mul(RtParams.viewProj, float4(p1, 1.0)).w;
        const float w2 = mul(RtParams.viewProj, float4(p2, 1.0)).w;
        if ((min(w0, min(w1, w2)) <= 0.0) && (max(w0, max(w1, w2)) > 0.0)) {
            InterlockedAdd(gHitInstanceId[kDiagStraddlingCulled], 1);
        }
    }

    return culled;
}

// Camera rays start with payload.hit = 2 so the any-hit shader can apply the raster's face culling to them only.
// The camera ray's coplanar tie-break starts with payload.hit = 3: every visible surface in the short ray interval
// is enumerated, and the one with the highest render index (drawn last, so on top in the raster) is recorded in
// px (primitive index), py/pz (barycentrics) and nx (hit distance). gx holds the closest hit's distance, gy how far
// from it a surface still counts as coplanar, and gz the same for decals, which may sit a little off their surface.
// With payload.hit = 3.5 decals are skipped, to find the surface under a blended decal.
[shader("anyhit")]
void SurfaceAnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    const uint instanceID = InstanceID();
    const uint renderIndex = instanceID & kInstanceRenderIndexMask;
    if ((payload.hit > 1.5) && ((instanceID & kInstanceCullingBit) != 0) && rasterCulls(renderIndex, PrimitiveIndex(), WorldRayDirection())) {
        if (diagFrame()) {
            InterlockedAdd(gHitInstanceId[kDiagCulledHits], 1);
        }
        IgnoreHit();
    }

    const bool tieBreak = (payload.hit > 2.5);
    const uint rtMaterial = tieBreak ? gRtMaterials[renderIndex] : 0u;
    if ((instanceID & kInstanceAlphaTestBit) != 0) {
        Material material = evalMaterial(renderIndex, PrimitiveIndex(), attr.barycentrics, coneFootprint(payload.hitT, payload.fogFactor));
        // The tie-break keeps the faint texels of blended decals: the camera ray blends them over the surface below.
        const bool faintDecal = tieBreak && ((rtMaterial & kMaterialDecalBlendFlag) != 0) && (material.color.a > kDecalMinAlpha);
        if (!material.visible && !faintDecal) {
            if (diagFrame()) {
                InterlockedAdd(gHitInstanceId[kDiagSurfaceIgnores], 1);
            }
            IgnoreHit();
        }
    }

    if (tieBreak) {
        const bool decal = ((rtMaterial & kMaterialDecalFlag) != 0);
        if (decal) {
            // The camera ray's decal query (see PrimaryRayGen): payload.gx is the base's distance, ar/ag/ab its face
            // normal, fg the cosine between the ray and that normal, nz the base's render index and fb whether the
            // base is scenery (a large draw, not a character). A decal counts if it lies roughly parallel to the base,
            // within its offset along the base's normal, and isn't a decal drawn before a character or small object
            // (characters would pick up the wall decals behind them).
            const float alongRay = RayTCurrent() - payload.gx;
            const float normalDistance = abs(alongRay) * payload.fg;
            const RenderIndices ri = instanceRenderIndices[renderIndex];
            const uint base = ri.faceIndicesStart + PrimitiveIndex() * 3;
            const float3 p0 = loadPos(indexBuffer.Load(base * 4));
            const float3 p1 = loadPos(indexBuffer.Load((base + 1) * 4));
            const float3 p2 = loadPos(indexBuffer.Load((base + 2) * 4));
            const float3 faceCross = cross(p1 - p0, p2 - p0);
            const float faceLength = length(faceCross);
            const float parallelism = (faceLength > 1e-12) ? abs(dot(faceCross / faceLength, float3(payload.ar, payload.ag, payload.ab))) : 0.0;
            const bool parallel = parallelism >= kDecalMinParallel;
            const bool orderAllowed = (float(renderIndex) > payload.nz) || (payload.fb > 0.5);
            const bool within = normalDistance <= decalOffset(rtMaterial, payload.gx);
            uint reason = 0;
            if (!orderAllowed) {
                reason = kDecalRejectOrder;
            }
            else if (!parallel) {
                reason = kDecalRejectParallel;
            }
            else if (!within) {
                reason = kDecalRejectDistance;
            }

            if (reason == 0) {
                if (float(renderIndex) > payload.renderIndex) {
                    payload.renderIndex = float(renderIndex);
                    payload.px = asfloat(PrimitiveIndex());
                    payload.py = attr.barycentrics.x;
                    payload.pz = attr.barycentrics.y;
                    payload.nx = RayTCurrent();
                    payload.ny = ((instanceID & kInstanceAlphaTestBit) != 0) ? 1.0 : 0.0;
                }
            }
            else {
                // The F10 decal view shows the reason of a pixel without a decal.
                payload.fr = float(reason);
                if (diagFrame() || (RtParams.decalCheck != 0)) {
                    const int slot = diagDecalSlot(renderIndex);
                    if (slot >= 0) {
                        const uint stats = kDiagDecalStats + uint(slot) * kDiagDecalStatsStride;
                        InterlockedAdd(gHitInstanceId[stats + reason], 1);
                        if (reason == kDecalRejectDistance) {
                            InterlockedAdd(gHitInstanceId[stats + ((alongRay > 0.0) ? kDecalStatBehind : kDecalStatInFront)], 1);
                            // Complemented, so the cleared counter (0) starts above every distance.
                            InterlockedMax(gHitInstanceId[stats + kDecalStatMinDistance], ~asuint(normalDistance));
                            const uint bin = (normalDistance < 2.0) ? 0 : ((normalDistance < 4.0) ? 1 : ((normalDistance < 8.0) ? 2 : ((normalDistance < 12.0) ? 3 :
                                ((normalDistance < 16.0) ? 4 : ((normalDistance < 24.0) ? 5 : ((normalDistance < 32.0) ? 6 : 7))))));
                            InterlockedAdd(gHitInstanceId[stats + kDecalStatHistogram + bin], 1);
                        }
                    }
                }
            }

            IgnoreHit();
        }

        const bool coplanar = abs(RayTCurrent() - payload.gx) <= payload.gy;
        if (coplanar && (float(renderIndex) > payload.renderIndex)) {
            payload.renderIndex = float(renderIndex);
            payload.px = asfloat(PrimitiveIndex());
            payload.py = attr.barycentrics.x;
            payload.pz = attr.barycentrics.y;
            payload.nx = RayTCurrent();
            payload.ny = ((instanceID & kInstanceAlphaTestBit) != 0) ? 1.0 : 0.0;
        }

        IgnoreHit();
    }
}

[shader("anyhit")]
void ShadowAnyHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // A billboard proxy never shadows the sprite it stands in for, and a tree (foliage) never shadows itself: its crossed
    // planes would cut a hard line across each other.
    if ((float(hitRenderIndex()) == payload.selfRenderIndex) &&
        (((InstanceID() & kInstanceProxyBit) != 0) || ((gRtMaterials[hitRenderIndex()] & kMaterialFoliageFlag) != 0)))
    {
        IgnoreHit();
    }

    // A billboard isn't shadowed by the proxies right next to it: its own tree's, whichever draw of the tree they
    // belong to. Proxies further away (other trees) still shadow it.
    if ((payload.selfBillboard > 0.5) && ((InstanceID() & kInstanceProxyBit) != 0) && (RayTCurrent() < kBillboardProxyReach)) {
        if (diagFrame()) {
            InterlockedAdd(gHitInstanceId[kDiagBillboardProxySkips], 1);
        }
        IgnoreHit();
    }

    // Shadows ignore face culling; only alpha tested materials can let light through.
    if ((InstanceID() & kInstanceAlphaTestBit) == 0) {
        payload.shadowed = 1.0;
        AcceptHitAndEndSearch();
    }

    // Shadow rays carry no cone; a coarse footprint is enough for an alpha test.
    Material material = evalMaterial(hitRenderIndex(), PrimitiveIndex(), attr.barycentrics, RayTCurrent() * RtParams.pixelSpreadAngle * 4.0);
    if (!material.visible) {
        if (diagFrame()) {
            InterlockedAdd(gHitInstanceId[kDiagShadowIgnores], 1);
        }
        IgnoreHit();
    }
    else {
        payload.shadowed = 1.0;
        AcceptHitAndEndSearch();
    }
}

// Fills the payload for a hit on a triangle of a draw. Used by the closest hit shader and by the camera ray's
// coplanar tie-break in PrimaryRayGen.
void shadeSurface(uint renderIndex, uint primitiveIndex, float2 barycentrics, float3 rayOrigin, float3 rayDirection, float hitT, float coneWidth, bool alphaTested, inout HitPayload payload) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const uint i0 = indexBuffer.Load(base * 4);
    const uint i1 = indexBuffer.Load((base + 1) * 4);
    const uint i2 = indexBuffer.Load((base + 2) * 4);
    const float3 p0 = loadPos(i0);
    const float3 p1 = loadPos(i1);
    const float3 p2 = loadPos(i2);
    const float w0 = saturate(1.0 - barycentrics.x - barycentrics.y);

    // Geometric normal facing the ray. A zero area or sliver triangle has no reliable face normal; it takes its shading
    // normal below (or faces the ray when it has none).
    const float3 edge1 = p1 - p0;
    const float3 edge2 = p2 - p0;
    const float3 faceCross = cross(edge1, edge2);
    const float faceCrossLength = length(faceCross);
    const bool degenerate = !all(isfinite(faceCross)) || !(faceCrossLength > (kDegenerateFaceRatio * length(edge1) * length(edge2) + 1e-12));
    float3 geometricNormal = degenerate ? -rayDirection : (faceCross / faceCrossLength);
    if (dot(geometricNormal, rayDirection) > 0.0) {
        geometricNormal = -geometricNormal;
    }

    // Lit N64 geometry (characters, mostly) carries real vertex normals; interpolating them hides the polygon edges.
    // Unlit geometry stores vertex colors where lit geometry stores normals, so it keeps its flat faces, which keeps
    // architecture crisp.
    float3 shadingNormal = geometricNormal;
    if (kSmoothNormals && vertexIsLit(i0) && vertexIsLit(i1) && vertexIsLit(i2)) {
        const float3 smoothNormal = loadNorm(i0) * w0 + loadNorm(i1) * barycentrics.x + loadNorm(i2) * barycentrics.y;
        if ((dot(smoothNormal, smoothNormal) > 1e-6) && all(isfinite(smoothNormal))) {
            float3 candidate = normalize(smoothNormal);
            if (dot(candidate, geometricNormal) < 0.0) {
                candidate = -candidate;
            }

            // Vertex normals that disagree with the face (a seam with reversed normals, a triangle the model's normals
            // don't describe) would light the face from the wrong side; the face's own normal is used instead.
            if (degenerate || (dot(candidate, geometricNormal) >= kMinShadingNormalAgreement)) {
                shadingNormal = candidate;
            }

            // Buildings often share vertex normals across the corners of a box, bending them far from a flat wall's
            // face, which shades the wall in curved patches. On architecture (scenery draws, or large triangles) a
            // triangle whose vertex normals stray far from its face keeps the face; the gentle bends of a round tower
            // stay smooth. The blend is per triangle, so it adds no edge inside one.
            const float maxEdge = sqrt(max(dot(edge1, edge1), max(dot(edge2, edge2), dot(p2 - p1, p2 - p1))));
            if (!degenerate && (((gRtMaterials[renderIndex] & kMaterialSceneryFlag) != 0) || (maxEdge >= kArchitectureTriangleEdge))) {
                float agreement = min(abs(dot(normalize(loadNorm(i0)), geometricNormal)),
                    min(abs(dot(normalize(loadNorm(i1)), geometricNormal)), abs(dot(normalize(loadNorm(i2)), geometricNormal))));
                agreement = isfinite(agreement) ? agreement : 0.0;
                shadingNormal = normalize(lerp(geometricNormal, shadingNormal, smoothstep(kArchitectureFlatAgreement, kArchitectureSmoothAgreement, agreement)));
            }
        }
    }

    if (degenerate) {
        geometricNormal = shadingNormal;
    }

    // Foliage takes one soft normal for both, so its crossed planes shade as a single round shape.
    if ((gRtMaterials[renderIndex] & kMaterialFoliageFlag) != 0) {
        shadingNormal = foliageNormal(renderIndex, rayOrigin + rayDirection * hitT, rayDirection, p0, p1, p2);
        geometricNormal = shadingNormal;
    }

    // Grazing angles stretch the cone footprint across the surface.
    const float footprint = coneWidth / max(abs(dot(geometricNormal, rayDirection)), 0.2);
    Material material = evalMaterial(renderIndex, primitiveIndex, barycentrics, footprint);
    // The center pixel's vertex shade (camera rays arrive with payload.hit = 2), for the F9 capture.
    if ((payload.hit > 1.5) && (payload.hit < 2.5) && diagFrame() && isDiagPixel(DispatchRaysIndex().xy)) {
        gHitInstanceId[kDiagCenterShade + 0] = asuint(material.shade.r);
        gHitInstanceId[kDiagCenterShade + 1] = asuint(material.shade.g);
        gHitInstanceId[kDiagCenterShade + 2] = asuint(material.shade.b);
        gHitInstanceId[kDiagCenterShade + 3] = renderIndex;
    }
    const float3 velocity = loadVelocity(i0) * w0 + loadVelocity(i1) * barycentrics.x + loadVelocity(i2) * barycentrics.y;
    payload.px = velocity.x;
    payload.py = velocity.y;
    payload.pz = velocity.z;
    payload.nx = shadingNormal.x;
    payload.ny = shadingNormal.y;
    payload.nz = shadingNormal.z;
    payload.gx = geometricNormal.x;
    payload.gy = geometricNormal.y;
    payload.gz = geometricNormal.z;
    payload.ar = saturate(material.color.r);
    payload.ag = saturate(material.color.g);
    payload.ab = saturate(material.color.b);
    payload.hitT = hitT;
    // kHitTwoSided marks alpha tested surfaces (sprites, foliage, fences), which are lit from either side, and
    // kHitDegenerate triangles without a reliable face normal.
    payload.hit = 1.0 + (alphaTested ? (kHitTwoSided - 1.0) : 0.0) + (degenerate ? kHitDegenerate : 0.0);
    payload.renderIndex = float(renderIndex);
    payload.fr = material.fogColor.r;
    payload.fg = material.fogColor.g;
    payload.fb = material.fogColor.b;
    payload.fogFactor = material.fogFactor;
    payload.alpha = saturate(material.color.a);
}

[shader("closesthit")]
void SurfaceClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // Camera rays arrive with payload.hit = 2 (see traceSurface).
    const bool cameraRay = (payload.hit > 1.5);
    const float coneWidth = coneFootprint(payload.hitT, payload.fogFactor);
    shadeSurface(hitRenderIndex(), PrimitiveIndex(), attr.barycentrics, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), coneWidth, (InstanceID() & kInstanceAlphaTestBit) != 0, payload);
    if (cameraRay && diagFrame() && isDiagPixel(DispatchRaysIndex().xy)) {
        const RenderIndices ri = instanceRenderIndices[hitRenderIndex()];
        const uint base = ri.faceIndicesStart + PrimitiveIndex() * 3;
        [unroll]
        for (uint v = 0; v < 3; v++) {
            const float3 p = loadPos(indexBuffer.Load((base + v) * 4));
            gHitInstanceId[kDiagCenterTriangle + v * 3 + 0] = asuint(p.x);
            gHitInstanceId[kDiagCenterTriangle + v * 3 + 1] = asuint(p.y);
            gHitInstanceId[kDiagCenterTriangle + v * 3 + 2] = asuint(p.z);
        }

        gHitInstanceId[kDiagCenterTriangle + 9] = hitDegenerate(payload.hit) ? 1u : 0u;
    }
}

[shader("miss")]
void PrimaryMiss(inout HitPayload payload) {
    payload.hit = 0.0;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload) {
    payload.shadowed = 0.0;
}

HitPayload emptyPayload() {
    HitPayload payload = (HitPayload)0;
    payload.renderIndex = -1.0;
    return payload;
}

HitPayload traceSurface(float3 origin, float3 direction, float tMin, float tMax, float coneWidth, float coneSpread, bool cameraRay, uint rayMask) {
    HitPayload payload = emptyPayload();
    payload.hitT = coneWidth;
    payload.fogFactor = coneSpread;
    payload.hit = cameraRay ? 2.0 : 0.0;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    TraceRay(SceneBVH, RAY_FLAG_NONE, rayMask, 0, 0, 0, ray, payload);
    return payload;
}

// Opaque geometry ends the search at once; cutout geometry runs ShadowAnyHit.
float traceShadow(float3 origin, float3 direction, float distance, float selfRenderIndex) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = max(distance, 1.0);

    ShadowPayload payload;
    payload.shadowed = 0.0;
    payload.selfRenderIndex = selfRenderIndex;
    payload.selfBillboard = ((selfRenderIndex >= 0.0) && ((gRtMaterials[uint(selfRenderIndex)] & kMaterialBillboardFlag) != 0)) ? 1.0 : 0.0;
    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kShadowRayMask, 1, 0, 1, ray, payload);
    return payload.shadowed;
}

// Offset along the normal that grows with distance, so float precision far from the camera does not make a
// surface shadow itself.
float surfaceBias(float hitT) {
    return max(0.15, hitT * 0.004);
}

struct LightSample {
    float3 factor;
    float ndotl;
    float visibility;
    float distance;
    float3 direction;
    // The face (geometric normal, the side the camera sees) against the light, and whether the surface is two-sided:
    // what the F10 "lit while facing away" view tests.
    float geometricFacing;
    bool twoSided;
};

// Direct light factor, using the same model as the shadow map pass:
// lerp(facingFloor, 1, |NdotL|) * lerp(shadowTint, 1, visibility), tinted slightly by the light color.
// Like the shadow map pass, a surface facing away from the light is not dark by itself: single sided N64 walls are
// often seen from their unlit side. The shadow ray starts on the side the light comes from, so the back of a closed
// mesh still finds the mesh in the way.
// Where a shadow ray starts. Normally on the side of the geometric surface the light comes from. Past the shadow
// terminator of a smooth-shaded surface (the face turns away from the light but its interpolated normal doesn't),
// it starts along the shading normal instead, so low-poly characters don't get blocky self-shadowing.
// A solid (one-sided) surface facing away from the light never starts its ray behind itself: behind it is the
// inside of the model (a head, a pillar), and a ray from there could slip out through an opening or a grazing edge and
// light the back of the model. It starts on the visible side, so its own surface or the model shadows it. Only
// two-sided surfaces (cutout sprites, foliage, cloth) start behind themselves, to be lit through from the back.
float3 shadowRayOrigin(float3 position, float3 geometricNormal, float3 shadingNormal, float3 lightDirection, float bias, bool twoSided) {
    if (dot(geometricNormal, lightDirection) >= 0.0) {
        return position + geometricNormal * bias;
    }
    // Only a shading normal close to the face may move the ray's start, so it can't leave the surface sideways.
    else if ((dot(shadingNormal, lightDirection) > 0.0) && (dot(shadingNormal, geometricNormal) > 0.5)) {
        return position + shadingNormal * (bias * 2.0);
    }
    else if (twoSided) {
        return position - geometricNormal * bias;
    }
    else {
        return position + geometricNormal * bias;
    }
}

// A direction inside a cone around the axis, from two uniform numbers.
float3 sampleCone(float3 axis, float angularRadius, float2 noise) {
    const float3 bitangent = getPerpendicularVector(axis);
    const float3 tangent = cross(bitangent, axis);
    const float radius = tan(angularRadius) * sqrt(noise.x);
    const float phi = 2.0 * M_PI * noise.y;
    return normalize(axis + (tangent * cos(phi) + bitangent * sin(phi)) * radius);
}

// Blue noise tile of one of the pixel's samples this frame. With n samples per pixel, frame f takes tiles f * n to
// f * n + n - 1, so no two samples of a pixel share a tile until the 64 tiles wrap around; each wrap also shifts the
// tile so the sequence doesn't repeat exactly. One sample per pixel is the plain frame count.
uint noiseFrame(uint sampleIndex) {
    return FrParams.frameCount * max(RtParams.lightSamples, 1u) + sampleIndex;
}

uint2 noiseShift(uint sampleIndex) {
    const uint cycle = (RtParams.lightSamples > 1) ? (noiseFrame(sampleIndex) / 64u) : 0u;
    return uint2(cycle * 13u, cycle * 29u);
}

// Per pixel noise that changes every frame, for the soft shadow and occlusion samples that temporal accumulation
// averages. The channel picks a decorrelated sample set.
float2 pixelNoise(uint2 pix, uint channel, uint sampleIndex) {
    const float3 noise = getBlueNoise(gBlueNoise, pix + noiseShift(sampleIndex) + uint2(channel * 23u, channel * 41u), noiseFrame(sampleIndex) + channel * 17u);
    return noise.xy;
}

// Cosine weighted direction for one of the pixel's samples; offset picks a decorrelated set as the channel does.
float3 pixelCosineDirection(uint2 pix, uint2 offset, uint frameOffset, uint sampleIndex, float3 normal) {
    return getCosHemisphereSampleBlueNoise(gBlueNoise, pix + noiseShift(sampleIndex) + offset, noiseFrame(sampleIndex) + frameOffset, normal);
}

LightSample sampleLight(float3 position, float3 normal, float3 geometricNormal, float hitT, float selfRenderIndex, bool twoSided, float2 noise) {
    LightSample result;
    result.factor = 1.0;
    result.ndotl = 1.0;
    result.visibility = 1.0;
    result.distance = 0.0;
    result.direction = float3(0.0, 1.0, 0.0);
    result.geometricFacing = 1.0;
    result.twoSided = twoSided;
    if (RtParams.emissiveLightStart == 0) {
        return result;
    }

    const bool foliage = (RtParams.foliageOffset != 0) && isFoliage(selfRenderIndex);

    // The room has the sun and moon switched off (rt_lights.json "room settings"): everything gets the level of a
    // surface that neither faces nor sees the light, and no shadow ray is traced. Manual and emissive lights add on top.
    if (RtParams.sunDisabled != 0) {
        result.visibility = 0.0;
        result.ndotl = 0.0;
        result.geometricFacing = -1.0;
        result.factor = RtParams.cameraW.w * RtParams.cameraW.xyz;
        return result;
    }

    PointLight light = SceneLights[0];
    float3 toLight = light.position - position;
    result.distance = length(toLight);
    result.direction = (result.distance > 1e-4) ? (toLight / result.distance) : float3(0.0, 1.0, 0.0);
    result.ndotl = dot(normal, result.direction);

    // Solid surfaces facing away from the sun get no sunlight, past a soft band that keeps the terminator of
    // smooth-shaded characters soft. Without this, the shadow ray of a one-sided wall's unlit side starts behind the
    // wall and finds nothing in the way, so the wall's shadow side lights up in patches wherever its smoothed
    // normals change which side the ray starts on. Alpha tested sprites and foliage stay lit from either side.
    // Both the face and its shading normal must face the light: when they disagree (a reversed vertex normal, a face
    // normal that doesn't match the model), the surface stays in shade rather than letting light through.
    const float geometricFacing = dot(geometricNormal, result.direction);
    const float shadingFacing = result.ndotl;
    // A solid face is lit by its geometry: once its face (the side the camera sees) turns away from the sun it gets no
    // sun at all, after a very narrow fade down to -kSunFaceFade. Smoothed vertex normals averaged across the corners
    // of a box or a helmet can face the sun while the face doesn't, so the shading normal may only darken, never let
    // light onto a face turned away.
    const float facingGate = twoSided ? 1.0 :
        (smoothstep(-kSunFaceFade, 0.0, geometricFacing) * smoothstep(-kSunTerminatorBand, kSunTerminatorBand, shadingFacing));
    result.geometricFacing = geometricFacing;
    result.twoSided = twoSided;
    if (facingGate > 0.0) {
        // Soft shadows: each pixel and frame tests a different point on the sun's disk.
        const float3 shadowDirection = sampleCone(result.direction, kSunAngularRadius, noise);
        const float bias = surfaceBias(hitT);
        result.visibility = facingGate * (1.0 - traceShadow(shadowRayOrigin(position, geometricNormal, normal, shadowDirection, bias, twoSided), shadowDirection, result.distance, selfRenderIndex));
        // The back of a two-sided surface (a cape, leaves) only lets part of the light through. Foliage fades to its
        // transmission smoothly across the side turned from the light, with no line where its soft normal turns away.
        if (foliage) {
            result.visibility *= lerp(kFoliageTransmission, 1.0, saturate(1.0 + geometricFacing * 2.0));
        }
        else if (twoSided && (geometricFacing < 0.0)) {
            result.visibility *= kBackTranslucency;
        }
    }
    else {
        result.visibility = 0.0;
    }

    float3 shadowTint = RtParams.cameraW.xyz;
    float facingFloor = RtParams.cameraW.w;
    float maxColor = max(max(light.diffuseColor.r, light.diffuseColor.g), max(light.diffuseColor.b, 1e-4));
    float3 lightTint = lerp(float3(1.0, 1.0, 1.0), light.diffuseColor / maxColor, kLightColorTint);
    // Two-sided surfaces brighten with the angle on either side; solid ones only on the side facing the light, so
    // the varying smooth normals of a wall's shadow side don't show as patches.
    // Foliage wraps the light around its soft normal, so its shaded side keeps some of it.
    const float facing = foliage ? saturate((result.ndotl + kFoliageWrap) / (1.0 + kFoliageWrap)) :
        (twoSided ? saturate(abs(result.ndotl)) : (saturate(result.ndotl) * smoothstep(-kSunFaceFade, 0.0, geometricFacing)));
    result.factor = lerp(facingFloor.xxx, float3(1.0, 1.0, 1.0), facing) * lerp(shadowTint, float3(1.0, 1.0, 1.0), result.visibility);
    result.factor *= lerp(float3(1.0, 1.0, 1.0), lightTint, result.visibility);
    return result;
}

// Emissive point lights (flames, glows, spell effects). EmissiveColorRayGen samples each source sprite and keeps,
// per light slot, its smoothed color and coverage in gHitNormalFog[slot] and its smoothed glow in
// gHitNormalFog[kEmissiveSlots + slot] (x = glow, y = saturation, z = brightness, w = category). Light falls off as a
// windowed inverse square that reaches zero at the light's range.
static const uint kEmissiveSlots = 64;
static const uint kEmissiveSlotMask = 0xFF;
static const uint kEmissiveResetFlag = 0x100;
static const uint kEmissiveNoSampleFlag = 0x400;
static const uint kEmissiveCategoryShift = 12;
static const uint kEmissiveSampleColorFlag = 0x4000;
static const uint kEmissiveGlowGateFlag = 0x8000;
// A manual light (rt_lights.json) with "casts shadows" false.
static const uint kManualNoShadowFlag = 0x10000;
static const uint kEmissiveCategoryAdditive = 0;
static const uint kEmissiveCategoryBlended = 1;
static const uint kEmissiveCategoryCutout = 2;
static const float kEmissiveStrength = 1.2;
// Per frame blend towards the new sample: quick when brightening so bursts show, slow when dimming so animated
// flames and spells don't strobe.
static const float kEmissiveAttack = 0.35;
static const float kEmissiveDecay = 0.1;
static const uint kEmissiveShadowedLights = 4;

// How much a sprite of this color emits, by category. Additive sprites glow when bright. Blended sprites also need
// saturated color, so grey or brown dust and smoke stay dark. Cutout sprites must look like a flame: bright,
// saturated and warm, so foliage and fences stay dark.
float emissiveGlow(float3 color, uint category) {
    const float brightness = max(color.r, max(color.g, color.b));
    const float darkest = min(color.r, min(color.g, color.b));
    const float saturation = (brightness > 1e-3) ? ((brightness - darkest) / brightness) : 0.0;
    if (category == kEmissiveCategoryAdditive) {
        return smoothstep(0.15, 0.45, brightness);
    }
    else if (category == kEmissiveCategoryBlended) {
        // Spell and light effects in this game are bright tints close to white (saturation around 0.2), while dust
        // and smoke are mid brightness greys and browns, or pure white puffs. Both brightness and a tint are needed.
        return smoothstep(0.7, 0.82, brightness) * smoothstep(0.08, 0.15, saturation);
    }
    else {
        // Flames run from orange to yellow: green well above zero but below red. Pure red flowers, berries and props
        // fail the hue test.
        const float warmth = saturate((color.r - color.b) * 3.0) * saturate((color.r - color.g * 0.6) * 3.0);
        const float flameHue = smoothstep(0.18, 0.32, color.g / max(color.r, 1e-3));
        return smoothstep(0.7, 0.9, brightness) * smoothstep(0.35, 0.6, saturation) * warmth * flameHue;
    }
}

// The category of a slot's last sample, plus 4 when its color is cool (red clearly the weakest channel: water,
// mist, ice), which the CPU uses to tell ambient water features from spells (see kEmissivePersistentFrames).
float emissiveCategoryTag(uint category, float3 color) {
    const float brightness = max(color.r, max(color.g, color.b));
    const bool cool = (color.r < min(color.g, color.b) + 0.02) && (color.r <= brightness - 0.1);
    return float(category) + (cool ? 4.0 : 0.0);
}

float emissiveAttenuation(float distance, float range) {
    const float nearRange = range * 0.25;
    const float inverseSquare = (nearRange * nearRange) / (distance * distance + nearRange * nearRange);
    const float x = saturate(distance / max(range, 1e-3));
    const float window = saturate(1.0 - x * x * x * x);
    return inverseSquare * window * window;
}

// Every emissive light comes from a texture listed in rt_emitters.json, so it shines at its listed intensity. With a
// sampled color it takes the hue of its sprite (normalized, so a faint frame of the animation doesn't dim it).
float3 emissiveRadiance(PointLight light) {
    if ((light.groupBits & kEmissiveSampleColorFlag) == 0) {
        return light.diffuseColor;
    }

    const uint slot = light.groupBits & kEmissiveSlotMask;
    const float4 sprite = gHitNormalFog[slot];
    const float peak = max(sprite.r, max(sprite.g, sprite.b));
    const float3 hue = (peak > 0.02) ? (sprite.rgb / peak) : float3(1.0, 1.0, 1.0);
    // Legacy glow (unlisted untextured effects): as bright as the sampled sprite glows, like before the list.
    if ((light.groupBits & kEmissiveGlowGateFlag) != 0) {
        const float glow = gHitNormalFog[kEmissiveSlots + slot].x;
        return light.diffuseColor * sprite.rgb * glow * saturate(sprite.a * 2.0 + 0.25);
    }

    return light.diffuseColor * hue;
}

// Light from the emissive lights at a surface. With shadows, the strongest kEmissiveShadowedLights lights get a
// shadow ray, and the others take their average visibility. Lights moving in and out of the shadowed set then
// change the result smoothly instead of switching their shadows on and off.
float3 sampleEmissive(float3 position, float3 normal, float3 geometricNormal, float hitT, bool shadows, float2 noise, float selfRenderIndex) {
    float3 total = 0.0;
    float3 strongestLight[kEmissiveShadowedLights];
    float3 strongestDirection[kEmissiveShadowedLights];
    float strongestDistance[kEmissiveShadowedLights];
    float strongestRadius[kEmissiveShadowedLights];
    float strongestLuminance[kEmissiveShadowedLights];
    float3 unshadowed = 0.0;
    [unroll]
    for (uint s = 0; s < kEmissiveShadowedLights; s++) {
        strongestLight[s] = 0.0;
        strongestDirection[s] = float3(0.0, 1.0, 0.0);
        strongestDistance[s] = 0.0;
        strongestRadius[s] = 0.0;
        strongestLuminance[s] = 0.0;
    }

    [loop]
    for (uint i = 0; i < RtParams.emissiveLightCount; i++) {
        const PointLight light = SceneLights[RtParams.emissiveLightStart + i];
        const float3 toLight = light.position - position;
        const float distance = length(toLight);
        if ((distance >= light.attenuationRadius) || (distance < 1e-3)) {
            continue;
        }

        const float3 direction = toLight / distance;
        const float ndotl = saturate(dot(normal, direction));
        const float3 contribution = emissiveRadiance(light) * ndotl * emissiveAttenuation(distance, light.attenuationRadius);

        // Manual fill lights without shadows light through walls: they never take one of the shadow rays.
        if ((light.groupBits & kManualNoShadowFlag) != 0) {
            unshadowed += contribution;
            continue;
        }

        total += contribution;

        // Keep the strongest lights for the shadow rays, sorted by luminance.
        float luminance = dot(contribution, float3(0.3, 0.59, 0.11));
        float3 entryLight = contribution;
        float3 entryDirection = direction;
        float entryDistance = distance;
        float entryRadius = light.pointRadius;
        [unroll]
        for (uint s = 0; s < kEmissiveShadowedLights; s++) {
            if (luminance > strongestLuminance[s]) {
                const float3 swapLight = strongestLight[s];
                const float3 swapDirection = strongestDirection[s];
                const float swapDistance = strongestDistance[s];
                const float swapRadius = strongestRadius[s];
                const float swapLuminance = strongestLuminance[s];
                strongestLight[s] = entryLight;
                strongestDirection[s] = entryDirection;
                strongestDistance[s] = entryDistance;
                strongestRadius[s] = entryRadius;
                strongestLuminance[s] = luminance;
                entryLight = swapLight;
                entryDirection = swapDirection;
                entryDistance = swapDistance;
                entryRadius = swapRadius;
                luminance = swapLuminance;
            }
        }
    }

    if (shadows) {
        const float bias = surfaceBias(hitT);
        float3 tracedLight = 0.0;
        float3 visibleLight = 0.0;
        float tracedLuminance = 0.0;
        float visibleLuminance = 0.0;
        [unroll]
        for (uint s = 0; s < kEmissiveShadowedLights; s++) {
            if (strongestLuminance[s] > 1e-4) {
                // Stop short of the light so the geometry right at the source (a candle holder, the flame sprite
                // itself) doesn't block it.
                const float maxT = max(strongestDistance[s] - max(strongestRadius[s], 0.5), 0.1);
                // Soft shadows: aim at a different point of the light's sphere each pixel and frame.
                const float lightRadius = strongestRadius[s] * kEmissiveLightRadiusScale;
                const float3 shadowDirection = sampleCone(strongestDirection[s], atan2(lightRadius, max(strongestDistance[s], 1e-3)), frac(noise + float2(0.37, 0.61) * float(s)));
                const float visibility = 1.0 - traceShadow(shadowRayOrigin(position, geometricNormal, normal, shadowDirection, bias, false), shadowDirection, maxT, selfRenderIndex);
                tracedLight += strongestLight[s];
                visibleLight += strongestLight[s] * visibility;
                tracedLuminance += strongestLuminance[s];
                visibleLuminance += strongestLuminance[s] * visibility;
            }
        }

        const float averageVisibility = (tracedLuminance > 1e-6) ? (visibleLuminance / tracedLuminance) : 1.0;
        total = visibleLight + (total - tracedLight) * averageVisibility;
    }

    return max(total + unshadowed, 0.0) * kEmissiveStrength;
}

// Texture stats of one draw for the material guess (RtMaterialDatabase): average color and coverage in
// gHitNormalFog[kMaterialProbeStart + probe * 2], luminance deviation and peak brightness in the entry after it.
static const uint kMaterialProbeStart = kEmissiveSlots * 2;

// One thread per emissive light: samples the source sprite's material and blends it into the light's color slot.
// Threads past the emissive lights sample the material probes. Both share a single, non-unrolled sampling loop:
// evalMaterial is the whole texture and combiner path, and every inlined copy of it adds seconds to the driver's
// pipeline compile.
[shader("raygeneration")]
void EmissiveColorRayGen() {
    const uint lightIndex = DispatchRaysIndex().x;
    const bool probe = (lightIndex >= RtParams.emissiveLightCount);
    if (lightIndex >= (RtParams.emissiveLightCount + RtParams.materialProbeCount)) {
        return;
    }

    const PointLight light = SceneLights[RtParams.emissiveLightStart + lightIndex];
    const uint slot = light.groupBits & kEmissiveSlotMask;
    // The source of a fading light isn't drawn this frame; keep its last color.
    if (!probe && ((light.groupBits & kEmissiveNoSampleFlag) != 0)) {
        return;
    }

    const uint category = (light.groupBits >> kEmissiveCategoryShift) & 0x3;
    const uint renderIndex = uint(light.flickerIntensity);
    const uint triangleCount = uint(light.shadowOffset);
    const uint pointsPerTriangle = probe ? 4u : 3u;
    const uint maxSamples = probe ? 64u : 24u;
    const uint triangleStep = max(triangleCount / (probe ? 16u : 8u), 1u);
    const float2 samplePoints[4] = { float2(0.3333, 0.3333), float2(0.6, 0.2), float2(0.2, 0.6), float2(0.15, 0.15) };
    float3 colorSum = 0.0;
    float alphaSum = 0.0;
    float luminanceSum = 0.0;
    float luminanceSquareSum = 0.0;
    float3 plainColorSum = 0.0;
    float plainLuminanceSum = 0.0;
    float plainLuminanceSquareSum = 0.0;
    float3 peakColor = 0.0;
    float peakBrightness = 0.0;
    uint sampleCount = 0;
    const uint sampleTotal = min(((triangleCount + triangleStep - 1) / triangleStep) * pointsPerTriangle, maxSamples);
    [loop]
    for (uint i = 0; i < sampleTotal; i++) {
        const uint t = (i / pointsPerTriangle) * triangleStep;
        const Material material = evalMaterial(renderIndex, t, samplePoints[i % pointsPerTriangle], 1.0);
        const float alpha = saturate(material.color.a);
        const float3 color = saturate(material.color.rgb);
        const float luminance = dot(color, float3(0.299, 0.587, 0.114));
        colorSum += color * alpha;
        alphaSum += alpha;
        luminanceSum += luminance * alpha;
        luminanceSquareSum += luminance * luminance * alpha;
        plainColorSum += color;
        plainLuminanceSum += luminance;
        plainLuminanceSquareSum += luminance * luminance;
        sampleCount++;

        // The brightest clearly visible texel: the core of a mostly transparent spark.
        const float brightness = max(color.r, max(color.g, color.b));
        if ((alpha > 0.3) && (brightness > peakBrightness)) {
            peakBrightness = brightness;
            peakColor = color;
        }
    }

    if (probe) {
        const uint probeSlot = light.groupBits & 0xFF;
        // A texture with every alpha bit clear drawn opaque (most character skins) shows all its colors: averaged unweighted.
        const bool weighted = (alphaSum > 1e-3);
        const float weight = weighted ? alphaSum : float(max(sampleCount, 1u));
        const float3 probeColor = (weighted ? colorSum : plainColorSum) / weight;
        const float luminanceMean = (weighted ? luminanceSum : plainLuminanceSum) / weight;
        const float luminanceVariance = max((weighted ? luminanceSquareSum : plainLuminanceSquareSum) / weight - luminanceMean * luminanceMean, 0.0);
        gHitNormalFog[kMaterialProbeStart + probeSlot * 2] = float4(probeColor, alphaSum / float(max(sampleCount, 1u)));
        gHitNormalFog[kMaterialProbeStart + probeSlot * 2 + 1] = float4(sqrt(luminanceVariance), peakBrightness, float(sampleCount), 1.0);
        return;
    }

    // Halfway between the average and the peak color, so a small bright core counts without one texel deciding.
    const float3 averageColor = (alphaSum > 1e-3) ? (colorSum / alphaSum) : float3(0.0, 0.0, 0.0);
    const float3 representative = (peakBrightness > 0.0) ? lerp(averageColor, peakColor, 0.5) : averageColor;
    const float4 sampled = (alphaSum > 1e-3) ? float4(representative, alphaSum / float(max(sampleCount, 1u))) : float4(0.0, 0.0, 0.0, 0.0);
    const float brightness = max(sampled.r, max(sampled.g, sampled.b));
    const float darkest = min(sampled.r, min(sampled.g, sampled.b));
    const float saturation = (brightness > 1e-3) ? ((brightness - darkest) / brightness) : 0.0;
    const float glow = emissiveGlow(sampled.rgb, category);
    if ((light.groupBits & kEmissiveResetFlag) != 0) {
        gHitNormalFog[slot] = sampled;
        gHitNormalFog[kEmissiveSlots + slot] = float4(glow, saturation, brightness, emissiveCategoryTag(category, sampled.rgb));
        return;
    }

    const float4 previous = gHitNormalFog[slot];
    const float previousGlow = gHitNormalFog[kEmissiveSlots + slot].x;
    const float previousBrightness = max(previous.r, max(previous.g, previous.b));
    const float colorRate = (brightness > previousBrightness) ? kEmissiveAttack : kEmissiveDecay;
    const float glowRate = (glow > previousGlow) ? kEmissiveAttack : kEmissiveDecay;
    gHitNormalFog[slot] = lerp(previous, sampled, colorRate);
    gHitNormalFog[kEmissiveSlots + slot] = float4(lerp(previousGlow, glow, glowRate), saturation, brightness, emissiveCategoryTag(category, sampled.rgb));
}

void writeDiagnostics(float3 origin, float3 direction, HitPayload payload) {
    gHitInstanceId[kDiagMarker] = 0xC0FFEE;
    gHitColor[0] = float4(origin, 1.0);
    gHitColor[1] = float4(direction, payload.hitT);
    gHitColor[2] = float4(payload.px, payload.py, payload.pz, payload.hit);
    gHitColor[3] = RtParams.viewport;
    gHitColor[4] = RtParams.resolution;
    gHitColor[5] = float4(RtParams.nearDist, float(RtParams.lightsCount), RtParams.farDist, payload.renderIndex);
    gHitColor[9] = float4(payload.ar, payload.ag, payload.ab, payload.hit);
    gHitColor[10] = float4(payload.fr, payload.fg, payload.fb, payload.fogFactor);
    if (payload.renderIndex >= 0.0) {
        const RenderIndices ri = instanceRenderIndices[uint(payload.renderIndex)];
        const RenderParams rp = DynamicRenderParams[ri.instanceIndex];
        const OtherMode otherMode = { rp.omL, rp.omH };
        gHitColor[11] = float4(float(ri.instanceIndex), renderFlagUsesTexture0(rp.flags) ? 1.0 : 0.0, renderFlagUsesTexture1(rp.flags) ? 1.0 : 0.0, float(otherMode.alphaCompare()));
        gHitColor[12] = float4(float(ri.rdpTileIndex), float(ri.rdpTileCount), otherMode.cvgXAlpha() ? 1.0 : 0.0, Blender::usesAlphaBlend(otherMode) ? 1.0 : 0.0);
    }
    else {
        gHitColor[11] = -1.0;
        gHitColor[12] = -1.0;
    }

}

uint packColor(float3 color) {
    uint3 c = uint3(saturate(color) * 255.0 + 0.5);
    return c.r | (c.g << 8) | (c.b << 16);
}

// Compares the path traced material with the rasterized pixel of the same surface. Unfogged opaque surfaces should
// match closely; records of a few mismatching pixels (draw, combiner, render mode, first tile) go to the log.
void diagCompareMaterial(uint2 pix, HitPayload payload, float3 raster) {
    const bool comparable = (payload.hit > 0.5) && (payload.fogFactor < 0.05) && (payload.renderIndex >= 0.0);
    const float3 albedo = float3(payload.ar, payload.ag, payload.ab);
    const float3 delta = abs(albedo - raster);
    const bool mismatch = comparable && (max(delta.r, max(delta.g, delta.b)) > 0.2);
    diagCount(kDiagCompared, comparable);
    diagCount(kDiagMismatched, mismatch);
    if (!mismatch || ((pix.x % 16) != 8) || ((pix.y % 16) != 8)) {
        return;
    }

    uint record;
    InterlockedAdd(gHitInstanceId[kDiagRecordCount], 1, record);
    if (record >= kDiagMaxRecords) {
        return;
    }

    const uint renderIndex = uint(payload.renderIndex);
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const RenderParams rp = DynamicRenderParams[ri.instanceIndex];
    const RDPTile tile = RDPTiles[ri.rdpTileIndex];
    const GPUTile gpuTile = GPUTiles[ri.rdpTileIndex];
    const uint tileBits = uint(tile.fmt & 0xF) | (uint(tile.siz & 0xF) << 4) | (uint(tile.palette & 0xF) << 8) | (uint(tile.cms & 0xF) << 12) | (uint(tile.cmt & 0xF) << 16) | ((ri.rdpTileCount & 0xF) << 20);
    gHitVelocityDistance[record * 3 + 0] = float4(float(renderIndex), asfloat(rp.ccL), asfloat(rp.ccH), asfloat(rp.omL));
    gHitVelocityDistance[record * 3 + 1] = float4(asfloat(rp.omH), asfloat(uint(rp.flags)), asfloat(packColor(albedo)), asfloat(packColor(raster)));
    gHitVelocityDistance[record * 3 + 2] = float4(asfloat(tileBits), asfloat(uint(gpuTile.flags)), asfloat(gpuTile.textureIndex), asfloat(pix.x | (pix.y << 16)));
}

// The camera ray through a pixel. viewI is the inverse of the raster view-projection (row-vector math on the CPU,
// so mul(M, v) here). The camera is the point that projects to clip (0, 0, c, 0); the direction goes through the
// point at NDC depth 0, which is in front of the camera for both depth conventions.
bool cameraRay(uint2 pix, out float3 origin, out float3 direction) {
    float2 p = float2(pix) + 0.5 + RtParams.pixelJitter;
    float2 vpSize = max(RtParams.viewport.zw, float2(1.0, 1.0));
    float2 vp = (p - RtParams.viewport.xy) / vpSize;
    float ndcX = vp.x * 2.0 - 1.0;
    // nearDist carries the sign of the viewport Y scale: +1 when the top of the viewport is clip Y = +1.
    float clipY = -lerp(-RtParams.nearDist, RtParams.nearDist, vp.y);
    float4 midH = mul(RtParams.viewI, float4(ndcX, clipY, 0.0, 1.0));
    float3 midPos = midH.xyz / midH.w;
    float4 originH = mul(RtParams.viewI, float4(0.0, 0.0, 1.0, 0.0));
    if (abs(originH.w) > 1e-12) {
        origin = originH.xyz / originH.w;
    }
    else {
        float4 nearH = mul(RtParams.viewI, float4(ndcX, clipY, -1.0, 1.0));
        origin = nearH.xyz / nearH.w;
    }

    direction = midPos - origin;
    float len2 = dot(direction, direction);
    diagCount(kDiagBadOrigin, !isFinite3(origin));
    if (!isFinite3(origin) || !isFinite3(direction) || (len2 <= 1e-12)) {
        return false;
    }

    direction *= rsqrt(len2);
    return true;
}

// Distance along the camera ray to the raster's near plane (NDC depth -1; the game's projections are OpenGL style).
// The rasterizer clips geometry closer than that, so camera rays start there.
float cameraNearT(uint2 pix, float3 origin, float3 direction) {
    float2 p = float2(pix) + 0.5 + RtParams.pixelJitter;
    float2 vpSize = max(RtParams.viewport.zw, float2(1.0, 1.0));
    float2 vp = (p - RtParams.viewport.xy) / vpSize;
    float ndcX = vp.x * 2.0 - 1.0;
    float clipY = -lerp(-RtParams.nearDist, RtParams.nearDist, vp.y);
    float4 nearH = mul(RtParams.viewI, float4(ndcX, clipY, -1.0, 1.0));
    if (abs(nearH.w) <= 1e-12) {
        return 1e-2;
    }

    float nearT = dot(nearH.xyz / nearH.w - origin, direction);
    return (isfinite(nearT) && (nearT > 1e-2)) ? nearT : 1e-2;
}

// The light factor of a surface that neither faces the light nor sees it: the scene's facing floor times its
// shadow tint.
float ambientBase() {
    return RtParams.cameraW.w * dot(RtParams.cameraW.xyz, float3(1.0, 1.0, 1.0) / 3.0);
}

void writeMiss(uint2 pix, float3 bg) {
    gDiffuse[pix] = float4(bg, 0.0);
    gShadingPosition[pix] = 0.0;
    gShadingNormal[pix] = 0.0;
    gShadingSpecular[pix] = 0.0;
    gViewDirection[pix] = 0.0;
    gDepth[pix] = 0.0;
    gInstanceId[pix] = -1;
    gNormalRoughness[pix] = 0.0;
}

[shader("raygeneration")]
void PrimaryRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    diagCount(kDiagInBounds, true);
    float3 bg = rasterColor(pix);
    gFlow[pix] = 0.0;
    gTransparent[pix] = 0.0;
    gReflection[pix] = 0.0;
    gRefraction[pix] = 0.0;
    const bool insideViewport = pixelInViewport(pix);
    diagCount(kDiagOutsideViewport, !insideViewport);
    float3 origin, direction;
    const bool validRay = insideViewport && cameraRay(pix, origin, direction);
    diagCount(kDiagBadDirection, insideViewport && !validRay);
    if (!validRay) {
        writeMiss(pix, bg);
        return;
    }

    const float nearT = cameraNearT(pix, origin, direction);
    HitPayload payload = traceSurface(origin, direction, nearT, 1.0e7, nearT * RtParams.pixelSpreadAngle, RtParams.pixelSpreadAngle, true, kCameraRayMask);
    diagCount(kDiagTraced, true);
    diagCount(kDiagHits, payload.hit > 0.5);
    if (diagFrame()) {
        // Pixels where a surface closer than the near plane was skipped, as the rasterizer clips it.
        const HitPayload nearHit = traceSurface(origin, direction, 1e-2, nearT, 0.0, RtParams.pixelSpreadAngle, true, kCameraRayMask);
        diagCount(kDiagNearClipped, nearHit.hit > 0.5);
    }

    // Coplanar tie-break. Layered character parts (faces, eyes, clothing details) lie on the surface of the base
    // mesh; the raster shows whichever the game drew last, while the closest hit would pick either by float noise and
    // flicker between frames. Enumerate the surfaces just around the hit and keep the latest draw.
    bool tieBreakChanged = false;
    bool decalApplied = false;
    bool decalBlended = false;
    float decalDebug = 0.0;
    if (payload.hit > 0.5) {
        const float window = max(kTieBreakMinWindow, payload.hitT * kTieBreakRelativeWindow);
        const float centerT = payload.hitT;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = max(centerT - window, nearT);
        ray.TMax = centerT + window;
        HitPayload tie = emptyPayload();
        tie.hit = 3.0;
        tie.hitT = 0.0;
        tie.fogFactor = RtParams.pixelSpreadAngle;
        tie.gx = centerT;
        tie.gy = window;
        tie.gz = window;
        TraceRay(SceneBVH, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kCameraRayMask, 0, 0, 0, ray, tie);
        tieBreakChanged = (tie.renderIndex > payload.renderIndex);

        // Decal resolve. Decals (dirt paths, windows, wall details) are only in the decal mask, so camera, bounce and
        // shadow rays never see them. The surface the camera ray hit is the base; a second query along the same ray,
        // decals only, looks for one lying roughly parallel to the base within its offset along the base's normal (the
        // N64 decal depth test allows paths well off the ground; along a ray grazing the ground that distance
        // stretches by 1 / cos). See SurfaceAnyHit for the rules. Its material replaces or blends over the base's,
        // while the position, normals and so the lighting and shadows stay the base's.
        // Layer 0 shades the tie-break winner if it isn't the closest hit, layer 1 the decal; both share one
        // shadeSurface call, which keeps the shader small.
        const bool decalCheck = diagFrame() || (RtParams.decalCheck != 0) || (RtParams.debugView == 8);
        HitPayload layerHit = tie;
        [loop]
        for (uint layer = 0; layer < 2; layer++) {
            if (layer == 1) {
                if (RtParams.decalsPresent == 0) {
                    break;
                }

                const float3 baseGeometric = float3(payload.gx, payload.gy, payload.gz);
                const float grazing = max(abs(dot(direction, baseGeometric)), kDecalTieMinCosine);
                // Search as far as the largest offset a decal may have; checks look twice as far, to measure how far
                // off the rejected decals lie.
                const float searchOffset = max(kDecalMaxOffset, centerT * kDecalTieRelativeOffset) * (decalCheck ? 2.0 : 1.0);
                const float searchWindow = max(min(searchOffset / grazing, kDecalTieMaxWindow * (decalCheck ? 2.0 : 1.0)), window);
                RayDesc decalRay;
                decalRay.Origin = origin;
                decalRay.Direction = direction;
                decalRay.TMin = max(centerT - searchWindow, nearT);
                decalRay.TMax = centerT + searchWindow;
                layerHit = emptyPayload();
                layerHit.hit = 3.0;
                layerHit.hitT = 0.0;
                layerHit.fogFactor = RtParams.pixelSpreadAngle;
                layerHit.gx = centerT;
                layerHit.gy = window;
                layerHit.ar = baseGeometric.x;
                layerHit.ag = baseGeometric.y;
                layerHit.ab = baseGeometric.z;
                layerHit.fg = grazing;
                layerHit.nz = payload.renderIndex;
                layerHit.fb = ((gRtMaterials[uint(payload.renderIndex)] & kMaterialSceneryFlag) != 0) ? 1.0 : 0.0;
                layerHit.fr = 0.0;
                TraceRay(SceneBVH, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kDecalRayMask, 0, 0, 0, decalRay, layerHit);
                decalDebug = layerHit.fr;
                if (layerHit.renderIndex < 0.0) {
                    break;
                }

                // A decal behind the base belongs to another surface when one lies between the base and the decal, or
                // as far past the decal again (the floor under a rug seen through a table top): the base hides it.
                const float behindBase = layerHit.nx - centerT;
                if (behindBase > window) {
                    RayDesc occluderRay;
                    occluderRay.Origin = origin;
                    occluderRay.Direction = direction;
                    occluderRay.TMin = centerT + window;
                    occluderRay.TMax = layerHit.nx + behindBase;
                    HitPayload occluder = emptyPayload();
                    occluder.hit = 1.0;
                    occluder.hitT = centerT * RtParams.pixelSpreadAngle;
                    occluder.fogFactor = RtParams.pixelSpreadAngle;
                    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kCameraRayMask, 0, 0, 0, occluderRay, occluder);
                    if (occluder.hit > 0.5) {
                        decalDebug = float(kDecalRejectOccluded);
                        break;
                    }
                }
            }
            else if (!tieBreakChanged) {
                continue;
            }

            HitPayload shaded = payload;
            shadeSurface(uint(layerHit.renderIndex), asuint(layerHit.px), float2(layerHit.py, layerHit.pz), origin, direction, layerHit.nx, layerHit.nx * RtParams.pixelSpreadAngle, layerHit.ny > 0.5, shaded);
            if (layer == 0) {
                payload = shaded;
                continue;
            }

            // A blended decal shows the base through its transparent parts (the soft edges of a path).
            const uint decalMaterial = gRtMaterials[uint(shaded.renderIndex)];
            const float decalAlpha = ((decalMaterial & kMaterialDecalBlendFlag) != 0) ? saturate(shaded.alpha) : 1.0;
            payload.ar = lerp(payload.ar, shaded.ar, decalAlpha);
            payload.ag = lerp(payload.ag, shaded.ag, decalAlpha);
            payload.ab = lerp(payload.ab, shaded.ab, decalAlpha);
            if (decalAlpha >= 0.5) {
                payload.renderIndex = shaded.renderIndex;
                payload.fr = shaded.fr;
                payload.fg = shaded.fg;
                payload.fb = shaded.fb;
                payload.fogFactor = shaded.fogFactor;
            }

            decalApplied = true;
            decalBlended = (decalAlpha < kDecalOpaqueAlpha);
            decalDebug = 1.0;
            if (decalCheck) {
                const int slot = diagDecalSlot(uint(shaded.renderIndex));
                if (slot >= 0) {
                    InterlockedAdd(gHitInstanceId[kDiagDecalStats + uint(slot) * kDiagDecalStatsStride + kDecalStatHits], 1);
                }
            }
        }

        payload.hitT = centerT;
    }

    if (diagFrame()) {
        diagCount(kDiagTieBreaks, tieBreakChanged);
        diagCount(kDiagDecalHits, decalApplied);
        diagCount(kDiagDecalBlends, decalBlended);
        const bool foliageHit = (payload.hit > 0.5) && (RtParams.foliageOffset != 0) && isFoliage(payload.renderIndex);
        diagCount(kDiagFoliagePixels, foliageHit);
        if (isDiagPixel(pix)) {
            gHitInstanceId[kDiagFoliageCenter + 0] = foliageHit ? 1u : 0u;
            gHitInstanceId[kDiagFoliageCenter + 1] = asuint(payload.nx);
            gHitInstanceId[kDiagFoliageCenter + 2] = asuint(payload.ny);
            gHitInstanceId[kDiagFoliageCenter + 3] = asuint(payload.nz);
        }
    }
    if (isDiagPixel(pix)) {
        writeDiagnostics(origin, direction, payload);
    }

    if (diagFrame()) {
        diagCompareMaterial(pix, payload, bg);
    }

    if (payload.hit > 0.5) {
        const float3 normal = float3(payload.nx, payload.ny, payload.nz);
        const float3 geometricNormal = float3(payload.gx, payload.gy, payload.gz);
        const float3 position = origin + direction * payload.hitT;
        float3 albedo = kHybridLighting ? bg : float3(payload.ar, payload.ag, payload.ab);
        gDiffuse[pix] = float4(albedo, 1.0);
        // World space velocity of the hit point, for reprojecting history.
        gShadingPosition[pix] = float4(payload.px, payload.py, payload.pz, 1.0);
        gShadingNormal[pix] = float4(normal, (hitTwoSided(payload.hit) ? 1.0 : 0.0) + (hitDegenerate(payload.hit) ? 2.0 : 0.0));
        // x: the decal resolve at this pixel for the F10 decal view (1 decal shown, 2 decal just outside the window).
        gShadingSpecular[pix] = float4(decalDebug, 0.0, 0.0, 0.0);
        gViewDirection[pix] = float4(-direction, 0.0);
        gDepth[pix] = payload.hitT;
        gInstanceId[pix] = int(payload.renderIndex);
        // Geometric normal and view depth, compared against next frame's reprojection.
        gNormalRoughness[pix] = float4(geometricNormal * 0.5 + 0.5, mul(RtParams.viewProj, float4(position, 1.0)).w);
        // The ambient part of the light factor, which ComposePS darkens with ambient occlusion.
        gFlow[pix] = float2(ambientBase(), 0.0);
        // Fog color and amount of the surface's blender, applied after lighting in ComposePS.
        gTransparent[pix] = kHybridLighting ? float4(0.0, 0.0, 0.0, 0.0) : float4(payload.fr, payload.fg, payload.fb, payload.fogFactor);
        if (diagFrame()) {
            diagCount(kDiagSmoothNormalHits, dot(normal, geometricNormal) < 0.9999);
            diagCount(kDiagDegenerateHits, hitDegenerate(payload.hit));
            diagCount(kDiagNormalDisagreement, dot(normal, geometricNormal) < kMinShadingNormalAgreement);
        }
    }
    else {
        writeMiss(pix, bg);
    }
}

// Rebuilds the primary hit at full precision from the 32-bit hit distance; the shading position texture is 16-bit.
bool primaryHit(uint2 pix, out float3 position, out float3 normal, out float3 geometricNormal, out float3 velocity, out float hitT, out bool twoSided) {
    position = 0.0;
    normal = float3(0.0, 1.0, 0.0);
    geometricNormal = normal;
    velocity = 0.0;
    hitT = 0.0;
    twoSided = false;
    float3 origin, direction;
    if ((gDiffuse[pix].a <= EPSILON) || !cameraRay(pix, origin, direction)) {
        return false;
    }

    hitT = gDepth[pix];
    position = origin + direction * hitT;
    const float4 shadingNormal = gShadingNormal[pix];
    normal = shadingNormal.xyz;
    twoSided = (fmod(shadingNormal.w, 2.0) > 0.5);
    float normalLen2 = dot(normal, normal);
    normal = (normalLen2 > 1e-8) ? (normal * rsqrt(normalLen2)) : -direction;
    geometricNormal = gNormalRoughness[pix].xyz * 2.0 - 1.0;
    float geometricLen2 = dot(geometricNormal, geometricNormal);
    geometricNormal = (geometricLen2 > 1e-8) ? (geometricNormal * rsqrt(geometricLen2)) : normal;
    velocity = gShadingPosition[pix].xyz;
    return true;
}

// Finds this surface point in last frame's image: moves it back by its velocity, projects it with last frame's
// view-projection, and accepts the history there when the depth and normal stored last frame agree.
bool reprojectHistory(float3 position, float3 velocity, float3 geometricNormal, out int2 previousPixel) {
    previousPixel = int2(0, 0);
    // The lighting changed at once (the sun switched off or on with the room): start over rather than fade.
    if (RtParams.historyReset != 0) {
        return false;
    }

    const float4 clip = mul(RtParams.prevViewProj, float4(position - velocity, 1.0));
    if (clip.w <= 1e-4) {
        return false;
    }

    const float2 ndc = clip.xy / clip.w;
    const float2 vp = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y / (2.0 * RtParams.nearDist));
    const float2 p = RtParams.viewport.xy + vp * RtParams.viewport.zw;
    previousPixel = int2(floor(p));
    if (any(previousPixel < 0) || (previousPixel.x >= int(RtParams.resolution.x)) || (previousPixel.y >= int(RtParams.resolution.y))) {
        return false;
    }

    const float4 previous = gPrevNormalRoughness[previousPixel];
    if (previous.w <= 0.0) {
        return false;
    }

    const bool depthMatches = abs(previous.w - clip.w) < (clip.w * kTemporalDepthTolerance + 0.5);
    const bool normalMatches = dot(previous.xyz * 2.0 - 1.0, geometricNormal) > kTemporalNormalTolerance;
    return depthMatches && normalMatches;
}

float colorLuminance(float3 color) {
    return dot(color, float3(0.3, 0.59, 0.11));
}

// This frame's direct light, the average of RtParams.lightSamples independent samples, goes to gFilteredDirectLight
// (alpha 1 on surfaces, 0 where the camera ray missed). RtTemporalCS blends it with history into gDirectLightAccum.
[shader("raygeneration")]
void DirectRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    float3 position, normal, geometricNormal, velocity;
    float hitT;
    bool twoSided;
    if (!primaryHit(pix, position, normal, geometricNormal, velocity, hitT, twoSided)) {
        gFilteredDirectLight[pix] = float4(1.0, 1.0, 1.0, 0.0);
        if (isDiagPixel(pix)) {
            gHitColor[13] = float4(0.0, 0.0, 0.0, 1.0);
        }
        return;
    }

    // The first sample's terms feed the debug views and the diagnostics.
    const float selfRenderIndex = float(gInstanceId[pix]);
    const uint sampleCount = clamp(RtParams.lightSamples, 1u, 4u);
    LightSample light = (LightSample)0;
    float3 emissive = 0.0;
    float3 current = 0.0;
    [loop]
    for (uint s = 0; s < sampleCount; s++) {
        const LightSample sampled = sampleLight(position, normal, geometricNormal, hitT, selfRenderIndex, twoSided, pixelNoise(pix, 0, s));
        const float3 sampledEmissive = kHybridLighting ? float3(0.0, 0.0, 0.0) : sampleEmissive(position, normal, geometricNormal, hitT, true, pixelNoise(pix, 1, s), selfRenderIndex);
        if (s == 0) {
            light = sampled;
            emissive = sampledEmissive;
        }

        current += sampled.factor + sampledEmissive;
    }

    current /= float(sampleCount);
    gFilteredDirectLight[pix] = float4(current, 1.0);

    // F10 debug views: the color replaces the image (white material, no ambient, fill, specular or fog).
    if (RtParams.debugView != 0) {
        float3 debugColor = 0.5;
        const float faceToSun = dot(geometricNormal, light.direction);
        switch (RtParams.debugView) {
        case 1: debugColor = geometricNormal * 0.5 + 0.5; break;
        case 2: debugColor = normal * 0.5 + 0.5; break;
        case 3: debugColor = (dot(normal, geometricNormal) < kMinShadingNormalAgreement) ? float3(1.0, 0.0, 0.0) : float3(0.5, 0.5, 0.5); break;
        case 4: debugColor = light.visibility.xxx; break;
        case 5: debugColor = light.factor; break;
        case 6: {
            const uint id = uint(gInstanceId[pix]) * 2654435761u;
            debugColor = float3(float((id >> 8) & 0xFF), float((id >> 16) & 0xFF), float((id >> 24) & 0xFF)) / 255.0;
            break;
        }
        case 7:
            // From the same values the lighting used: sun reaching a solid face turned away from it.
            debugColor = light.twoSided ? float3(0.2, 0.4, 1.0) :
                (((light.geometricFacing < -kSunFaceFade) && (light.visibility > 0.02)) ? float3(1.0, 0.0, 0.0) : (0.5 * light.factor));
            break;
        case 8: {
            // Decal resolve: green where a decal is shown; where one was found but not taken, yellow when it lies too
            // far from the base, red when it was drawn before a character or small object, blue when not parallel,
            // magenta when the base hides it.
            const uint decalState = uint(gShadingSpecular[pix].x + 0.5);
            debugColor = (decalState == 1) ? float3(0.0, 0.9, 0.2) : ((decalState == kDecalRejectDistance) ? float3(1.0, 0.9, 0.0) :
                ((decalState == kDecalRejectOrder) ? float3(1.0, 0.1, 0.1) : ((decalState == kDecalRejectParallel) ? float3(0.2, 0.4, 1.0) :
                ((decalState == kDecalRejectOccluded) ? float3(0.9, 0.2, 0.9) : float3(0.4, 0.4, 0.4)))));
            break;
        }
        }

        gFilteredDirectLight[pix] = float4(debugColor, 1.0);
        gDiffuse[pix] = float4(1.0, 1.0, 1.0, 1.0);
        gFlow[pix] = float2(0.0, 0.0);
        gTransparent[pix] = float4(0.0, 0.0, 0.0, 0.0);
    }
    if (diagFrame()) {
        int2 previousPixel;
        const bool historyValid = reprojectHistory(position, velocity, geometricNormal, previousPixel);
        diagCount(kDiagShadowRays, RtParams.emissiveLightStart > 0);
        diagCount(kDiagShadowBlocked, (RtParams.emissiveLightStart > 0) && (light.visibility < 0.5));
        diagCount(kDiagHistoryTested, true);
        diagCount(kDiagHistoryValid, historyValid);
    }

    if (isDiagPixel(pix)) {
        gHitColor[13] = float4(light.ndotl, light.visibility, light.distance, light.factor.g);
        // A ray to the center of the light, finding the closest blocker.
        HitPayload blocker = traceSurface(shadowRayOrigin(position, geometricNormal, normal, light.direction, surfaceBias(hitT), twoSided), light.direction, 0.0, light.distance, 0.0, RtParams.pixelSpreadAngle, false, kShadowRayMask);
        gHitColor[14] = float4(normal, blocker.renderIndex);
        gHitColor[15] = float4(light.direction, (blocker.hit > 0.5) ? blocker.hitT : -1.0);
        // The center surface's sides and what lights it, for the F9 "center surface" lines.
        gHitColor[6] = float4(geometricNormal, twoSided ? 1.0 : 0.0);
        gHitColor[7] = float4(emissive, dot(geometricNormal, light.direction));
    }
}

// One diffuse bounce with the bounce surface's real material and its own shadowed direct light, and a short ambient
// occlusion ray. This frame's fill and occlusion, averaged over RtParams.lightSamples samples, go to
// gFilteredIndirectLight; RtTemporalCS blends them with history into gIndirectLightAccum.
[shader("raygeneration")]
void IndirectRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    float3 position, normal, geometricNormal, velocity;
    float hitT;
    bool twoSided;
    if (kHybridLighting || !primaryHit(pix, position, normal, geometricNormal, velocity, hitT, twoSided)) {
        // No fill and no occlusion.
        gFilteredIndirectLight[pix] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float bias = surfaceBias(hitT);
    const float3 origin = position + geometricNormal * bias;
    // Open sky, unless the room's sun and moon are off: N64 interiors are often open above or behind the camera, and
    // the sky would light a room meant to be lit only by its own lights.
    const float skyLight = (RtParams.sunDisabled != 0) ? 0.0 : kSkyBounce;
    const uint sampleCount = clamp(RtParams.lightSamples, 1u, 4u);
    float3 bounceLight = 0.0;
    float3 secondLight = 0.0;
    float occlusion = 0.0;
    [loop]
    for (uint s = 0; s < sampleCount; s++) {
        float3 bounceDir = pixelCosineDirection(pix, uint2(0u, 0u), 0u, s, normal);
        if (dot(bounceDir, geometricNormal) <= 0.0) {
            bounceDir = reflect(bounceDir, geometricNormal);
        }

        // Bounce rays are blurry, so they sample coarse texture levels.
        HitPayload bounce = traceSurface(origin, bounceDir, 0.0, 1.0e6, hitT * RtParams.pixelSpreadAngle, kBounceConeSpread, false, kCameraRayMask);
        if (bounce.hit > 0.5) {
            const float3 bouncePos = origin + bounceDir * bounce.hitT;
            const float3 bounceNormal = float3(bounce.nx, bounce.ny, bounce.nz);
            const float3 bounceGeometricNormal = float3(bounce.gx, bounce.gy, bounce.gz);
            const bool bounceTwoSided = hitTwoSided(bounce.hit);
            LightSample bounceSample = sampleLight(bouncePos, bounceNormal, bounceGeometricNormal, hitT + bounce.hitT, bounce.renderIndex, bounceTwoSided, pixelNoise(pix, 2, s));
            // Emissive lights at the bounce point, without shadow rays to keep the bounce cheap.
            const float3 bounceEmissive = sampleEmissive(bouncePos, bounceNormal, bounceGeometricNormal, hitT + bounce.hitT, false, pixelNoise(pix, 3, s), -1.0);
            const float3 albedo1 = float3(bounce.ar, bounce.ag, bounce.ab);
            // An emissive surface (lava) the bounce ray hits gives off its own light too.
            bounceLight += albedo1 * (bounceSample.factor + bounceEmissive) + surfaceEmission(uint(bounce.renderIndex), albedo1);

            // Second bounce (Graphics menu "Second light bounce"): one more cosine weighted ray from the bounce point.
            // With cosine sampling the estimate is just the light coming back along the ray, times this surface's
            // albedo. Only the sun's share above the shadow floor counts there, since the floor stands for ambient light
            // the first bounce already has.
            if ((RtParams.indirectBounces >= 2) && (max(albedo1.r, max(albedo1.g, albedo1.b)) > kSecondBounceMinAlbedo)) {
                const float3 secondNormal = bounceTwoSided ? ((dot(bounceNormal, bounceDir) > 0.0) ? -bounceNormal : bounceNormal) : bounceNormal;
                const float3 secondGeometric = (dot(secondNormal, bounceGeometricNormal) >= 0.0) ? bounceGeometricNormal : -bounceGeometricNormal;
                const float secondBias = surfaceBias(hitT + bounce.hitT);
                const float3 secondOrigin = bouncePos + secondGeometric * secondBias;
                float3 secondDir = pixelCosineDirection(pix, uint2(53u, 97u), 29u, s, secondNormal);
                if (dot(secondDir, secondGeometric) <= 0.0) {
                    secondDir = reflect(secondDir, secondGeometric);
                }

                const float secondCone = (hitT + bounce.hitT) * RtParams.pixelSpreadAngle + bounce.hitT * kBounceConeSpread;
                HitPayload second = traceSurface(secondOrigin, secondDir, 0.0, 1.0e6, secondCone, kBounceConeSpread * 2.0, false, kCameraRayMask);
                float3 secondIncoming;
                if (second.hit > 0.5) {
                    const float3 secondPos = secondOrigin + secondDir * second.hitT;
                    const float3 secondHitNormal = float3(second.nx, second.ny, second.nz);
                    const float3 secondHitGeometric = float3(second.gx, second.gy, second.gz);
                    const float secondHitT = hitT + bounce.hitT + second.hitT;
                    LightSample secondSample = sampleLight(secondPos, secondHitNormal, secondHitGeometric, secondHitT, second.renderIndex, hitTwoSided(second.hit), pixelNoise(pix, 4, s));
                    const float3 secondEmissive = sampleEmissive(secondPos, secondHitNormal, secondHitGeometric, secondHitT, false, pixelNoise(pix, 5, s), -1.0);
                    const float3 shadowFloor = RtParams.cameraW.w * RtParams.cameraW.xyz;
                    const float3 albedo2 = float3(second.ar, second.ag, second.ab);
                    secondIncoming = albedo2 * (max(secondSample.factor - shadowFloor, 0.0) + secondEmissive) + surfaceEmission(uint(second.renderIndex), albedo2);
                }
                else {
                    secondIncoming = skyLight;
                }

                secondLight += albedo1 * secondIncoming;
            }
        }
        else {
            bounceLight += skyLight;
        }

        // Ray traced ambient occlusion: one short cosine weighted ray per sample. Like the sun's shadow rays, it skips
        // a billboard's own proxy, which crosses the sprite and would darken the half it leans over.
        const float3 occlusionDir = pixelCosineDirection(pix, uint2(31u, 17u), 11u, s, normal);
        if (dot(occlusionDir, geometricNormal) > 0.0) {
            occlusion += traceShadow(origin, occlusionDir, kAORadius, float(gInstanceId[pix]));
        }
    }

    bounceLight /= float(sampleCount);
    secondLight /= float(sampleCount);
    occlusion /= float(sampleCount);

    // Fill for the part of the surface the direct light does not reach. ComposePS combines it as
    // direct + (1 - direct) * fill, so fully lit surfaces keep the game's brightness. Rooms lit only by their own lights
    // take a fuller fill (see kFillBounceSunless).
    const float fillStrength = (RtParams.sunDisabled != 0) ? kFillBounceSunless : ((RtParams.indirectBounces >= 2) ? kFillBounceTwo : kFillBounce);
    const float3 fill = saturate((bounceLight + secondLight) * fillStrength);
    if (isDiagPixel(pix)) {
        // This frame's fill from each bounce at the center pixel, before temporal accumulation.
        const float3 firstFill = bounceLight * fillStrength;
        const float3 secondFill = secondLight * fillStrength;
        gHitInstanceId[kDiagBounceFill + 0] = asuint(firstFill.r);
        gHitInstanceId[kDiagBounceFill + 1] = asuint(firstFill.g);
        gHitInstanceId[kDiagBounceFill + 2] = asuint(firstFill.b);
        gHitInstanceId[kDiagBounceFill + 3] = asuint(secondFill.r);
        gHitInstanceId[kDiagBounceFill + 4] = asuint(secondFill.g);
        gHitInstanceId[kDiagBounceFill + 5] = asuint(secondFill.b);
        gHitInstanceId[kDiagBounceFill + 6] = asuint(fillStrength);
        gHitInstanceId[kDiagBounceFill + 7] = RtParams.indirectBounces;
    }

    const float4 current = float4(fill, 1.0 - occlusion);
    gFilteredIndirectLight[pix] = (RtParams.debugView != 0) ? float4(0.0, 0.0, 0.0, 1.0) : current;
    if (isDiagPixel(pix)) {
        // This frame's bounce fill and ambient occlusion at the center pixel.
        gHitVelocityDistance[15] = current;
    }
    if (diagFrame()) {
        const uint occlusionUnits = uint(saturate(current.a) * 255.0 + 0.5);
        const uint occlusionSum = WaveActiveSum(occlusionUnits);
        if (WaveIsFirstLane()) {
            InterlockedAdd(gHitInstanceId[kDiagOcclusionSum], occlusionSum);
        }

        diagCount(kDiagOcclusionPixels, true);
    }
}

// Surface material from RenderIndices::rtMaterial (see RtMaterialDatabase): roughness, metalness and specular
// strength. A strength of zero (every surface without a material entry) has no specular at all.
struct SurfaceSpecular {
    float roughness;
    float metalness;
    float strength;
};

SurfaceSpecular unpackSurfaceSpecular(uint packed) {
    SurfaceSpecular result;
    result.roughness = max(float(packed & 0xFF) / 255.0, kSpecularMinRoughness);
    result.metalness = float((packed >> 8) & 0xFF) / 255.0;
    result.strength = float((packed >> 16) & 0xFF) / 255.0;
    return result;
}

// Normal incidence reflectance: stronger specular materials reflect more, metals take their color from the texture.
float3 specularF0(SurfaceSpecular surface, float3 albedo) {
    const float dielectric = 0.04 + 0.12 * surface.strength;
    return lerp(dielectric.xxx, albedo, surface.metalness);
}

// GGX distribution, Smith-Schlick visibility and Schlick Fresnel, times N.L: the light a surface reflects towards the
// viewer for unit light from direction l.
float3 ggxSpecular(float3 n, float3 v, float3 l, float roughness, float3 f0) {
    const float ndotl = saturate(dot(n, l));
    const float ndotv = max(dot(n, v), 1e-3);
    if (ndotl <= 0.0) {
        return 0.0;
    }

    const float3 h = normalize(v + l);
    const float ndoth = saturate(dot(n, h));
    const float vdoth = saturate(dot(v, h));
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denominator = ndoth * ndoth * (a2 - 1.0) + 1.0;
    const float d = a2 / (M_PI * denominator * denominator);
    const float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float g = (ndotl / (ndotl * (1.0 - k) + k)) * (ndotv / (ndotv * (1.0 - k) + k));
    const float3 f = f0 + (1.0 - f0) * pow(1.0 - vdoth, 5.0);
    return d * g * f / (4.0 * ndotv);
}

// Specular pass: highlights from the sun or moon and the emissive lights, plus a mirror reflection on glossy
// surfaces. Every sample is deterministic (no per-frame noise), so highlights don't need denoising and don't smear in
// the temporal filters. Writes the specular light in rgb and the matching diffuse reduction in alpha for ComposePS.
[shader("raygeneration")]
void ReflectionRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    gReflection[pix] = float4(0.0, 0.0, 0.0, 1.0);
    if (isDiagPixel(pix)) {
        gHitColor[8] = float4(0.0, 0.0, 0.0, 1.0);
    }
    float3 position, normal, geometricNormal, velocity;
    float hitT;
    bool twoSided;
    if (kHybridLighting || (RtParams.debugView != 0) || !primaryHit(pix, position, normal, geometricNormal, velocity, hitT, twoSided)) {
        return;
    }

    const uint renderIndex = uint(gInstanceId[pix]);
    // Emissive materials (lava) glow on top of their lighting, like the specular light: shadows and ambient occlusion
    // don't touch it, and ComposePS keeps the sum in display range.
    const float3 emission = surfaceEmission(renderIndex, gDiffuse[pix].rgb);
    gReflection[pix] = float4(emission, 1.0);
    if (isDiagPixel(pix) && diagFrame()) {
        gHitInstanceId[kDiagEmissiveCenter + 0] = any(emission > 0.0) ? 1u : 0u;
        gHitInstanceId[kDiagEmissiveCenter + 1] = asuint(emission.r);
        gHitInstanceId[kDiagEmissiveCenter + 2] = asuint(emission.g);
        gHitInstanceId[kDiagEmissiveCenter + 3] = asuint(emission.b);
    }
    if (diagFrame() && (RtParams.emissionOffset != 0)) {
        diagCount(kDiagEmissivePixels, any(emission > 0.0));
    }

    const SurfaceSpecular surface = unpackSurfaceSpecular(gRtMaterials[renderIndex]);
    // Water surfaces get their reflection and glint from the water pass (RefractionRayGen).
    if ((surface.strength <= 0.0) || ((RtParams.waterDraws != 0) && ((gRtMaterials[renderIndex] & kMaterialWaterPassFlag) != 0))) {
        return;
    }

    float3 cameraOrigin, cameraDirection;
    cameraRay(pix, cameraOrigin, cameraDirection);
    const float3 v = -cameraDirection;
    const float3 albedo = gDiffuse[pix].rgb;
    const float3 f0 = specularF0(surface, albedo);
    const float weight = saturate(surface.strength * 2.0);
    const float ndotv = saturate(dot(normal, v));
    const float bias = surfaceBias(hitT);
    float3 specular = 0.0;

    // Sun or moon: the part of the light that shadows take away, tinted like the diffuse light.
    if ((RtParams.emissiveLightStart > 0) && (RtParams.sunDisabled == 0)) {
        const PointLight light = SceneLights[0];
        const float3 toLight = light.position - position;
        const float distance = length(toLight);
        const float3 l = toLight / max(distance, 1e-4);
        if ((dot(normal, l) > 0.0) && (dot(geometricNormal, l) > 0.0)) {
            const float visibility = 1.0 - traceShadow(shadowRayOrigin(position, geometricNormal, normal, l, bias, twoSided), l, distance, float(renderIndex));
            if (visibility > 0.0) {
                const float maxColor = max(max(light.diffuseColor.r, light.diffuseColor.g), max(light.diffuseColor.b, 1e-4));
                const float3 lightTint = lerp(float3(1.0, 1.0, 1.0), light.diffuseColor / maxColor, kLightColorTint);
                const float sunShare = 1.0 - dot(RtParams.cameraW.xyz, float3(0.299, 0.587, 0.114));
                specular += ggxSpecular(normal, v, l, max(surface.roughness, kSunSpecularMinRoughness), f0) * lightTint * (sunShare * visibility * kSunSpecularStrength);
            }
        }
    }

    // Emissive lights, unshadowed. A light's size widens its highlight.
    [loop]
    for (uint i = 0; i < RtParams.emissiveLightCount; i++) {
        const PointLight light = SceneLights[RtParams.emissiveLightStart + i];
        const float3 toLight = light.position - position;
        const float distance = length(toLight);
        if ((distance >= light.attenuationRadius) || (distance < 1e-3)) {
            continue;
        }

        const float3 l = toLight / distance;
        if (dot(geometricNormal, l) <= 0.0) {
            continue;
        }

        const float roughness = saturate(surface.roughness + light.pointRadius / (2.0 * distance));
        specular += ggxSpecular(normal, v, l, roughness, f0) * emissiveRadiance(light) * emissiveAttenuation(distance, light.attenuationRadius) * kEmissiveStrength;
    }

    // Glossy reflection: one mirror ray, with the surface's roughness widening the texture footprint of what it sees.
    const float3 reflectionFresnel = f0 + (max(float3(1.0, 1.0, 1.0) - surface.roughness, f0) - f0) * pow(1.0 - ndotv, 5.0);
    const float glossy = 1.0 - smoothstep(kGlossyReflectionRoughness * 0.5, kGlossyReflectionRoughness, surface.roughness);
    if (glossy > 0.0) {
        float3 reflectionDirection = reflect(cameraDirection, normal);
        if (dot(reflectionDirection, geometricNormal) <= 0.0) {
            reflectionDirection = reflect(reflectionDirection, geometricNormal);
        }

        const float3 origin = position + geometricNormal * bias;
        const HitPayload reflected = traceSurface(origin, reflectionDirection, 0.0, 1.0e6, hitT * RtParams.pixelSpreadAngle, RtParams.pixelSpreadAngle + surface.roughness * 0.5, false, kCameraRayMask);
        float3 reflectedLight;
        if (reflected.hit > 0.5) {
            const float3 reflectedPosition = origin + reflectionDirection * reflected.hitT;
            const float3 reflectedNormal = float3(reflected.nx, reflected.ny, reflected.nz);
            const float3 reflectedGeometricNormal = float3(reflected.gx, reflected.gy, reflected.gz);
            const LightSample reflectedSample = sampleLight(reflectedPosition, reflectedNormal, reflectedGeometricNormal, hitT + reflected.hitT, reflected.renderIndex, hitTwoSided(reflected.hit), float2(0.0, 0.0));
            reflectedLight = float3(reflected.ar, reflected.ag, reflected.ab) * reflectedSample.factor + surfaceEmission(uint(reflected.renderIndex), float3(reflected.ar, reflected.ag, reflected.ab));
            reflectedLight = lerp(reflectedLight, float3(reflected.fr, reflected.fg, reflected.fb), saturate(reflected.fogFactor));
        }
        else {
            reflectedLight = ambientBase() * kReflectionSkyColor;
        }

        specular += reflectedLight * reflectionFresnel * glossy;
    }

    specular *= weight;

    // Light that leaves as specular doesn't also leave as diffuse. Metals keep part of their texture as diffuse, since
    // the N64 textures already paint their look.
    const float diffuseScale = (1.0 - weight * dot(reflectionFresnel, float3(0.299, 0.587, 0.114)) * glossy) * (1.0 - kMetalDiffuseLoss * surface.metalness * weight);
    gReflection[pix] = float4(min(specular, kSpecularMaxRadiance) + emission, saturate(diffuseScale));
    if (isDiagPixel(pix)) {
        gHitColor[8] = gReflection[pix];
    }
}

// Advanced Water Rendering. Water comes in two kinds:
// - Blended layers without depth writes (fountains, ponds). They are not part of the path traced scene; the renderer
//   puts them in the acceleration structure for this pass only (kWaterRayMask). The layer in front of the camera
//   ray's hit is composited the way the game blends it (its texture, scrolling and opacity, from the hit shader).
// - Opaque, depth writing surfaces (the sea by Limelin). They are path traced like any surface, and their lit color
//   (with the sun's shadows from the lighting passes) is the base; gRtMaterials flags them (kMaterialWaterPassFlag).
// Both get animated waves on the normal, a Fresnel reflection of the scene or the sky, a sharp sun glint, clarity and
// absorption by depth from a ray through the water, and a little foam at the shoreline. The game's texture stays the
// base color. Output for ComposePS: rgb is added over the composed pixel, alpha is how much of that pixel the water
// hides. Blended water seen against the sky (no scene hit behind it) only adds its reflection there.
static const float kWaterF0 = 0.02;
// How far the shading goes from the game's water: 0 leaves it as drawn, 1 is the full effect.
static const float kWaterModernStrength = 1.0;
// Blended water: opacity fades towards the shore over this depth (along the view ray, from the water to the ground),
// down to this fraction of the game's opacity at the waterline.
static const float kWaterShoreDepth = 25.0;
static const float kWaterShoreOpacity = 0.35;
// Opaque water: the ground shows through within kWaterClearDepth (at most kWaterShallowClarity of it, tinted by the
// water), and towards kWaterDeepDepth (or with no ground found) the water darkens to kWaterDeepDarken of its color.
static const float kWaterClearDepth = 30.0;
static const float kWaterShallowClarity = 0.55;
static const float kWaterDeepDepth = 200.0;
static const float kWaterDeepDarken = 0.82;
// How far the ray through opaque water looks for the ground.
static const float kWaterThroughDistance = 400.0;
// Foam where the water is shallower than this, and its brightness.
static const float kWaterFoamDepth = 5.0;
static const float kWaterFoamStrength = 0.25;
// Waves on the normal (see waterWaveNormal) and the sun glint on them.
static const float kWaterWaveStrength = 1.0;
static const float kWaterGlintStrength = 1.0;
static const float kWaterMinGlintRoughness = 0.05;
static const float kWaterMaxGlintRoughness = 0.1;
// Brightness of the sky seen in the water, relative to the sun's color, when the room has a sun or moon.
static const float kWaterSkyLevel = 0.75;

// Slope of one train of waves moving along dir. A slow warp along the crests breaks up straight lines, and the slope
// leans towards the front of each wave so the crests look sharper than a sine (a little like Gerstner waves).
float2 waveSlope(float2 p, float2 dir, float wavelength, float speed, float t) {
    const float k = 2.0 * M_PI / wavelength;
    const float2 side = float2(-dir.y, dir.x);
    const float warp = 0.9 * sin(dot(p, side) * k * 0.31 + t * 0.4);
    const float phase = (dot(p, dir) - t * speed) * k + warp;
    return dir * (cos(phase) * (1.0 + 0.35 * sin(phase)));
}

// The water's normal tilted by three crossing wave layers in world space, moving in real time (RtParams.waterTime is
// in seconds). Layers shorter than a few pixels at this distance fade out so they don't shimmer.
float3 waterWaveNormal(float3 position, float3 normal, float distance) {
    const float t = RtParams.waterTime;
    const float pixelSize = max(distance * RtParams.pixelSpreadAngle, 1e-3);
    const float2 p = position.xz;
    float2 slope = waveSlope(p, float2(0.8, 0.6), 90.0, 9.0, t) * (0.10 * saturate(90.0 / (pixelSize * 8.0) - 0.5));
    slope += waveSlope(p, float2(-0.45, 0.893), 34.0, 5.5, t) * (0.07 * saturate(34.0 / (pixelSize * 8.0) - 0.5));
    slope += waveSlope(p, float2(0.17, -0.985), 13.0, 3.5, t) * (0.045 * saturate(13.0 / (pixelSize * 8.0) - 0.5));
    slope *= kWaterWaveStrength * kWaterModernStrength;
    return normalize(normal + float3(slope.x, 0.0, slope.y) * abs(normal.y));
}

// Blue or blue-green: the colors water found only by its scrolling texture must have.
float waterHue(float3 color) {
    return saturate((color.b - color.r - 0.02) * 15.0);
}

// The sky a reflection sees when it leaves the scene. Where the reflected direction lands on screen, the rasterized
// sky there is the best match, if that pixel shows sky. Otherwise a gradient from the fog color at the horizon to a
// clear sky color above, as bright as the sun allows.
float3 waterSkyColor(float3 position, float3 direction, float3 fogColor, bool sunLit) {
    const float4 clip = mul(RtParams.viewProj, float4(position + direction * 1.0e5, 1.0));
    if (clip.w > 1e-4) {
        const float2 ndc = clip.xy / clip.w;
        const float2 vp = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y / (2.0 * RtParams.nearDist));
        const int2 skyPixel = int2(floor(RtParams.viewport.xy + vp * RtParams.viewport.zw));
        if (all(skyPixel >= 0) && (skyPixel.x < int(RtParams.resolution.x)) && (skyPixel.y < int(RtParams.resolution.y)) && (gDiffuse[skyPixel].a <= EPSILON)) {
            return rasterColor(uint2(skyPixel));
        }
    }

    float skyLevel = ambientBase();
    if (sunLit) {
        const float3 sunColor = SceneLights[0].diffuseColor;
        skyLevel = max(skyLevel, saturate(max(sunColor.r, max(sunColor.g, sunColor.b))) * kWaterSkyLevel);
    }

    const float3 zenith = kReflectionSkyColor * skyLevel;
    const float3 horizon = (dot(fogColor, float3(1.0, 1.0, 1.0)) > 0.03) ? lerp(zenith, fogColor, 0.6) : zenith * 1.1;
    return lerp(horizon, zenith, saturate(direction.y * 2.5));
}

[shader("raygeneration")]
void RefractionRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    gRefraction[pix] = 0.0;
    // Emissive surfaces (lava, often an animated texture) are marked for the upscaler's reactive mask.
    if ((RtParams.emissionOffset != 0) && (RtParams.debugView == 0) && (gDiffuse[pix].a > EPSILON) &&
        any(surfaceEmission(uint(gInstanceId[pix]), gDiffuse[pix].rgb) > 0.0))
    {
        gRefraction[pix] = float4(0.0, 0.0, 0.0, kEmissiveReactiveCoverage);
    }

    if ((RtParams.waterDraws == 0) || kHybridLighting || (RtParams.debugView != 0)) {
        return;
    }

    float3 origin, direction;
    if (!pixelInViewport(pix) || !cameraRay(pix, origin, direction)) {
        return;
    }

    float3 position, normal, geometricNormal, velocity;
    float hitT;
    bool twoSided;
    const bool sceneHit = primaryHit(pix, position, normal, geometricNormal, velocity, hitT, twoSided);
    const float nearT = cameraNearT(pix, origin, direction);
    const bool sunLit = (RtParams.emissiveLightStart > 0) && (RtParams.sunDisabled == 0);

    // Find the water: the path traced surface itself (opaque water), or a blended layer in front of the scene or sky.
    // Kind: 1 blended over the scene, 2 opaque, 3 blended against the sky.
    uint kind;
    uint renderIndex;
    float waterT;
    float alpha;
    float3 waterGeometric;
    float3 albedo;
    float3 fogColor;
    float fogAmount;
    const uint primaryMaterial = sceneHit ? gRtMaterials[uint(gInstanceId[pix])] : 0u;
    if ((primaryMaterial & kMaterialWaterPassFlag) != 0) {
        albedo = gDiffuse[pix].rgb;
        if (((primaryMaterial & kMaterialWaterHueGateFlag) != 0) && (waterHue(albedo) < 0.5)) {
            if (diagFrame() && isDiagPixel(pix)) {
                gHitInstanceId[kDiagWaterCenter + 0] = 4;
                gHitInstanceId[kDiagWaterCenter + 1] = uint(gInstanceId[pix]);
            }

            return;
        }

        kind = 2;
        renderIndex = uint(gInstanceId[pix]);
        waterT = hitT;
        alpha = 1.0;
        waterGeometric = geometricNormal;
        const float4 fog = gTransparent[pix];
        fogColor = fog.rgb;
        fogAmount = fog.a;
    }
    else {
        const float tMax = sceneHit ? hitT : 1.0e7;
        if (tMax <= nearT) {
            return;
        }

        const HitPayload water = traceSurface(origin, direction, nearT, tMax, nearT * RtParams.pixelSpreadAngle, RtParams.pixelSpreadAngle, false, kWaterRayMask);
        if (water.hit < 0.5) {
            return;
        }

        kind = sceneHit ? 1 : 3;
        renderIndex = uint(water.renderIndex);
        waterT = water.hitT;
        alpha = water.alpha;
        waterGeometric = float3(water.gx, water.gy, water.gz);
        albedo = float3(water.ar, water.ag, water.ab);
        fogColor = float3(water.fr, water.fg, water.fb);
        fogAmount = water.fogFactor;
    }

    // rt_materials.json water entries: specular is the reflectivity, roughness the sharpness of the sun glint.
    const SurfaceSpecular surface = unpackSurfaceSpecular(gRtMaterials[renderIndex]);
    const float reflectivity = (surface.strength > 0.0) ? surface.strength : 1.0;
    const float glintRoughness = clamp(surface.roughness, kWaterMinGlintRoughness, kWaterMaxGlintRoughness);
    const float3 waterPosition = origin + direction * waterT;
    const float3 v = -direction;
    const float3 wave = waterWaveNormal(waterPosition, waterGeometric, waterT);
    const float ndotv = saturate(dot(wave, v));
    const float fresnel = saturate((kWaterF0 + (1.0 - kWaterF0) * pow(1.0 - ndotv, 5.0)) * reflectivity) * kWaterModernStrength;
    const float bias = surfaceBias(waterT);

    // The sun on the water: shadows and the glint.
    float3 waterLight = 1.0;
    float sunVisibility = 0.0;
    if (sunLit) {
        const LightSample sun = sampleLight(waterPosition, waterGeometric, waterGeometric, waterT, float(renderIndex), false, float2(0.0, 0.0));
        sunVisibility = sun.visibility;
        waterLight = lerp(RtParams.cameraW.xyz, float3(1.0, 1.0, 1.0), sunVisibility);
    }

    // Depth of the water: to the scene behind blended water, or along a ray through opaque water to the ground.
    float depth = -1.0;
    float3 groundColor = 0.0;
    if (kind == 1) {
        depth = max(hitT - waterT, 0.0);
    }
    else if (kind == 2) {
        const HitPayload ground = traceSurface(waterPosition - waterGeometric * bias, direction, 0.0, kWaterThroughDistance, waterT * RtParams.pixelSpreadAngle, RtParams.pixelSpreadAngle, false, kCameraRayMask);
        if (ground.hit > 0.5) {
            depth = ground.hitT;
            // Lit like the water above it, and tinted by the water's color.
            const float3 tint = albedo / max(max(albedo.r, max(albedo.g, albedo.b)), 1e-3);
            groundColor = float3(ground.ar, ground.ag, ground.ab) * waterLight * lerp(float3(1.0, 1.0, 1.0), tint, 0.5);
        }
    }

    // Reflection of the scene or the sky along the wave normal.
    float3 reflectionDirection = reflect(direction, wave);
    if (dot(reflectionDirection, waterGeometric) <= 0.0) {
        reflectionDirection = reflect(reflectionDirection, waterGeometric);
    }

    const float3 reflectionOrigin = waterPosition + waterGeometric * bias;
    const HitPayload reflected = traceSurface(reflectionOrigin, reflectionDirection, 0.0, 1.0e6, waterT * RtParams.pixelSpreadAngle, RtParams.pixelSpreadAngle + glintRoughness * 0.5, false, kCameraRayMask);
    float3 reflectedLight;
    if (reflected.hit > 0.5) {
        const float3 reflectedPosition = reflectionOrigin + reflectionDirection * reflected.hitT;
        const float3 reflectedNormal = float3(reflected.nx, reflected.ny, reflected.nz);
        const float3 reflectedGeometricNormal = float3(reflected.gx, reflected.gy, reflected.gz);
        const LightSample reflectedSample = sampleLight(reflectedPosition, reflectedNormal, reflectedGeometricNormal, waterT + reflected.hitT, reflected.renderIndex, hitTwoSided(reflected.hit), float2(0.0, 0.0));
        reflectedLight = float3(reflected.ar, reflected.ag, reflected.ab) * reflectedSample.factor + surfaceEmission(uint(reflected.renderIndex), float3(reflected.ar, reflected.ag, reflected.ab));
        reflectedLight = lerp(reflectedLight, float3(reflected.fr, reflected.fg, reflected.fb), saturate(reflected.fogFactor));
    }
    else {
        reflectedLight = waterSkyColor(reflectionOrigin, reflectionDirection, fogColor, sunLit);
    }

    // A sharp sun glint on the waves.
    float3 glint = 0.0;
    if (sunLit && (sunVisibility > 0.0)) {
        const PointLight light = SceneLights[0];
        const float3 l = normalize(light.position - waterPosition);
        const float maxColor = max(max(light.diffuseColor.r, light.diffuseColor.g), max(light.diffuseColor.b, 1e-4));
        const float3 lightTint = lerp(float3(1.0, 1.0, 1.0), light.diffuseColor / maxColor, kLightColorTint);
        const float sunShare = 1.0 - dot(RtParams.cameraW.xyz, float3(0.299, 0.587, 0.114));
        glint = ggxSpecular(wave, v, l, glintRoughness, kWaterF0.xxx) * lightTint * (sunShare * sunVisibility * kWaterGlintStrength * kWaterModernStrength);
        glint = min(glint, kSpecularMaxRadiance);
    }

    // ComposePS shows: pixel * (1 - coverage) + layer.
    float coverage;
    float3 layer;
    float opacity = alpha;
    if (kind == 1) {
        // The game's blend, opacity * water + (1 - opacity) * below, clearer at the shore, under the reflection.
        opacity *= lerp(1.0, lerp(kWaterShoreOpacity, 1.0, smoothstep(0.0, kWaterShoreDepth, depth)), kWaterModernStrength);
        coverage = 1.0 - (1.0 - opacity) * (1.0 - fresnel);
        layer = (1.0 - fresnel) * opacity * albedo * waterLight + fresnel * reflectedLight;
    }
    else if (kind == 2) {
        // The lit water as path traced, showing the ground in the shallows and darker in deep water.
        const float clarity = (depth >= 0.0) ? (kWaterShallowClarity * (1.0 - smoothstep(0.0, kWaterClearDepth, depth)) * kWaterModernStrength) : 0.0;
        const float deepness = (depth >= 0.0) ? smoothstep(0.0, kWaterDeepDepth, depth) : 1.0;
        const float darken = lerp(1.0, kWaterDeepDarken, deepness * kWaterModernStrength);
        coverage = 1.0 - (1.0 - fresnel) * (1.0 - clarity) * darken;
        layer = (1.0 - fresnel) * clarity * groundColor + fresnel * reflectedLight;
    }
    else {
        // Against the sky the pixel is the game's own water; only the reflection goes on top.
        coverage = fresnel;
        layer = fresnel * reflectedLight;
    }

    // Foam along the shoreline, broken up by the waves.
    if ((depth >= 0.0) && (depth < kWaterFoamDepth)) {
        const float foam = (1.0 - depth / kWaterFoamDepth) * kWaterFoamStrength * kWaterModernStrength * saturate(0.6 + 4.0 * (wave.x + wave.z));
        layer += foam * (sunLit ? waterLight : ambientBase().xxx);
    }

    layer += glint;
    layer = lerp(layer, fogColor * coverage, saturate(fogAmount));
    gRefraction[pix] = float4(layer, saturate(coverage));

    if (diagFrame()) {
        diagCount(kDiagWaterPixels, true);
        if (isDiagPixel(pix)) {
            gHitInstanceId[kDiagWaterCenter + 0] = kind;
            gHitInstanceId[kDiagWaterCenter + 1] = renderIndex;
            gHitInstanceId[kDiagWaterCenter + 2] = asuint(opacity);
            gHitInstanceId[kDiagWaterCenter + 3] = asuint(fresnel);
            gHitInstanceId[kDiagWaterCenter + 4] = asuint(depth);
        }
    }
}
