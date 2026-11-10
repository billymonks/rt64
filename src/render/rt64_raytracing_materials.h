//
// RT64
//

#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace RT64 {
    // Surface classes for the ray traced materials. Only Stone, PolishedFloor, Metal and Water get a specular lobe by
    // default; the others are matte and only name what the guess saw.
    enum class RtMaterialClass : uint8_t {
        Matte = 0,
        Sprite,
        Grass,
        Dirt,
        Wood,
        Cloth,
        Skin,
        Stone,
        PolishedFloor,
        Metal,
        Water,
        Lava,
        Foliage,
        Count
    };

    // What the GPU saw when it sampled the texture of a draw (see EmissiveColorRayGen in RtLibrary.hlsl).
    struct RtMaterialStats {
        float color[3] = {};
        float coverage = 0.0f;
        float luminanceDeviation = 0.0f;
        float peakBrightness = 0.0f;
    };

    // Where the texture was seen.
    struct RtMaterialPlacement {
        // Drawn with N64 lighting (vertex normals): characters, enemies and items rather than level geometry.
        bool lit = false;
        bool alphaTested = false;
        // Area weighted average of |normal.y|: 1 for floors and ceilings, 0 for walls.
        float upFacing = 0.0f;
        uint32_t triangleCount = 0;
        // The game's map and room when it was seen, and the texture's N64 size and format.
        int map = -1;
        int room = -1;
        const char *areaName = "";
        uint32_t textureWidth = 0;
        uint32_t textureHeight = 0;
        uint8_t textureFormat = 0;
        uint8_t textureSize = 0;
    };

    struct RtMaterialEntry {
        std::string label;
        RtMaterialClass materialClass = RtMaterialClass::Matte;
        float roughness = 0.9f;
        float metalness = 0.0f;
        float specular = 0.0f;
        // Light the surface gives off ("emission" in rt_materials.json): its texture color (or "emission color")
        // times this, on top of its lighting. Lava glows this way.
        float emission = 0.0f;
        bool emissionFromTexture = true;
        float emissionColor[3] = { 1.0f, 1.0f, 1.0f };
        // Average texture color from "stats", for the color of the lights an emissive surface casts.
        float averageColor[3] = { 0.5f, 0.5f, 0.5f };
        bool user = false;
        uint32_t packed = 0;
        // Readable context, kept from the first time the texture was seen.
        std::string surface;
        std::string area;
        int map = -1;
        int room = -1;
        std::string texture;
        std::string preview;
        std::string stats;
    };

    // Which kinds of effect draw an emitter entry applies to (RtEmitterParams::kinds); none set means any.
    enum RtEmitterKind : uint32_t {
        RtEmitterKindAdditive = 0x1,
        RtEmitterKindBlended = 0x2,
        RtEmitterKindCutout = 0x4,
        RtEmitterKindUntextured = 0x8,
        RtEmitterKindOpaque = 0x10
    };

    // What identifies an effect draw: its texture, the vertex load its polygons come from, its combiner and render
    // mode, its colors and its size. The signature id hashes the most identifying of them.
    struct RtEmitterSignature {
        uint64_t signature = 0;
        // Texture key (tmem hash), 0 for untextured draws.
        uint64_t texture = 0;
        uint32_t vertexAddress = 0;
        uint32_t combiner[2] = {};
        uint32_t otherMode[2] = {};
        float prim[4] = {};
        float env[4] = {};
        float vertexColor[3] = {};
        uint32_t triangles = 0;
        uint32_t kind = 0;
        // The key texture-only entries were written with: the texture, or for untextured draws a hash of the combiner
        // and primitive color. Not part of the signature id.
        uint64_t legacyKey = 0;

        void computeSignature() {
            uint64_t hash = 0xCBF29CE484222325ULL;
            auto mix = [&hash](uint64_t value) {
                for (int i = 0; i < 8; i++) {
                    hash ^= (value >> (i * 8)) & 0xFF;
                    hash *= 0x100000001B3ULL;
                }
            };

            mix(texture);
            mix(vertexAddress);
            mix((uint64_t(combiner[0]) << 32) | combiner[1]);
            mix((uint64_t(otherMode[0]) << 32) | otherMode[1]);
            mix(triangles);
            signature = (hash == 0) ? 1 : hash;
        }
    };

    // Which fields of a draw's signature an emitter entry requires. Entries from the texture-only format match on the
    // texture alone.
    struct RtEmitterMatch {
        bool signature = false;
        bool texture = false;
        bool vertexAddress = false;
        bool combiner = false;
        bool otherMode = false;
        bool prim = false;
        bool env = false;
        bool vertexColor = false;
        bool triangleRange = false;
        RtEmitterSignature value;
        uint32_t minTriangles = 0;
        uint32_t maxTriangles = 0;
        float colorTolerance = 0.1f;

        bool any() const {
            return signature || texture || vertexAddress || combiner || otherMode || prim || env || vertexColor || triangleRange;
        }

        bool matches(const RtEmitterSignature &draw) const {
            auto close = [this](const float *a, const float *b, int count) {
                for (int c = 0; c < count; c++) {
                    if (std::fabs(a[c] - b[c]) > colorTolerance) {
                        return false;
                    }
                }

                return true;
            };

            if (!any()) return false;
            if (signature && (value.signature != draw.signature)) return false;
            if (texture && (value.texture != draw.texture) && !((draw.texture == 0) && (value.texture != 0) && (value.texture == draw.legacyKey))) return false;
            if (vertexAddress && (value.vertexAddress != draw.vertexAddress)) return false;
            if (combiner && ((value.combiner[0] != draw.combiner[0]) || (value.combiner[1] != draw.combiner[1]))) return false;
            if (otherMode && ((value.otherMode[0] != draw.otherMode[0]) || (value.otherMode[1] != draw.otherMode[1]))) return false;
            if (prim && !close(value.prim, draw.prim, 4)) return false;
            if (env && !close(value.env, draw.env, 4)) return false;
            if (vertexColor && !close(value.vertexColor, draw.vertexColor, 3)) return false;
            if (triangleRange && ((draw.triangles < minTriangles) || ((maxTriangles > 0) && (draw.triangles > maxTriangles)))) return false;
            return true;
        }
    };

    // How a listed emitter lights its surroundings, for the light tracking in RaytracingResources.
    struct RtEmitterParams {
        // The entry's id: its signature id, or the texture key of a texture-only entry.
        uint64_t id = 0;
        RtEmitterMatch match;
        bool enabled = false;
        // Only shine as much as the GPU finds the sprite glows (the heuristic used before the list): set for the
        // legacy glow of unlisted untextured effects.
        bool glowGate = false;
        // Color from the sprite as the GPU samples it, or the fixed color below.
        bool sampleColor = true;
        float color[3] = { 1.0f, 1.0f, 1.0f };
        float intensity = 1.0f;
        float rangeScale = 1.0f;
        // Most lights this texture makes at once (its largest clusters); 0 for no limit.
        uint32_t maxLights = 0;
        // Frames to fade in and out; 0 for the defaults.
        uint32_t fadeInFrames = 0;
        uint32_t fadeOutFrames = 0;
        uint32_t kinds = 0;
        // Only in these maps and rooms (room -1 for any room of the map); empty for everywhere.
        static const uint32_t MaxAreas = 8;
        uint32_t areaCount = 0;
        int areaMaps[MaxAreas] = {};
        int areaRooms[MaxAreas] = {};

        bool appliesTo(uint32_t kind, int map, int room) const {
            if ((kinds != 0) && ((kinds & kind) == 0)) {
                return false;
            }

            if (areaCount == 0) {
                return true;
            }

            for (uint32_t a = 0; a < areaCount; a++) {
                if ((areaMaps[a] == map) && ((areaRooms[a] < 0) || (areaRooms[a] == room))) {
                    return true;
                }
            }

            return false;
        }
    };

    struct RtEmitterEntry {
        std::string label;
        RtEmitterParams params;
        // False for candidates the game found but nobody listed yet.
        bool listed = false;
        // "user" (edited by hand or in the HTML editor), "shipped" (from assets) or "game" (recorded candidate).
        std::string source = "game";
        // The draw as the game last saw it, to copy match fields from.
        RtEmitterSignature observed;
        bool hasObserved = false;
        // The heuristic used before the list would have made it glow (untextured and additive effects, or a sprite
        // the GPU found glowing).
        bool wouldGlow = false;
        std::string kind;
        std::string area;
        int map = -1;
        int room = -1;
        std::string size;
        std::string preview;
    };

    // An invisible point light placed by hand in rt_lights.json (Eltale Studio, Rooms mode), for rooms without a
    // light of their own.
    struct RtManualLight {
        std::string id;
        bool enabled = false;
        int map = -1;
        // -1 for every room of the map.
        int room = -1;
        float position[3] = {};
        float color[3] = { 1.0f, 1.0f, 1.0f };
        float intensity = 1.0f;
        float range = 150.0f;
        float sourceRadius = 4.0f;
        bool castsShadows = true;
        float flicker = 0.0f;
        float flickerSpeed = 1.0f;
        // From the id, so each flickering light wavers on its own.
        float flickerPhase = 0.0f;
    };

    // rt_lights.json "room settings": per map (room -1 for all its rooms) or per room, whether the sun or moon lights it.
    struct RtRoomSetting {
        int map = -1;
        int room = -1;
        bool sunAndMoon = true;
        bool fromLocal = true;
    };

    struct RtMaterialProbe {
        uint64_t key = 0;
        uint32_t renderIndex = 0;
        uint32_t triangleCount = 0;
        RtMaterialPlacement placement;
        // An effect texture sampled only to tag its emitter candidates, not for a material guess.
        bool effect = false;
    };

    // Material guesses and user overrides by texture key (the tmem hash, like the emissive keys), kept in
    // rt_materials.json under Documents\Quest64Recompiled\Ray Tracing, with a PNG preview of each texture in rt_materials\ and
    // an rt_materials.html index to browse them. The file is seeded from assets/materials/rt_materials.json next to the
    // executable, and its user entries are read again when it changes. Loading, saving and the previews all run on a
    // background thread; until the file is loaded every surface is matte.
    class RtMaterialDatabase {
    public:
        ~RtMaterialDatabase();
        void loadOnce();
        // Packed material of a texture for gRtMaterials, 0 when unknown.
        uint32_t packedMaterial(uint64_t key);
        // An emissive texture's emission for gRtMaterials (strength in the top byte, color in the rest, 0 for the
        // texture's color) and the light it gives off (average color times strength). False when not emissive.
        bool emissiveMaterial(uint64_t key, uint32_t &packedEmission, float radiance[3]);
        // True once per session for a texture without a user entry, until a probe for it is pending.
        bool wantsProbe(uint64_t key, uint32_t frame);
        void markProbePending(uint64_t key, uint32_t frame);
        void finishProbe(uint64_t key);
        void addGuess(uint64_t key, const RtMaterialStats &stats, const RtMaterialPlacement &placement);
        // A water surface found by Advanced Water Rendering. Water is never path traced, so its texture would get no
        // entry; this records it once as a "water" guess, listed in rt_materials.html.
        void noteWaterDraw(uint64_t key, const RtMaterialPlacement &placement);
        // True once for a texture with an entry but no preview image yet.
        bool wantsPreview(uint64_t key);
        // Tightly packed RGBA8 pixels of the texture, written as a PNG on the background thread.
        void submitPreview(uint64_t key, std::vector<uint8_t> &&pixels, uint32_t width, uint32_t height);
        void saveIfDirty(uint32_t frame, bool force);
        bool find(uint64_t key, RtMaterialEntry &entry);
        void classCounts(uint32_t counts[size_t(RtMaterialClass::Count)], uint32_t &userEntries);
        size_t size();
        std::string path();

        // Emissive lights come only from draws matching an entry listed in rt_emitters.json. Returns whether any listed
        // entry matches the draw; params are the matching entry's, preferring one that's enabled and applies here
        // (applies tells whether it does).
        bool matchEmitter(const RtEmitterSignature &draw, int map, int room, RtEmitterParams &params, bool &applies);
        bool emitterParams(uint64_t id, RtEmitterParams &params);
        bool findEmitter(uint64_t id, RtEmitterEntry &entry);
        // An effect draw the light detection found, recorded once by signature in the candidates list.
        void recordEmitterCandidate(const RtEmitterSignature &draw, float extent, int map, int room, const char *areaName, bool wouldGlow);
        void emitterCounts(uint32_t &listed, uint32_t &enabled, uint32_t &candidates);
        // The GPU sample of an effect texture passed the old bright-and-tinted sprite rule: its candidates get tagged.
        void markTextureWouldGlow(uint64_t textureKey);
        // rt_emitters.json "legacy glow for untextured effects": unlisted untextured polygon effects (the item pillar,
        // hit sparks) glow the way they did before the list, as far as the GPU finds them bright. On by default.
        bool legacyGlowUntextured() const { return legacyGlow; }

        // Manual point lights (rt_lights.json), swapped in whole by the background thread. The render thread takes
        // the current list once per frame.
        std::shared_ptr<const std::vector<RtManualLight>> manualLightList();
        std::string manualLightsPath();
        // Whether the sun or moon lights this room, and where that comes from: "default", "local" or "shipped", and
        // whether a "map-wide" or "room" entry decided it.
        bool sunAndMoonEnabled(int map, int room, const char *&source, const char *&scope);
        // Called once per frame: about once a second asks the background thread whether rt_lights.json or
        // rt_materials.json changed (see kManualLightsHotReload and kMaterialsHotReload).
        void pollEditedFiles();

    private:
        void loadManualLights(bool onlyIfChanged);
        // The user entries of an rt_materials.json changed outside the game replace the ones in memory.
        void loadEditedMaterials();
        void load();
        void writeFiles(const std::unordered_map<uint64_t, RtMaterialEntry> &snapshot, const std::unordered_map<uint64_t, RtEmitterEntry> &emitterSnapshot);
        void post(std::function<void()> job);
        // Points the material entry and every emitter entry of this texture at its preview image; mutex held.
        void setPreview(uint64_t textureKey, const std::string &preview);
        void workerLoop();

        std::mutex mutex;
        std::atomic<bool> loadStarted = false;
        std::atomic<bool> loaded = false;
        // Authoring mode (see load): the Documents folder exists, so its files override the shipped ones and are saved.
        // Otherwise the shipped files are the whole configuration and nothing is written.
        std::atomic<bool> authoring = false;
        bool dirty = false;
        uint32_t lastSaveFrame = 0;
        std::string localPath;
        // When rt_materials.json was last read or written by the game, to tell an outside edit from its own saves.
        std::filesystem::file_time_type materialsWriteTime{};
        std::string previewDirectory;
        std::unordered_map<uint64_t, RtMaterialEntry> entries;
        std::unordered_set<uint64_t> probedThisSession;
        std::unordered_map<uint64_t, uint32_t> pendingProbes;
        std::unordered_set<uint64_t> previewsRequested;
        std::unordered_map<uint64_t, RtEmitterEntry> emitters;
        std::vector<uint64_t> listedEmitterIds;
        std::atomic<bool> legacyGlow = true;
        std::shared_ptr<const std::vector<RtManualLight>> manualLights = std::make_shared<const std::vector<RtManualLight>>();
        std::string manualLightsFile;
        std::vector<RtRoomSetting> roomSettings;
        std::filesystem::file_time_type manualLightsWriteTime{};
        std::atomic<bool> editedFilesCheckPending = false;
        std::chrono::steady_clock::time_point editedFilesLastPoll{};
        std::unordered_set<uint64_t> emitterTextureKeys;
        std::unordered_set<uint64_t> wouldGlowTextures;

        std::mutex jobMutex;
        std::condition_variable jobChanged;
        std::deque<std::function<void()>> jobs;
        std::thread worker;
        bool workerStop = false;
    };

    RtMaterialDatabase &rtMaterialDatabase();

    // A copy of a shipped configuration file (rt_materials.json, rt_emitters.json or rt_lights.json) compiled into the
    // game. It's used instead of assets\materials, so a build always runs with the configuration it was built with.
    // Call before ray tracing starts; the data must stay valid for the whole run.
    void setRaytracingShippedConfig(const char *fileName, const char *data, size_t size);
    const char *rtMaterialClassName(RtMaterialClass materialClass);
    uint32_t packRtMaterial(float roughness, float metalness, float specular, RtMaterialClass materialClass);
};
