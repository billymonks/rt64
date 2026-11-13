//
// RT64
//

#include "rt64_raytracing_materials.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include <json/json.hpp>

#define STB_IMAGE_WRITE_STATIC
#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <Windows.h>
#   include <ShlObj.h>
#   pragma comment(lib, "shell32.lib")
#   pragma comment(lib, "ole32.lib")
#endif

namespace RT64 {
    void rtDiagLog(const char *fmt, ...);

    using json = nlohmann::json;
    using ordered_json = nlohmann::ordered_json;

    // Saved at most this often while new guesses come in, and on exit.
    static const uint32_t kMaterialSaveIntervalFrames = 600;
    // A probe that never came back (its frame was dropped) may be requested again after this many frames.
    static const uint32_t kMaterialProbeRetryFrames = 300;
    static const size_t kMaxMaterialEntries = 8192;

    struct RtMaterialClassInfo {
        const char *name;
        float roughness;
        float metalness;
        float specular;
        float emission = 0.0f;
    };

    // Defaults per class. Being wrong about a shiny surface looks worse than being wrong about a matte one, so only
    // a few classes get a specular lobe.
    static const RtMaterialClassInfo kClassInfo[size_t(RtMaterialClass::Count)] = {
        { "matte", 0.9f, 0.0f, 0.0f },
        { "sprite", 0.9f, 0.0f, 0.0f },
        { "grass", 0.9f, 0.0f, 0.0f },
        { "dirt", 0.9f, 0.0f, 0.0f },
        { "wood", 0.8f, 0.0f, 0.0f },
        { "cloth", 0.9f, 0.0f, 0.0f },
        { "skin", 0.7f, 0.0f, 0.0f },
        { "stone", 0.55f, 0.0f, 0.35f },
        { "polished", 0.3f, 0.0f, 0.5f },
        { "metal", 0.35f, 0.7f, 0.6f },
        { "water", 0.06f, 0.0f, 1.0f },
        // Glows in its texture's color; never guessed, only set by hand.
        { "lava", 0.7f, 0.0f, 0.2f, 1.5f },
        // Trees and bushes drawn as cutout planes: soft, wrapped lighting (see RtMaterialFoliageFlag). Upright unlit
        // cutouts are treated this way anyway; the class forces it on for a texture.
        { "foliage", 0.9f, 0.0f, 0.0f },
        // Windows (stained glass): god rays stream from them in their own color (see RtMaterialWindowFlag). Never
        // guessed, only set by hand.
        { "window", 0.9f, 0.0f, 0.0f },
    };

    const char *rtMaterialClassName(RtMaterialClass materialClass) {
        return (size_t(materialClass) < size_t(RtMaterialClass::Count)) ? kClassInfo[size_t(materialClass)].name : "matte";
    }

    static RtMaterialClass classFromName(const std::string &name) {
        for (size_t c = 0; c < size_t(RtMaterialClass::Count); c++) {
            if (name == kClassInfo[c].name) {
                return RtMaterialClass(c);
            }
        }

        return RtMaterialClass::Matte;
    }

    uint32_t packRtMaterial(float roughness, float metalness, float specular, RtMaterialClass materialClass) {
        auto unorm = [](float value) { return uint32_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f)); };
        if (specular <= 0.0f) {
            return 0;
        }

        return unorm(roughness) | (unorm(metalness) << 8) | (unorm(specular) << 16) | (uint32_t(materialClass) << 24);
    }

    static void packEntry(RtMaterialEntry &entry) {
        entry.packed = packRtMaterial(entry.roughness, entry.metalness, entry.specular, entry.materialClass);
    }

    static std::string keyString(uint64_t key) {
        char text[32];
        std::snprintf(text, sizeof(text), "%016llX", (unsigned long long)key);
        return text;
    }

    static std::string areaString(int map, int room, const char *name) {
        if (map < 0) {
            return "unknown area";
        }

        char text[160];
        if ((name != nullptr) && (name[0] != '\0')) {
            std::snprintf(text, sizeof(text), "%s (map %02X, room %d)", name, map, room);
        }
        else {
            std::snprintf(text, sizeof(text), "map %02X, room %d", map, room);
        }

        return text;
    }

    static std::string textureString(uint32_t width, uint32_t height, uint8_t format, uint8_t size) {
        static const char *kFormats[] = { "RGBA", "YUV", "CI", "IA", "I" };
        static const char *kSizes[] = { "4-bit", "8-bit", "16-bit", "32-bit" };
        char text[64];
        std::snprintf(text, sizeof(text), "%ux%u %s %s", width, height, (format < 5) ? kFormats[format] : "?", (size < 4) ? kSizes[size] : "?");
        return text;
    }

    static std::string htmlEscape(const std::string &text) {
        std::string out;
        for (char c : text) {
            switch (c) {
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '&': out += "&amp;"; break;
            case '"': out += "&quot;"; break;
            default: out += c; break;
            }
        }

        return out;
    }

    static std::filesystem::path executableDirectory() {
#   if defined(_WIN32)
        wchar_t path[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameW(nullptr, path, MAX_PATH);
        if ((length > 0) && (length < MAX_PATH)) {
            return std::filesystem::path(path).parent_path();
        }
#   elif defined(__linux__)
        // The working directory is usually not the game's folder on Linux (a launcher, a desktop entry).
        std::error_code error;
        const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
        if (!error && !self.empty()) {
            return self.parent_path();
        }
#   endif
        return std::filesystem::current_path();
    }

    // Paths as UTF-8 text and back: a redirected Documents folder or the user name can hold any character.
    static std::string utf8(const std::filesystem::path &path) {
        const auto text = path.u8string();
        return std::string(text.begin(), text.end());
    }

    static std::filesystem::path fromUtf8(const std::string &text) {
#   if defined(__cpp_char8_t)
        return std::filesystem::path(std::u8string(text.begin(), text.end()));
#   else
        return std::filesystem::u8path(text);
#   endif
    }

    // Where the files used to live. Browsers won't let the HTML editors save under AppData, so they moved to
    // Documents (see localDirectory); old files are copied over once.
    static std::filesystem::path legacyDirectory() {
#   if defined(_WIN32)
        const char *localAppData = std::getenv("LOCALAPPDATA");
        if (localAppData != nullptr) {
            return std::filesystem::path(localAppData) / "Quest64Recompiled";
        }
#   endif
        return executableDirectory();
    }

    // Where the files go when the Documents folder can't be found: the game's settings folder,
    // %LOCALAPPDATA%\EltaleReborn-PC\Ray Tracing.
    static std::filesystem::path fallbackDirectory() {
#   if defined(_WIN32)
        const char *localAppData = std::getenv("LOCALAPPDATA");
        if (localAppData != nullptr) {
            return std::filesystem::path(localAppData) / "EltaleReborn-PC" / "Ray Tracing";
        }
#   endif
        return executableDirectory();
    }

    // Documents\Quest64Recompiled\Ray Tracing, from the Documents known folder so a redirected (OneDrive) Documents
    // works. Chromium's file pickers allow folders below Documents, unlike AppData.
    static std::filesystem::path localDirectory() {
#   if defined(_WIN32)
        PWSTR documents = nullptr;
        if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, KF_FLAG_DEFAULT, nullptr, &documents)) && (documents != nullptr)) {
            const std::filesystem::path path = std::filesystem::path(documents) / "Quest64Recompiled" / "Ray Tracing";
            CoTaskMemFree(documents);
            return path;
        }

        if (documents != nullptr) {
            CoTaskMemFree(documents);
        }
#   endif
        return fallbackDirectory();
    }

    // Copies the material and emitter files and their previews from the old AppData folder when the new folder has
    // none yet. The originals stay where they were.
    static void migrateLegacyFiles(const std::filesystem::path &directory) {
        const std::filesystem::path legacy = legacyDirectory();
        std::error_code error;
        if ((legacy == directory) || std::filesystem::exists(directory / "rt_materials.json", error) || std::filesystem::exists(directory / "rt_emitters.json", error)) {
            return;
        }

        if (!std::filesystem::exists(legacy / "rt_materials.json", error) && !std::filesystem::exists(legacy / "rt_emitters.json", error)) {
            return;
        }

        std::filesystem::create_directories(directory, error);
        uint32_t copied = 0;
        for (const char *name : { "rt_materials.json", "rt_emitters.json" }) {
            if (std::filesystem::exists(legacy / name, error)) {
                std::filesystem::copy_file(legacy / name, directory / name, std::filesystem::copy_options::skip_existing, error);
                copied += error ? 0 : 1;
            }
        }

        if (std::filesystem::exists(legacy / "rt_materials", error)) {
            std::filesystem::copy(legacy / "rt_materials", directory / "rt_materials", std::filesystem::copy_options::recursive | std::filesystem::copy_options::skip_existing, error);
        }

        rtDiagLog("materials: copied %u files and the previews from %s to %s (the originals are kept)", copied, utf8(legacy).c_str(), utf8(directory).c_str());
    }

    static bool readEntriesDocument(const json &document, std::unordered_map<uint64_t, RtMaterialEntry> &out) {
        if (!document.is_object() || !document.contains("materials")) {
            return false;
        }

        auto readEntry = [&](const std::string &keyText, const json &value) {
            if (!value.is_object()) {
                return;
            }

            const std::string text = value.value("key", keyText);
            char *end = nullptr;
            const uint64_t key = std::strtoull(text.c_str(), &end, 16);
            if ((end == text.c_str()) || (key == 0)) {
                return;
            }

            RtMaterialEntry entry;
            entry.label = value.value("label", std::string());
            entry.materialClass = classFromName(value.value("class", std::string("matte")));
            const RtMaterialClassInfo &info = kClassInfo[size_t(entry.materialClass)];
            entry.roughness = value.value("roughness", info.roughness);
            entry.metalness = value.value("metalness", info.metalness);
            entry.specular = value.value("specular", info.specular);
            // "emission": strength (the class default when missing); "emission color": "albedo" (the texture's color)
            // or [r, g, b].
            entry.emission = info.emission;
            const auto emission = value.find("emission");
            if ((emission != value.end()) && emission->is_number()) {
                entry.emission = std::clamp(emission->get<float>(), 0.0f, 8.0f);
            }

            const auto emissionColor = value.find("emission color");
            if ((emissionColor != value.end()) && emissionColor->is_array() && (emissionColor->size() >= 3) &&
                (*emissionColor)[0].is_number() && (*emissionColor)[1].is_number() && (*emissionColor)[2].is_number())
            {
                entry.emissionFromTexture = false;
                for (int c = 0; c < 3; c++) {
                    entry.emissionColor[c] = std::clamp((*emissionColor)[c].get<float>(), 0.0f, 1.0f);
                }
            }

            entry.user = (value.value("source", std::string("guess")) == "user");
            entry.surface = value.value("on", std::string());
            entry.area = value.value("first seen", std::string());
            entry.map = value.value("map", -1);
            entry.room = value.value("room", -1);
            entry.texture = value.value("texture", std::string());
            entry.preview = value.value("preview", std::string());
            entry.stats = value.value("stats", std::string());
            float average[3];
            const size_t averageAt = entry.stats.find("average color ");
            if ((averageAt != std::string::npos) && (std::sscanf(entry.stats.c_str() + averageAt, "average color %f %f %f", &average[0], &average[1], &average[2]) == 3)) {
                for (int c = 0; c < 3; c++) {
                    entry.averageColor[c] = std::clamp(average[c], 0.0f, 1.0f);
                }
            }

            packEntry(entry);
            out[key] = entry;
        };

        // Entries are grouped by area in an array; the first version of the file was an object by key.
        const json &materials = document["materials"];
        if (materials.is_array()) {
            for (const json &value : materials) {
                try {
                    readEntry(std::string(), value);
                }
                catch (const std::exception &e) {
                    rtDiagLog("materials: skipped an entry that couldn't be read: %s", e.what());
                }
            }
        }
        else if (materials.is_object()) {
            for (auto &item : materials.items()) {
                try {
                    readEntry(item.key(), item.value());
                }
                catch (const std::exception &e) {
                    rtDiagLog("materials: skipped entry %s that couldn't be read: %s", item.key().c_str(), e.what());
                }
            }
        }

        return true;
    }

    static bool readEntries(const std::filesystem::path &path, std::unordered_map<uint64_t, RtMaterialEntry> &out) {
        std::ifstream stream(path);
        if (!stream.is_open()) {
            return false;
        }

        json document = json::parse(stream, nullptr, false);
        if (document.is_discarded() || !readEntriesDocument(document, out)) {
            rtDiagLog("materials: %s is not valid material JSON, ignored", utf8(path).c_str());
            return false;
        }

        return true;
    }

    struct EmbeddedConfig {
        const char *data = nullptr;
        size_t size = 0;
    };

    static std::mutex embeddedConfigMutex;
    static std::unordered_map<std::string, EmbeddedConfig> embeddedConfigs;

    void setRaytracingShippedConfig(const char *fileName, const char *data, size_t size) {
        if ((fileName == nullptr) || (data == nullptr) || (size == 0)) {
            return;
        }

        std::scoped_lock lock(embeddedConfigMutex);
        embeddedConfigs[fileName] = { data, size };
    }

    // A shipped file: the copy built into the game when there is one, as it always matches this build, otherwise the
    // file in assets\materials. origin says which was read, or why neither could be.
    static bool parseShipped(const std::filesystem::path &path, json &document, std::string &origin) {
        EmbeddedConfig embedded;
        {
            std::scoped_lock lock(embeddedConfigMutex);
            auto it = embeddedConfigs.find(utf8(path.filename()));
            if (it != embeddedConfigs.end()) {
                embedded = it->second;
            }
        }

        if (embedded.data != nullptr) {
            document = json::parse(embedded.data, embedded.data + embedded.size, nullptr, false);
            if (!document.is_discarded()) {
                origin = "the copy built into the game";
                return true;
            }

            rtDiagLog("ray tracing config: the copy of %s built into the game is not valid JSON, trying %s", utf8(path.filename()).c_str(), utf8(path).c_str());
        }

        std::ifstream stream(path);
        if (!stream.is_open()) {
            origin = utf8(path) + " (NOT FOUND, and no copy is built into the game)";
            return false;
        }

        document = json::parse(stream, nullptr, false);
        if (document.is_discarded()) {
            origin = utf8(path) + " (NOT VALID JSON)";
            return false;
        }

        origin = utf8(path);
        return true;
    }

    static bool readShippedEntries(const std::filesystem::path &path, std::unordered_map<uint64_t, RtMaterialEntry> &out, std::string &origin) {
        json document;
        if (!parseShipped(path, document, origin)) {
            return false;
        }

        if (!readEntriesDocument(document, out)) {
            origin += " (NOT VALID material JSON)";
            return false;
        }

        return true;
    }

    static const char *emitterKindName(uint32_t kind) {
        switch (kind) {
        case RtEmitterKindAdditive: return "additive";
        case RtEmitterKindBlended: return "blended";
        case RtEmitterKindCutout: return "cutout";
        case RtEmitterKindUntextured: return "untextured";
        case RtEmitterKindOpaque: return "opaque";
        default: return "";
        }
    }

    static uint32_t emitterKindFromName(const std::string &name) {
        for (uint32_t bit = 1; bit <= RtEmitterKindOpaque; bit <<= 1) {
            if (name == emitterKindName(bit)) {
                return bit;
            }
        }

        return 0;
    }

    static std::string hex32(uint32_t value) {
        char text[16];
        std::snprintf(text, sizeof(text), "%08X", value);
        return text;
    }

    static uint64_t parseHex(const json &value) {
        if (value.is_string()) {
            const std::string text = value.get<std::string>();
            if ((text == "none") || text.empty()) {
                return 0;
            }

            return std::strtoull(text.c_str(), nullptr, 16);
        }

        return value.is_number_unsigned() ? value.get<uint64_t>() : 0;
    }

    static void readColor(const json &value, float *out, int count) {
        for (int c = 0; c < count; c++) {
            out[c] = (value.is_array() && (int(value.size()) > c) && value[c].is_number()) ? value[c].get<float>() : 0.0f;
        }
    }

    static void readSignature(const json &value, RtEmitterSignature &signature) {
        signature.signature = value.contains("signature") ? parseHex(value["signature"]) : 0;
        signature.texture = value.contains("texture") ? parseHex(value["texture"]) : 0;
        signature.vertexAddress = value.contains("vertex address") ? uint32_t(parseHex(value["vertex address"])) : 0;
        if (value.contains("combiner") && value["combiner"].is_array() && (value["combiner"].size() >= 2)) {
            signature.combiner[0] = uint32_t(parseHex(value["combiner"][0]));
            signature.combiner[1] = uint32_t(parseHex(value["combiner"][1]));
        }

        if (value.contains("other mode") && value["other mode"].is_array() && (value["other mode"].size() >= 2)) {
            signature.otherMode[0] = uint32_t(parseHex(value["other mode"][0]));
            signature.otherMode[1] = uint32_t(parseHex(value["other mode"][1]));
        }

        if (value.contains("prim color")) readColor(value["prim color"], signature.prim, 4);
        if (value.contains("env color")) readColor(value["env color"], signature.env, 4);
        if (value.contains("vertex color")) readColor(value["vertex color"], signature.vertexColor, 3);
        // A count in "observed", a [min, max] range in "match".
        signature.triangles = 0;
        if (value.contains("triangles")) {
            const json &triangles = value["triangles"];
            if (triangles.is_number()) {
                signature.triangles = uint32_t(std::max(triangles.get<double>(), 0.0));
            }
            else if (triangles.is_array() && !triangles.empty() && triangles[0].is_number()) {
                signature.triangles = uint32_t(std::max(triangles[0].get<double>(), 0.0));
            }
        }

        signature.kind = (value.contains("kind") && value["kind"].is_string()) ? emitterKindFromName(value["kind"].get<std::string>()) : 0;
    }

    static ordered_json signatureJson(const RtEmitterSignature &signature) {
        auto color = [](const float *values, int count) {
            ordered_json array = ordered_json::array();
            for (int c = 0; c < count; c++) {
                array.push_back(std::round(values[c] * 1000.0f) / 1000.0f);
            }

            return array;
        };

        ordered_json value = ordered_json::object();
        value["signature"] = keyString(signature.signature);
        value["texture"] = (signature.texture != 0) ? keyString(signature.texture) : std::string("none");
        value["vertex address"] = hex32(signature.vertexAddress);
        value["combiner"] = { hex32(signature.combiner[0]), hex32(signature.combiner[1]) };
        value["other mode"] = { hex32(signature.otherMode[0]), hex32(signature.otherMode[1]) };
        value["prim color"] = color(signature.prim, 4);
        value["env color"] = color(signature.env, 4);
        value["vertex color"] = color(signature.vertexColor, 3);
        value["triangles"] = signature.triangles;
        value["kind"] = emitterKindName(signature.kind);
        return value;
    }

    // The "match" object of an entry: only the fields present must match.
    static void readMatch(const json &value, RtEmitterMatch &match) {
        RtEmitterSignature &v = match.value;
        readSignature(value, v);
        match.signature = value.contains("signature");
        match.texture = value.contains("texture");
        match.vertexAddress = value.contains("vertex address");
        match.combiner = value.contains("combiner");
        match.otherMode = value.contains("other mode");
        match.prim = value.contains("prim color");
        match.env = value.contains("env color");
        match.vertexColor = value.contains("vertex color");
        match.colorTolerance = value.value("color tolerance", 0.1f);
        if (value.contains("triangles") && value["triangles"].is_array() && (value["triangles"].size() >= 2)) {
            match.triangleRange = true;
            match.minTriangles = value["triangles"][0].is_number() ? uint32_t(std::max(value["triangles"][0].get<double>(), 0.0)) : 0;
            match.maxTriangles = value["triangles"][1].is_number() ? uint32_t(std::max(value["triangles"][1].get<double>(), 0.0)) : 0;
        }
    }

    static ordered_json matchJson(const RtEmitterMatch &match) {
        const ordered_json all = signatureJson(match.value);
        ordered_json value = ordered_json::object();
        if (match.signature) value["signature"] = all["signature"];
        if (match.texture) value["texture"] = all["texture"];
        if (match.vertexAddress) value["vertex address"] = all["vertex address"];
        if (match.combiner) value["combiner"] = all["combiner"];
        if (match.otherMode) value["other mode"] = all["other mode"];
        if (match.prim) value["prim color"] = all["prim color"];
        if (match.env) value["env color"] = all["env color"];
        if (match.vertexColor) value["vertex color"] = all["vertex color"];
        if (match.prim || match.env || match.vertexColor) value["color tolerance"] = match.colorTolerance;
        if (match.triangleRange) value["triangles"] = { match.minTriangles, match.maxTriangles };
        return value;
    }

    static bool readEmitterEntry(const json &value, bool listedSection, uint64_t &id, RtEmitterEntry &entry) {
        if (!value.is_object()) {
            return false;
        }

        RtEmitterParams &params = entry.params;
        if (value.contains("match") && value["match"].is_object()) {
            readMatch(value["match"], params.match);
        }
        else if (value.contains("key")) {
            // The texture-only format: the key is the texture.
            params.match.texture = true;
            params.match.value.texture = parseHex(value["key"]);
        }

        id = value.contains("id") ? parseHex(value["id"]) : 0;
        if (id == 0) {
            id = params.match.signature ? params.match.value.signature : params.match.value.texture;
        }

        if ((id == 0) || !params.match.any()) {
            return false;
        }

        params.id = id;
        entry.label = value.value("label", std::string());
        entry.source = value.value("source", std::string(listedSection ? "shipped" : "game"));
        params.enabled = value.value("enabled", listedSection);
        // A candidate switched on in place counts as listed.
        entry.listed = listedSection || params.enabled;
        params.sampleColor = true;
        if (value.contains("color") && value["color"].is_array() && (value["color"].size() >= 3)) {
            params.sampleColor = false;
            readColor(value["color"], params.color, 3);
        }

        params.intensity = value.value("intensity", 1.0f);
        params.rangeScale = value.value("range scale", 1.0f);
        params.maxLights = uint32_t(std::max(value.value("max lights", 0), 0));
        params.fadeInFrames = uint32_t(std::max(value.value("fade in frames", 0), 0));
        params.fadeOutFrames = uint32_t(std::max(value.value("fade out frames", 0), 0));
        params.kinds = 0;
        if (value.contains("only kinds") && value["only kinds"].is_array()) {
            for (const json &kind : value["only kinds"]) {
                if (kind.is_string()) {
                    params.kinds |= emitterKindFromName(kind.get<std::string>());
                }
            }
        }

        params.areaCount = 0;
        if (value.contains("only areas") && value["only areas"].is_array()) {
            for (const json &area : value["only areas"]) {
                if (area.is_object() && (params.areaCount < RtEmitterParams::MaxAreas)) {
                    params.areaMaps[params.areaCount] = area.value("map", -1);
                    params.areaRooms[params.areaCount] = area.value("room", -1);
                    params.areaCount++;
                }
            }
        }

        entry.hasObserved = value.contains("observed") && value["observed"].is_object();
        if (entry.hasObserved) {
            readSignature(value["observed"], entry.observed);
        }

        entry.wouldGlow = (value.value("old heuristic", std::string()) == "would glow");
        entry.kind = value.value("kind", std::string());
        entry.area = value.value("first seen", std::string());
        entry.map = value.value("map", -1);
        entry.room = value.value("room", -1);
        entry.size = value.value("size", std::string());
        entry.preview = value.value("preview", std::string());
        return true;
    }

    static const char *kLegacyGlowSetting = "legacy glow for untextured effects";

    static bool readEmittersDocument(const json &document, std::unordered_map<uint64_t, RtEmitterEntry> &out, int *legacyGlow) {
        if (!document.is_object()) {
            return false;
        }

        if ((legacyGlow != nullptr) && document.contains(kLegacyGlowSetting) && document[kLegacyGlowSetting].is_boolean()) {
            *legacyGlow = document[kLegacyGlowSetting].get<bool>() ? 1 : 0;
        }

        auto readList = [&](const char *name, bool listedSection) {
            if (!document.contains(name) || !document[name].is_array()) {
                return;
            }

            for (const json &value : document[name]) {
                // A malformed entry is skipped, never fatal.
                try {
                    uint64_t id = 0;
                    RtEmitterEntry entry;
                    if (readEmitterEntry(value, listedSection, id, entry)) {
                        out[id] = entry;
                    }
                }
                catch (const std::exception &e) {
                    rtDiagLog("emitters: skipped an entry in \"%s\" that couldn't be read: %s", name, e.what());
                }
            }
        };

        readList("candidates", false);
        readList("emitters", true);
        return true;
    }

    static bool readEmitters(const std::filesystem::path &path, std::unordered_map<uint64_t, RtEmitterEntry> &out, int *legacyGlow = nullptr) {
        std::ifstream stream(path);
        if (!stream.is_open()) {
            return false;
        }

        json document = json::parse(stream, nullptr, false);
        if (document.is_discarded() || !readEmittersDocument(document, out, legacyGlow)) {
            rtDiagLog("emitters: %s is not valid JSON, ignored", utf8(path).c_str());
            return false;
        }

        return true;
    }

    static bool readShippedEmitters(const std::filesystem::path &path, std::unordered_map<uint64_t, RtEmitterEntry> &out, int *legacyGlow, std::string &origin) {
        json document;
        if (!parseShipped(path, document, origin)) {
            return false;
        }

        if (!readEmittersDocument(document, out, legacyGlow)) {
            origin += " (NOT VALID emitter JSON)";
            return false;
        }

        return true;
    }

    static ordered_json emitterJson(uint64_t id, const RtEmitterEntry &entry) {
        const RtEmitterParams &params = entry.params;
        ordered_json value = ordered_json::object();
        value["id"] = keyString(id);
        value["label"] = entry.label;
        value["source"] = entry.source;
        value["enabled"] = params.enabled;
        value["match"] = matchJson(params.match);
        if (params.sampleColor) {
            value["color"] = "sample";
        }
        else {
            value["color"] = { params.color[0], params.color[1], params.color[2] };
        }

        value["intensity"] = params.intensity;
        value["range scale"] = params.rangeScale;
        value["max lights"] = params.maxLights;
        value["fade in frames"] = params.fadeInFrames;
        value["fade out frames"] = params.fadeOutFrames;
        ordered_json kinds = ordered_json::array();
        for (uint32_t bit = 1; bit <= RtEmitterKindOpaque; bit <<= 1) {
            if (params.kinds & bit) {
                kinds.push_back(emitterKindName(bit));
            }
        }

        value["only kinds"] = kinds;
        ordered_json areas = ordered_json::array();
        for (uint32_t a = 0; a < params.areaCount; a++) {
            areas.push_back({ { "map", params.areaMaps[a] }, { "room", params.areaRooms[a] } });
        }

        value["only areas"] = areas;
        if (entry.hasObserved) {
            value["observed"] = signatureJson(entry.observed);
        }

        if (entry.wouldGlow) {
            value["old heuristic"] = "would glow";
        }

        value["kind"] = entry.kind;
        value["first seen"] = entry.area;
        value["map"] = entry.map;
        value["room"] = entry.room;
        value["size"] = entry.size;
        value["preview"] = entry.preview;
        return value;
    }

    // rt_lights.json is re-read when it changes, so lights placed in Eltale Studio show up without restarting. Set to
    // false to read it only at boot like the other files.
    static const bool kManualLightsHotReload = true;
    static const std::chrono::milliseconds kManualLightsPollInterval(1000);

    // rt_materials.json's user entries are read again when another program (Eltale Studio, the HTML editor) changes it,
    // so an edited material shows in the game within a second.
    static const bool kMaterialsHotReload = true;

    // A room setting's "fog" object, or one of its "times" keyframes over the base fog. Missing fields keep the base's.
    static RtFogParams readFogParams(const json &value, const RtFogParams &base) {
        auto number = [&](const char *key, float fallback, float lo, float hi) {
            return (value.contains(key) && value[key].is_number()) ? std::clamp(value[key].get<float>(), lo, hi) : fallback;
        };

        RtFogParams fog = base;
        fog.density = number("density", base.density, 0.0f, 1.0f);
        if (value.contains("color") && value["color"].is_array()) {
            readColor(value["color"], fog.color, 3);
            for (float &c : fog.color) {
                c = std::clamp(c, 0.0f, 4.0f);
            }
        }

        fog.ambient = number("ambient", base.ambient, 0.0f, 4.0f);
        fog.sun = number("sun", base.sun, 0.0f, 10.0f);
        fog.lights = number("lights", base.lights, 0.0f, 10.0f);
        fog.shadows = number("shadows", base.shadows, 0.0f, 4.0f);
        fog.anisotropy = number("anisotropy", base.anisotropy, -0.9f, 0.9f);
        fog.heightFalloff = number("height falloff", base.heightFalloff, 0.0f, 1.0f);
        fog.baseHeight = number("base height", base.baseHeight, -100000.0f, 100000.0f);
        fog.distance = number("distance", base.distance, 10.0f, 100000.0f);
        fog.gameFog = number("game fog", base.gameFog, 0.0f, 1.0f);
        fog.rays = number("rays", base.rays, 0.0f, 8.0f);
        fog.raysLength = number("rays length", base.raysLength, 0.05f, 1.5f);
        if (value.contains("rays color") && value["rays color"].is_array()) {
            readColor(value["rays color"], fog.raysColor, 3);
            for (float &c : fog.raysColor) {
                c = std::clamp(c, 0.0f, 4.0f);
            }
        }

        fog.windows = number("windows", base.windows, 0.0f, 8.0f);
        fog.rayAzimuth = number("ray azimuth", base.rayAzimuth, -360.0f, 360.0f);
        fog.rayElevation = number("ray elevation", base.rayElevation, -90.0f, 90.0f);
        fog.windowBias = number("window bias", base.windowBias, 0.0f, 1.0f);
        return fog;
    }

    static RtFogParams lerpFog(const RtFogParams &a, const RtFogParams &b, float t) {
        auto mix = [t](float x, float y) { return x + (y - x) * t; };
        RtFogParams fog;
        fog.density = mix(a.density, b.density);
        for (int c = 0; c < 3; c++) {
            fog.color[c] = mix(a.color[c], b.color[c]);
        }

        fog.ambient = mix(a.ambient, b.ambient);
        fog.sun = mix(a.sun, b.sun);
        fog.lights = mix(a.lights, b.lights);
        fog.shadows = mix(a.shadows, b.shadows);
        fog.anisotropy = mix(a.anisotropy, b.anisotropy);
        fog.heightFalloff = mix(a.heightFalloff, b.heightFalloff);
        fog.baseHeight = mix(a.baseHeight, b.baseHeight);
        fog.distance = mix(a.distance, b.distance);
        fog.gameFog = mix(a.gameFog, b.gameFog);
        fog.rays = mix(a.rays, b.rays);
        fog.raysLength = mix(a.raysLength, b.raysLength);
        for (int c = 0; c < 3; c++) {
            fog.raysColor[c] = mix(a.raysColor[c], b.raysColor[c]);
        }

        fog.windows = mix(a.windows, b.windows);
        fog.rayAzimuth = mix(a.rayAzimuth, b.rayAzimuth);
        fog.rayElevation = mix(a.rayElevation, b.rayElevation);
        fog.windowBias = mix(a.windowBias, b.windowBias);
        return fog;
    }

    // The fog of a setting at an hour of the game clock: its keyframes interpolated around the clock (after the last
    // one comes the first one of the next day).
    static RtFogParams fogAtHour(const RtRoomSetting &setting, float hour) {
        const std::vector<RtFogKey> &keys = setting.fogTimes;
        if (keys.empty() || (hour < 0.0f)) {
            return setting.fog;
        }

        if (keys.size() == 1) {
            return keys[0].params;
        }

        size_t next = 0;
        while ((next < keys.size()) && (keys[next].hour <= hour)) {
            next++;
        }

        const RtFogKey &after = keys[next % keys.size()];
        const RtFogKey &before = keys[(next + keys.size() - 1) % keys.size()];
        float span = after.hour - before.hour;
        float into = hour - before.hour;
        if (span <= 0.0f) {
            span += 24.0f;
        }

        if (into < 0.0f) {
            into += 24.0f;
        }

        return lerpFog(before.params, after.params, (span > 1e-4f) ? std::clamp(into / span, 0.0f, 1.0f) : 0.0f);
    }

    // Reads the "lights" list of an rt_lights.json; false when the file is missing or not valid JSON. Bad entries are
    // skipped.
    static bool readManualLightsDocument(const json &document, const std::string &source, std::vector<RtManualLight> &out, std::vector<RtRoomSetting> &settings, bool fromLocal) {
        if (!document.is_object() || !document.contains("lights") || !document["lights"].is_array()) {
            rtDiagLog("manual lights: %s has no \"lights\" list, ignored", source.c_str());
            return false;
        }

        auto number = [](const json &value, const char *key, float fallback) {
            return (value.contains(key) && value[key].is_number()) ? value[key].get<float>() : fallback;
        };

        // "room settings": [{ "map": 21, "room": 0, "sun and moon": false }], room -1 for every room of the map.
        if (document.contains("room settings") && document["room settings"].is_array()) {
            for (const json &value : document["room settings"]) {
                try {
                    if (!value.is_object() || !value.contains("map") || !value["map"].is_number()) {
                        continue;
                    }

                    RtRoomSetting setting;
                    setting.map = int(number(value, "map", -1.0f));
                    setting.room = int(number(value, "room", -1.0f));
                    setting.hasSunAndMoon = value.contains("sun and moon") && value["sun and moon"].is_boolean();
                    setting.sunAndMoon = !(setting.hasSunAndMoon && !value["sun and moon"].get<bool>());
                    setting.fromLocal = fromLocal;
                    // "fog": { "density": 0.002, "color": [r, g, b], ..., "times": [{ "hour": 6, "density": 0.004 }, ...] }
                    if (value.contains("fog") && value["fog"].is_object()) {
                        const json &fog = value["fog"];
                        setting.hasFog = true;
                        setting.fog = readFogParams(fog, RtFogParams());
                        if (fog.contains("times") && fog["times"].is_array()) {
                            for (const json &time : fog["times"]) {
                                if (!time.is_object() || !time.contains("hour") || !time["hour"].is_number()) {
                                    continue;
                                }

                                RtFogKey key;
                                key.hour = std::fmod(std::fmod(time["hour"].get<float>(), 24.0f) + 24.0f, 24.0f);
                                key.params = readFogParams(time, setting.fog);
                                setting.fogTimes.push_back(key);
                            }

                            std::sort(setting.fogTimes.begin(), setting.fogTimes.end(), [](const RtFogKey &a, const RtFogKey &b) { return a.hour < b.hour; });
                        }
                    }
                    settings.push_back(setting);
                }
                catch (const std::exception &e) {
                    rtDiagLog("manual lights: skipped a room setting in %s that couldn't be read: %s", source.c_str(), e.what());
                }
            }
        }

        for (const json &value : document["lights"]) {
            try {
                if (!value.is_object() || !value.contains("id") || !value["id"].is_string()) {
                    continue;
                }

                RtManualLight light;
                light.id = value["id"].get<std::string>();
                light.enabled = value.contains("enabled") && value["enabled"].is_boolean() && value["enabled"].get<bool>();
                light.map = int(number(value, "map", -1.0f));
                light.room = int(number(value, "room", -1.0f));
                if (value.contains("position")) readColor(value["position"], light.position, 3);
                light.color[0] = light.color[1] = light.color[2] = 1.0f;
                if (value.contains("color")) readColor(value["color"], light.color, 3);
                for (float &c : light.color) {
                    c = std::clamp(c, 0.0f, 1.0f);
                }

                light.intensity = std::max(number(value, "intensity", 1.0f), 0.0f);
                light.range = std::clamp(number(value, "range", 150.0f), 1.0f, 2000.0f);
                light.sourceRadius = std::max(number(value, "source radius", 4.0f), 0.5f);
                light.castsShadows = !(value.contains("casts shadows") && value["casts shadows"].is_boolean() && !value["casts shadows"].get<bool>());
                light.flicker = std::clamp(number(value, "flicker", 0.0f), 0.0f, 1.0f);
                light.flickerSpeed = std::max(number(value, "flicker speed", 1.0f), 0.0f);
                uint32_t hash = 2166136261u;
                for (char c : light.id) {
                    hash = (hash ^ uint8_t(c)) * 16777619u;
                }

                light.flickerPhase = float(hash % 6283u) / 1000.0f;
                out.push_back(light);
            }
            catch (const std::exception &e) {
                rtDiagLog("manual lights: skipped a light in %s that couldn't be read: %s", source.c_str(), e.what());
            }
        }

        return true;
    }

    static bool readManualLights(const std::filesystem::path &path, std::vector<RtManualLight> &out, std::vector<RtRoomSetting> &settings, bool fromLocal) {
        std::ifstream stream(path);
        if (!stream.is_open()) {
            return false;
        }

        json document = json::parse(stream, nullptr, false);
        if (document.is_discarded()) {
            rtDiagLog("manual lights: %s is not valid JSON, ignored", utf8(path).c_str());
            return false;
        }

        return readManualLightsDocument(document, utf8(path), out, settings, fromLocal);
    }

    static bool readShippedManualLights(const std::filesystem::path &path, std::vector<RtManualLight> &out, std::vector<RtRoomSetting> &settings, std::string &origin) {
        json document;
        if (!parseShipped(path, document, origin)) {
            return false;
        }

        if (!readManualLightsDocument(document, origin, out, settings, false)) {
            origin += " (NO \"lights\" LIST)";
            return false;
        }

        return true;
    }

    std::shared_ptr<const std::vector<RtManualLight>> RtMaterialDatabase::manualLightList() {
        std::scoped_lock lock(mutex);
        return manualLights;
    }

    bool RtMaterialDatabase::sunAndMoonEnabled(int map, int room, const char *&source, const char *&scope) {
        source = "default";
        scope = "no entry";
        if (map < 0) {
            return true;
        }

        std::scoped_lock lock(mutex);
        const RtRoomSetting *mapWide = nullptr;
        for (const RtRoomSetting &setting : roomSettings) {
            if ((setting.map != map) || !setting.hasSunAndMoon) {
                continue;
            }

            if ((setting.room >= 0) && (setting.room == room)) {
                source = setting.fromLocal ? "local" : "shipped";
                scope = "room entry";
                return setting.sunAndMoon;
            }

            if (setting.room < 0) {
                mapWide = &setting;
            }
        }

        if (mapWide != nullptr) {
            source = mapWide->fromLocal ? "local" : "shipped";
            scope = "map-wide";
            return mapWide->sunAndMoon;
        }

        return true;
    }

    bool RtMaterialDatabase::volumetricFog(int map, int room, float hour, RtFogParams &fog, const char *&source, const char *&scope) {
        source = "default";
        scope = "no entry";
        fog = RtFogParams();
        if (map < 0) {
            return false;
        }

        std::scoped_lock lock(mutex);
        const RtRoomSetting *chosen = nullptr;
        for (const RtRoomSetting &setting : roomSettings) {
            if ((setting.map != map) || !setting.hasFog) {
                continue;
            }

            if ((setting.room >= 0) && (setting.room == room)) {
                chosen = &setting;
                scope = "room entry";
                break;
            }

            if (setting.room < 0) {
                chosen = &setting;
                scope = "map-wide";
            }
        }

        if (chosen == nullptr) {
            return false;
        }

        source = chosen->fromLocal ? "local" : "shipped";
        fog = fogAtHour(*chosen, hour);
        return true;
    }

    std::string RtMaterialDatabase::manualLightsPath() {
        std::scoped_lock lock(mutex);
        return manualLightsFile;
    }

    void RtMaterialDatabase::loadManualLights(bool onlyIfChanged) {
        // Players read only the shipped list; in authoring mode the Documents file overrides it and is hot reloaded.
        const bool authoringMode = authoring;
        const std::filesystem::path local = localDirectory() / "rt_lights.json";
        const std::filesystem::path shipped = executableDirectory() / "assets" / "materials" / "rt_lights.json";
        std::error_code error;
        if (!authoringMode && onlyIfChanged) {
            return;
        }

        const std::filesystem::file_time_type writeTime = authoringMode ? std::filesystem::last_write_time(local, error) : std::filesystem::file_time_type{};
        if (onlyIfChanged) {
            std::scoped_lock lock(mutex);
            if (error || (writeTime == manualLightsWriteTime)) {
                return;
            }
        }

        // The local file's lights win by id, even when disabled; the shipped list adds the ids it doesn't have.
        std::vector<RtManualLight> localLights;
        std::vector<RtRoomSetting> localSettings;
        const bool localExists = authoringMode && std::filesystem::exists(local, error);
        const bool localRead = authoringMode && readManualLights(local, localLights, localSettings, true);
        if (onlyIfChanged && localExists && !localRead) {
            // Probably caught mid-save or broken by hand: keep the lights we have.
            std::scoped_lock lock(mutex);
            manualLightsWriteTime = writeTime;
            return;
        }

        std::vector<RtManualLight> shippedLights;
        std::vector<RtRoomSetting> shippedSettings;
        std::string shippedOrigin;
        const bool shippedRead = readShippedManualLights(shipped, shippedLights, shippedSettings, shippedOrigin);

        // Room settings by (map, room); the local file wins.
        std::vector<RtRoomSetting> mergedSettings = localSettings;
        for (const RtRoomSetting &setting : shippedSettings) {
            const bool known = std::any_of(mergedSettings.begin(), mergedSettings.end(), [&](const RtRoomSetting &other) { return (other.map == setting.map) && (other.room == setting.room); });
            if (!known) {
                mergedSettings.push_back(setting);
            }
        }
        auto merged = std::make_shared<std::vector<RtManualLight>>(localLights);
        for (const RtManualLight &light : shippedLights) {
            const bool known = std::any_of(merged->begin(), merged->end(), [&](const RtManualLight &other) { return other.id == light.id; });
            if (!known) {
                merged->push_back(light);
            }
        }

        const size_t enabled = size_t(std::count_if(merged->begin(), merged->end(), [](const RtManualLight &light) { return light.enabled; }));
        {
            std::scoped_lock lock(mutex);
            manualLights = merged;
            roomSettings = mergedSettings;
            manualLightsFile = authoringMode ? utf8(local) : utf8(shipped);
            manualLightsWriteTime = writeTime;
        }

        const size_t sunOff = size_t(std::count_if(mergedSettings.begin(), mergedSettings.end(), [](const RtRoomSetting &setting) { return !setting.sunAndMoon; }));
        const size_t withFog = size_t(std::count_if(mergedSettings.begin(), mergedSettings.end(), [](const RtRoomSetting &setting) { return setting.hasFog; }));
        rtDiagLog("manual lights: %zu (%zu enabled; %zu from %s, %zu shipped from %s%s); room settings: %zu (%zu with the sun and moon off, %zu with fog)%s", merged->size(), enabled,
            localLights.size(), authoringMode ? utf8(local).c_str() : "no Documents file", shippedLights.size(), shippedOrigin.c_str(), shippedRead ? "" : " - SHIPPED LIGHTS MISSING",
            mergedSettings.size(), sunOff, withFog, authoringMode ? (localRead ? " (authoring mode: the Documents file wins by id)" : " (authoring mode, Documents file not found, using the shipped list)") : " (player mode: shipped configuration only)");

        // Enabled and total lights per area, and the rooms without the sun and moon, so a player's log shows what loaded.
        if (!onlyIfChanged) {
            std::map<std::pair<int, int>, std::pair<uint32_t, uint32_t>> areas;
            for (const RtManualLight &light : *merged) {
                auto &counts = areas[{ light.map, light.room }];
                counts.first += light.enabled ? 1 : 0;
                counts.second++;
            }

            std::string line;
            uint32_t inLine = 0;
            for (const auto &area : areas) {
                char text[64];
                snprintf(text, sizeof(text), "%s%d/%d %u of %u", line.empty() ? "" : ", ", area.first.first, area.first.second, area.second.first, area.second.second);
                line += text;
                if (++inLine == 16) {
                    rtDiagLog("manual lights per area (map/room, room -1 = all rooms: enabled of total): %s", line.c_str());
                    line.clear();
                    inLine = 0;
                }
            }

            if (!line.empty()) {
                rtDiagLog("manual lights per area (map/room, room -1 = all rooms: enabled of total): %s", line.c_str());
            }

            std::string off;
            for (const RtRoomSetting &setting : mergedSettings) {
                if (!setting.sunAndMoon) {
                    char text[32];
                    snprintf(text, sizeof(text), "%s%d/%d", off.empty() ? "" : ", ", setting.map, setting.room);
                    off += text;
                }
            }

            rtDiagLog("room settings with the sun and moon off (map/room): %s", off.empty() ? "none" : off.c_str());
        }
    }

    void RtMaterialDatabase::pollEditedFiles() {
        if ((!kManualLightsHotReload && !kMaterialsHotReload) || !loaded || !authoring) {
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        if (((now - editedFilesLastPoll) < kManualLightsPollInterval) || editedFilesCheckPending.exchange(true)) {
            return;
        }

        editedFilesLastPoll = now;
        post([this]() {
            struct ClearPending {
                std::atomic<bool> &flag;
                ~ClearPending() { flag = false; }
            } clearPending{ editedFilesCheckPending };
            if (kManualLightsHotReload) {
                loadManualLights(true);
            }

            if (kMaterialsHotReload) {
                loadEditedMaterials();
            }
        });
    }

    void RtMaterialDatabase::loadEditedMaterials() {
        std::string path;
        std::filesystem::file_time_type knownTime;
        {
            std::scoped_lock lock(mutex);
            path = localPath;
            knownTime = materialsWriteTime;
        }

        if (path.empty()) {
            return;
        }

        const std::filesystem::path jsonPath = fromUtf8(path);
        std::error_code error;
        const std::filesystem::file_time_type writeTime = std::filesystem::last_write_time(jsonPath, error);
        if (error || (writeTime == knownTime)) {
            return;
        }

        std::unordered_map<uint64_t, RtMaterialEntry> onDisk;
        const bool read = readEntries(jsonPath, onDisk);
        uint32_t changed = 0;
        {
            std::scoped_lock lock(mutex);
            materialsWriteTime = writeTime;
            // Only the user's entries come from the file: the game's guesses in memory are as new as the file's.
            for (auto &pair : onDisk) {
                if (!read) {
                    break;
                }

                auto it = entries.find(pair.first);
                if (!pair.second.user) {
                    // Set by hand before and no longer (reverted to the guess in the file): the file's guess.
                    if ((it != entries.end()) && it->second.user) {
                        if (pair.second.preview.empty()) {
                            pair.second.preview = it->second.preview;
                        }

                        it->second = pair.second;
                        changed++;
                    }

                    continue;
                }

                if (it != entries.end()) {
                    const RtMaterialEntry &known = it->second;
                    // packed is 0 whenever specular is, so the class is compared on its own.
                    const bool same = known.user && (known.materialClass == pair.second.materialClass) && (known.packed == pair.second.packed) && (known.emission == pair.second.emission) &&
                        (known.emissionFromTexture == pair.second.emissionFromTexture) && (known.label == pair.second.label) &&
                        std::equal(std::begin(known.emissionColor), std::end(known.emissionColor), std::begin(pair.second.emissionColor));
                    if (same) {
                        continue;
                    }

                    if (pair.second.preview.empty()) {
                        pair.second.preview = known.preview;
                    }
                }

                entries[pair.first] = pair.second;
                changed++;
            }
        }

        if (!read) {
            // Probably broken by hand: the materials in memory stay until the next change.
            rtDiagLog("materials: %s changed but couldn't be read, kept the materials in memory", path.c_str());
        }
        else if (changed > 0) {
            rtDiagLog("materials: %u user entries changed in %s, applied", changed, path.c_str());
        }
    }

    RtMaterialDatabase &rtMaterialDatabase() {
        static RtMaterialDatabase database;
        return database;
    }

    RtMaterialDatabase::~RtMaterialDatabase() {
        if (loaded && authoring) {
            std::unordered_map<uint64_t, RtMaterialEntry> snapshot;
            std::unordered_map<uint64_t, RtEmitterEntry> emitterSnapshot;
            bool save = false;
            {
                std::scoped_lock lock(mutex);
                save = dirty;
                if (save) {
                    snapshot = entries;
                    emitterSnapshot = emitters;
                    dirty = false;
                }
            }

            if (save) {
                post([this, snapshot, emitterSnapshot]() { writeFiles(snapshot, emitterSnapshot); });
            }
        }

        {
            std::scoped_lock lock(jobMutex);
            workerStop = true;
        }

        jobChanged.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void RtMaterialDatabase::post(std::function<void()> job) {
        {
            std::scoped_lock lock(jobMutex);
            jobs.push_back(std::move(job));
            if (!worker.joinable()) {
                worker = std::thread(&RtMaterialDatabase::workerLoop, this);
            }
        }

        jobChanged.notify_one();
    }

    void RtMaterialDatabase::workerLoop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(jobMutex);
                jobChanged.wait(lock, [this]() { return workerStop || !jobs.empty(); });
                if (jobs.empty()) {
                    return;
                }

                job = std::move(jobs.front());
                jobs.pop_front();
            }

            // Nothing on this thread may take the game down: an error here only costs the job.
            try {
                job();
            }
            catch (const std::exception &e) {
                rtDiagLog("materials: background job failed: %s", e.what());
            }
            catch (...) {
                rtDiagLog("materials: background job failed with an unknown error");
            }
        }
    }

    std::string RtMaterialDatabase::path() {
        std::scoped_lock lock(mutex);
        if (localPath.empty() && loaded) {
            return utf8(executableDirectory() / "assets" / "materials" / "rt_materials.json") + " (shipped, read only)";
        }

        return localPath;
    }

    void RtMaterialDatabase::loadOnce() {
        if (!loadStarted.exchange(true)) {
            post([this]() { load(); });
        }
    }

    void RtMaterialDatabase::load() {
        const auto start = std::chrono::steady_clock::now();
        const std::filesystem::path directory = localDirectory();
        // Authoring mode: Documents\Quest64Recompiled\Ray Tracing has an rt_lights.json (or the old AppData files exist,
        // which move there). The Documents files then override the shipped ones and are saved with the HTML editors and
        // texture previews. Otherwise the shipped files in assets\materials are the whole configuration, read only, and
        // nothing is written. The folder alone doesn't count: 1.2.0 saved rt_materials.json and rt_emitters.json there
        // for every player, and those stale copies would override the shipped configuration. Only authors have an
        // rt_lights.json (Eltale Studio writes it; the game never does).
        std::error_code error;
        const std::filesystem::path legacy = legacyDirectory();
        const bool folderExists = std::filesystem::is_directory(directory, error);
        const bool authorLights = std::filesystem::exists(directory / "rt_lights.json", error);
        const bool legacyFiles = std::filesystem::exists(legacy / "rt_materials.json", error) || std::filesystem::exists(legacy / "rt_emitters.json", error);
        const bool authoringMode = authorLights || legacyFiles;
        authoring = authoringMode;
        if (authoringMode) {
            rtDiagLog("ray tracing config: AUTHORING mode, because %s exists: the Documents files override the shipped ones",
                authorLights ? utf8(directory / "rt_lights.json").c_str() : utf8(legacy).c_str());
        }
        else {
            rtDiagLog("ray tracing config: PLAYER mode, reading only the shipped files in %s%s", utf8(executableDirectory() / "assets" / "materials").c_str(),
                folderExists ? " (a Documents Ray Tracing folder without rt_lights.json, probably left by version 1.2.0, is ignored)" : "");
        }

        if (authoringMode) {
            migrateLegacyFiles(directory);
        }

        const std::filesystem::path local = directory / "rt_materials.json";
        const std::filesystem::path shipped = executableDirectory() / "assets" / "materials" / "rt_materials.json";
        std::unordered_map<uint64_t, RtMaterialEntry> localEntries;
        const bool localRead = authoringMode && readEntries(local, localEntries);

        // The shipped file seeds the local one. Its user entries replace local guesses, so fixes made in the repo
        // reach players; local user entries always win.
        std::unordered_map<uint64_t, RtMaterialEntry> seeded;
        uint32_t seededCount = 0;
        std::string shippedOrigin;
        const bool shippedRead = readShippedEntries(shipped, seeded, shippedOrigin);
        if (shippedRead) {
            for (auto &pair : seeded) {
                auto it = localEntries.find(pair.first);
                if ((it == localEntries.end()) || (pair.second.user && !it->second.user)) {
                    localEntries[pair.first] = pair.second;
                    seededCount++;
                }
            }
        }

        // Emitters: the local file's listed entries win; the shipped list adds entries and replaces local candidates.
        const std::filesystem::path localEmitters = directory / "rt_emitters.json";
        const std::filesystem::path shippedEmitters = executableDirectory() / "assets" / "materials" / "rt_emitters.json";
        std::unordered_map<uint64_t, RtEmitterEntry> emitterEntries;
        int localLegacyGlow = -1;
        int shippedLegacyGlow = -1;
        const bool localEmittersRead = authoringMode && readEmitters(localEmitters, emitterEntries, &localLegacyGlow);
        std::unordered_map<uint64_t, RtEmitterEntry> seededEmitters;
        uint32_t seededEmitterCount = 0;
        std::string shippedEmittersOrigin;
        const bool shippedEmittersRead = readShippedEmitters(shippedEmitters, seededEmitters, &shippedLegacyGlow, shippedEmittersOrigin);
        if (shippedEmittersRead) {
            for (auto &pair : seededEmitters) {
                auto it = emitterEntries.find(pair.first);
                if ((it == emitterEntries.end()) || (pair.second.listed && (it->second.source != "user"))) {
                    if (it != emitterEntries.end()) {
                        // Keep what the game learned about it.
                        pair.second.area = it->second.area;
                        pair.second.map = it->second.map;
                        pair.second.room = it->second.room;
                        pair.second.size = it->second.size;
                        pair.second.preview = it->second.preview;
                        pair.second.observed = it->second.observed;
                        pair.second.hasObserved = it->second.hasObserved;
                    }

                    emitterEntries[pair.first] = pair.second;
                    seededEmitterCount++;
                }
            }
        }

        size_t count = 0;
        size_t emitterCount = 0;
        size_t userCount = 0;
        size_t userEmitterCount = 0;
        for (const auto &pair : localEntries) {
            userCount += pair.second.user ? 1 : 0;
        }

        for (const auto &pair : emitterEntries) {
            userEmitterCount += (pair.second.source == "user") ? 1 : 0;
        }

        {
            std::scoped_lock lock(mutex);
            // Without authoring there's no local file: the paths stay empty, so nothing is saved or previewed.
            localPath = authoringMode ? utf8(local) : std::string();
            if (localRead) {
                materialsWriteTime = std::filesystem::last_write_time(local, error);
            }

            previewDirectory = authoringMode ? utf8(directory / "rt_materials") : std::string();
            for (auto &pair : localEntries) {
                entries.emplace(pair.first, pair.second);
            }

            for (auto &pair : emitterEntries) {
                emitters.emplace(pair.first, pair.second);
            }

            legacyGlow = (localLegacyGlow >= 0) ? (localLegacyGlow != 0) : ((shippedLegacyGlow >= 0) ? (shippedLegacyGlow != 0) : true);
            listedEmitterIds.clear();
            for (const auto &pair : emitters) {
                if (pair.second.listed) {
                    listedEmitterIds.push_back(pair.first);
                }

                const uint64_t texture = pair.second.hasObserved ? pair.second.observed.texture : (pair.second.params.match.texture ? pair.second.params.match.value.texture : 0);
                if (texture != 0) {
                    emitterTextureKeys.insert(texture);
                }
            }

            // User entries are tried first, so they win over shipped ones that match the same draws.
            std::stable_sort(listedEmitterIds.begin(), listedEmitterIds.end(), [this](uint64_t a, uint64_t b) {
                return (emitters[a].source == "user") && (emitters[b].source != "user");
            });

            dirty = dirty || !localRead || (seededCount > 0) || !localEmittersRead || (seededEmitterCount > 0);
            count = entries.size();
            emitterCount = emitters.size();
        }

        loadManualLights(false);
        loaded = true;
        const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (authoringMode) {
            rtDiagLog("materials: authoring mode: the files in %s override the shipped ones and are saved, with the HTML editors and previews",
                utf8(directory).c_str());
            rtDiagLog("materials: %zu entries, %zu user entries from %s (%s), %u taken from %s (%s); emitters: %zu entries, %zu user entries from %s (%s), %u taken from %s (%s); loaded in %.1f ms on a background thread",
                count, userCount, utf8(local).c_str(), localRead ? "read" : "not found, created on first save", seededCount, shippedOrigin.c_str(), shippedRead ? "read" : "SHIPPED MATERIALS MISSING",
                emitterCount, userEmitterCount, utf8(localEmitters).c_str(), localEmittersRead ? "read" : "not found, created on first save", seededEmitterCount,
                shippedEmittersOrigin.c_str(), shippedEmittersRead ? "read" : "SHIPPED EMITTERS MISSING", milliseconds);
        }
        else {
            rtDiagLog("materials: shipped configuration only, nothing is written: %zu material entries (%zu user) from %s (%s); %zu emitter entries (%zu user) from %s (%s); loaded in %.1f ms on a background thread",
                count, userCount, shippedOrigin.c_str(), shippedRead ? "read" : "SHIPPED MATERIALS MISSING", emitterCount, userEmitterCount, shippedEmittersOrigin.c_str(),
                shippedEmittersRead ? "read" : "SHIPPED EMITTERS MISSING", milliseconds);
        }
    }

    // A self-contained editor page for rt_emitters.json or rt_materials.json: the file's contents are embedded, every
    // row can be edited, and Save writes the real file through the File System Access API (Edge and Chrome), with
    // Download JSON as the fallback.
    static const char *kEditorTemplate = R"EDITOR(<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%TITLE%</title><style>
body{font-family:Segoe UI,sans-serif;background:#1e1e1e;color:#ddd;margin:0}
#bar{position:sticky;top:0;background:#252526;padding:10px 16px;border-bottom:1px solid #444;z-index:2}
#bar h1{margin:0 0 4px 0;font-size:20px}#bar p{margin:2px 0 8px 0;font-size:13px;color:#aaa}
#list{padding:0 16px 40px 16px}h2{margin-top:24px;border-bottom:1px solid #555;font-size:16px}
table{border-collapse:collapse;width:100%}td{padding:5px 8px;border-bottom:1px solid #333;vertical-align:top;font-size:13px}
tr.changed{background:#2d3a2d}img,.swatch{width:64px;height:64px;image-rendering:pixelated;object-fit:contain;background:#444;border:1px solid #555}
code{color:#9cdcfe}.badge{display:inline-block;background:#6a4a00;color:#ffd77a;border-radius:3px;padding:0 5px;margin:2px 4px 0 0;font-size:11px}.small{font-size:11px;color:#999;margin-top:3px}input.label{width:220px}input.areas{width:110px}input.num{width:60px}
.chk{white-space:nowrap;margin-right:8px;display:inline-block}.match .chk{display:block}.val{display:inline-block;width:36px;text-align:right}
button{background:#3a3d41;color:#ddd;border:1px solid #555;padding:4px 10px;cursor:pointer;border-radius:3px}button.primary{background:#0e639c;border-color:#1177bb;color:#fff}
#status{margin-left:12px;color:#f0c040}input,select{background:#2d2d2d;color:#ddd;border:1px solid #555}
</style></head><body>
<div id="bar"><h1>%TITLE%</h1>
<p>Edit any row, then press Save. Material changes show in the running game within a second; emitter changes take effect the next time the game starts.
Save writes the file directly in Edge or Chrome: the first time, open <code>%FOLDER%\%FILE%</code> (the folder this page is in, not Documents itself); later saves reuse it. Save refuses any other file.
In other browsers use Download JSON and replace that file. Edited rows are marked "source": "user" and the game never overwrites them.</p>
<button class="primary" id="save">Save</button> <button id="download">Download JSON</button> <button id="load">Load JSON</button>
<input type="file" id="file" accept=".json" hidden>
&nbsp; <input id="search" placeholder="Search key, label, colour..." size="28"> <select id="area"><option value="">All areas</option></select>
<label class="chk"><input type="checkbox" id="changed"> Show only changed</label>
<span id="emitteronly"><label class="chk"><input type="checkbox" id="wouldglow"> Show only "old heuristic: would glow"</label>
<label class="chk" title="Unlisted untextured polygon effects (item pillar, hit sparks) glow as they did before the list. List one with enabled off to keep it dark."><input type="checkbox" id="legacy"> Legacy glow for unlisted untextured effects</label></span>
<span id="status"></span>
<div id="stale" hidden style="margin-top:6px;background:#5a3a00;padding:6px;border-radius:3px"><span id="stalemsg"></span> <button id="loadfresh">Load it</button></div></div>
<div id="list"></div>
<script id="data" type="application/json">%DATA%</script>
<script>
const MODE='%MODE%', FILE='%FILE%', FOLDER=%FOLDERJSON%;
const LEGACY='legacy glow for untextured effects';
let legacyChanged=false;
const PICKER_TYPES=[{description:'JSON',accept:{'application/json':['.json']}}];
function pickerError(e,what){if(e.name==='AbortError')return;const blocked=(e.name==='SecurityError'||e.name==='NotAllowedError'||/system|blocked|not allowed/i.test(e.message||''));status(blocked?what+' blocked by the browser for that folder. Pick '+FOLDER+'\\'+FILE+', or use Download JSON and copy the file there.':what+' failed: '+e.message+' (try Download JSON)');}
const doc=JSON.parse(document.getElementById('data').textContent);
const changed=new Set();
let handle=null;
const $=id=>document.getElementById(id);
function status(t){$('status').textContent=t;}
function el(tag,attrs,...kids){const e=document.createElement(tag);for(const k in (attrs||{})){const v=attrs[k];if(k==='on'){for(const ev in v)e.addEventListener(ev,v[ev]);}else if(k==='style'||k==='title'||k==='placeholder'){e.setAttribute(k,v);}else{e[k]=v;}}for(const c of kids){if(c!=null)e.append(c.nodeType?c:document.createTextNode(String(c)));}return e;}
function idOf(e){return MODE==='emitters'?e.id:e.key;}
function touch(e){e.source='user';changed.add(idOf(e));status(changed.size+' unsaved change(s)');}
function rgbToHex(c){return '#'+c.slice(0,3).map(v=>Math.round(Math.max(0,Math.min(1,v||0))*255).toString(16).padStart(2,'0')).join('');}
function hexToRgb(h){return [1,3,5].map(i=>Math.round(parseInt(h.substr(i,2),16)/255*1000)/1000);}
function fmt(v){if(v===undefined||v===null)return '-';if(Array.isArray(v))return v.map(x=>typeof x==='number'?Math.round(x*100)/100:x).join(' ');return String(v);}
function hex2(n){return n>=0?n.toString(16).toUpperCase().padStart(2,'0'):'-1';}
function num(e,f,step,min,max){return el('input',{type:'number',step:step,min:min,max:max,value:e[f]??0,className:'num',on:{change:ev=>{e[f]=parseFloat(ev.target.value)||0;touch(e);}}});}
function slider(e,f){const out=el('span',{className:'val'},(e[f]??0).toFixed(2));const s=el('input',{type:'range',min:0,max:1,step:0.01,value:e[f]??0,on:{input:ev=>{e[f]=parseFloat(ev.target.value);out.textContent=e[f].toFixed(2);touch(e);}}});return el('span',{},s,out);}
function swatch(e){const o=e.observed||{};let c=o['vertex color'];const p=o['prim color'];if(p&&p.slice(0,3).some(v=>v>0.02))c=p;c=c||[0.5,0.5,0.5];return el('div',{className:'swatch',style:'background:'+rgbToHex(c),title:'untextured draw: its primitive or vertex colour'});}
function thumb(e){if(e.preview){const i=el('img',{src:e.preview});i.onerror=()=>i.replaceWith(swatch(e));return i;}return swatch(e);}
function move(e,from,to){doc[from]=(doc[from]||[]).filter(x=>x!==e);(doc[to]=doc[to]||[]).push(e);}
const KINDS=['additive','blended','cutout','untextured','opaque'];
const MATCH=['signature','texture','vertex address','combiner','other mode','prim color','env color','vertex color','triangles'];
function emitterRow(e,listed){
 e.match=e.match||{};const o=e.observed||{};
 const isSample=!Array.isArray(e.color);
 const picker=el('input',{type:'color',value:rgbToHex(Array.isArray(e.color)?e.color:[1,1,1]),disabled:isSample,on:{input:ev=>{e.color=hexToRgb(ev.target.value);touch(e);}}});
 const sample=el('input',{type:'checkbox',checked:isSample,on:{change:ev=>{if(ev.target.checked){e.color='sample';picker.disabled=true;}else{e.color=hexToRgb(picker.value);picker.disabled=false;}touch(e);}}});
 const kinds=el('div',{},...KINDS.map(k=>el('label',{className:'chk'},el('input',{type:'checkbox',checked:(e['only kinds']||[]).includes(k),on:{change:ev=>{let a=(e['only kinds']||[]).filter(x=>x!==k);if(ev.target.checked)a.push(k);e['only kinds']=a;touch(e);}}}),k)));
 const areas=el('input',{type:'text',className:'areas',placeholder:'04 or 04:2',value:(e['only areas']||[]).map(a=>hex2(a.map)+(a.room>=0?':'+a.room:'')).join(' '),on:{change:ev=>{e['only areas']=ev.target.value.split(/[\s,;]+/).filter(Boolean).map(t=>{const [m,r]=t.split(':');return {map:parseInt(m,16),room:r===undefined?-1:parseInt(r,10)};});touch(e);}}});
 const match=el('div',{className:'match'},el('div',{className:'small'},'Match on (all ticked must match):'),...MATCH.map(f=>{const has=f in e.match;let v=has?e.match[f]:o[f];if(f==='triangles'&&v!==undefined&&!Array.isArray(v))v=[v,v];const cb=el('input',{type:'checkbox',checked:has,disabled:v===undefined,on:{change:ev=>{if(ev.target.checked){e.match[f]=v;if(f.endsWith('color')&&e.match['color tolerance']===undefined)e.match['color tolerance']=0.1;}else{delete e.match[f];}touch(e);}}});return el('label',{className:'chk'},cb,f+': ',el('code',{},fmt(v)));}));
 const enabled=el('input',{type:'checkbox',checked:!!e.enabled,on:{change:ev=>{e.enabled=ev.target.checked;touch(e);}}});
 const label=el('input',{type:'text',className:'label',value:e.label||'',on:{change:ev=>{e.label=ev.target.value;touch(e);}}});
 const list=(on)=>{if(!Object.keys(e.match).length&&o.signature)e.match={signature:o.signature};e.enabled=on;move(e,'candidates','emitters');touch(e);render();};
 const action=listed?el('button',{on:{click:()=>{e.enabled=false;move(e,'emitters','candidates');touch(e);render();}}},'Move to candidates')
  :el('span',{},el('button',{className:'primary',on:{click:()=>list(true)}},'Enable as emitter'),' ',el('button',{title:'List it switched off, so it stays dark (also stops the legacy glow)',on:{click:()=>list(false)}},'Keep dark'));
 const untextured=(o.texture==='none')||(e.kind==='untextured');
 const facts=el('div',{className:'small'},(e.kind||o.kind||'?')+(o.triangles!==undefined?' · '+o.triangles+' triangles':'')+(o['vertex address']?' · vertex address '+o['vertex address']:'')+' · '+(e.size||'')+' · '+(e.source||''));
 const badges=el('div',{},e['old heuristic']?el('span',{className:'badge'},'old heuristic: would glow'):null,(!listed&&untextured&&doc[LEGACY])?el('span',{className:'badge'},'lit now by legacy glow'):null);
 return el('tr',{className:changed.has(e.id)?'changed':''},
  el('td',{},thumb(e)),
  el('td',{},el('code',{},e.id),facts,badges),
  el('td',{},label,el('div',{className:'small'},'first seen: '+(e['first seen']||'?'))),
  el('td',{},el('label',{className:'chk'},enabled,'enabled'),el('br'),action),
  el('td',{},el('label',{className:'chk'},sample,'sample colour'),el('br'),picker),
  el('td',{},'intensity ',num(e,'intensity',0.05,0,10),el('br'),'range scale ',num(e,'range scale',0.05,0,10),el('br'),'max lights ',num(e,'max lights',1,0,64),el('br'),'fade in ',num(e,'fade in frames',1,0,600),' out ',num(e,'fade out frames',1,0,600)),
  el('td',{},'only kinds',kinds,'only areas ',areas),
  el('td',{},match));
}
const CLASSES=['matte','sprite','grass','dirt','wood','cloth','skin','stone','polished','metal','water','lava','foliage','window'];
function materialRow(e){
 const cls=el('select',{on:{change:ev=>{e.class=ev.target.value;if(e.class==='lava'&&!(e.emission>0)){e.roughness=0.7;e.metalness=0;e.specular=0.2;e.emission=1.5;e['emission color']='albedo';}touch(e);render();}}},...CLASSES.map(c=>el('option',{value:c,selected:e.class===c},c)));
 const fromTexture=!Array.isArray(e['emission color']);
 const emitPicker=el('input',{type:'color',value:rgbToHex(Array.isArray(e['emission color'])?e['emission color']:[1,0.4,0.1]),disabled:fromTexture,on:{input:ev=>{e['emission color']=hexToRgb(ev.target.value);touch(e);}}});
 const emitTexture=el('input',{type:'checkbox',checked:fromTexture,on:{change:ev=>{if(ev.target.checked){e['emission color']='albedo';emitPicker.disabled=true;}else{e['emission color']=hexToRgb(emitPicker.value);emitPicker.disabled=false;}touch(e);}}});
 const emitOut=el('span',{className:'val'},(e.emission??0).toFixed(2));
 const emitSlider=el('input',{type:'range',min:0,max:4,step:0.05,value:e.emission??0,title:'Light the surface gives off: 0 none, 1 its full texture colour, more glows brighter and lights its surroundings',on:{input:ev=>{e.emission=parseFloat(ev.target.value);emitOut.textContent=e.emission.toFixed(2);touch(e);}}});
 const label=el('input',{type:'text',className:'label',value:e.label||'',on:{change:ev=>{e.label=ev.target.value;touch(e);}}});
 return el('tr',{className:changed.has(e.key)?'changed':''},
  el('td',{},thumb(e)),
  el('td',{},el('code',{},e.key),el('div',{className:'small'},e.source||'')),
  el('td',{},label,el('div',{className:'small'},(e.on||'')+' · '+(e.texture||''))),
  el('td',{},'class ',cls),
  el('td',{},'roughness ',slider(e,'roughness'),el('br'),'metalness ',slider(e,'metalness'),el('br'),'specular ',slider(e,'specular')),
  el('td',{},'emission ',el('span',{},emitSlider,emitOut),el('br'),el('label',{className:'chk'},emitTexture,'texture colour'),' ',emitPicker),
  el('td',{className:'small'},e.stats||''));
}
function entries(){return MODE==='emitters'?[...(doc.emitters||[]).map(e=>[e,true]),...(doc.candidates||[]).map(e=>[e,false])]:(doc.materials||[]).map(e=>[e,true]);}
function render(){
 const q=$('search').value.toLowerCase(),area=$('area').value,only=$('changed').checked,glowOnly=MODE==='emitters'&&$('wouldglow').checked,list=$('list');
 list.textContent='';const groups=new Map();
 for(const [e,listed] of entries()){
  const a=e['first seen']||'unknown area';
  if(area&&a!==area)continue;if(only&&!changed.has(idOf(e))&&e.source!=='user')continue;if(glowOnly&&!e['old heuristic'])continue;if(q&&!JSON.stringify(e).toLowerCase().includes(q))continue;
  const g=(MODE==='emitters'?(listed?'Listed emitters':'Candidates (no light until enabled)')+' · ':'')+a;
  if(!groups.has(g))groups.set(g,[]);groups.get(g).push([e,listed]);
 }
 for(const [g,rows] of groups){list.append(el('h2',{},g+' ('+rows.length+')'));const t=el('table');for(const [e,listed] of rows)t.append(MODE==='emitters'?emitterRow(e,listed):materialRow(e));list.append(t);}
 if(!groups.size)list.append(el('p',{},'Nothing matches.'));
}
function fillAreas(){const s=$('area'),cur=s.value;s.length=1;[...new Set(entries().map(([e])=>e['first seen']||'unknown area'))].sort().forEach(a=>s.append(el('option',{value:a},a)));s.value=cur;}
function idb(){return new Promise((res,rej)=>{const r=indexedDB.open('rt-editor',1);r.onupgradeneeded=()=>r.result.createObjectStore('handles');r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error);});}
async function idbGet(){try{const db=await idb();return await new Promise(res=>{const t=db.transaction('handles').objectStore('handles').get(FILE);t.onsuccess=()=>res(t.result||null);t.onerror=()=>res(null);});}catch(e){return null;}}
async function idbPut(h){try{const db=await idb();db.transaction('handles','readwrite').objectStore('handles').put(h,FILE);}catch(e){}}
async function writable(h){try{if((await h.queryPermission({mode:'readwrite'}))==='granted')return true;return (await h.requestPermission({mode:'readwrite'}))==='granted';}catch(e){return false;}}
const EXPECTED=(FOLDER+'\\'+FILE).toLowerCase();
const SECTION=MODE==='emitters'?'emitters':'materials';
// Whether a picked file is the one the game reads: the game stamps its path in "_location" every time it writes it.
async function checkFile(h){
 let text='';try{text=await (await h.getFile()).text();}catch(e){return {ok:false,why:'it could not be read'};}
 let d=null;try{d=JSON.parse(text);}catch(e){return {ok:false,why:text.length?'it is not valid JSON':'it is empty (a new file)'};}
 if(!d||typeof d!=='object'||!(SECTION in d))return {ok:false,why:'it is not a '+FILE+' written by the game'};
 if(!d._location)return {ok:false,why:'it has no location stamp from the game (an old or copied file)'};
 if(String(d._location).toLowerCase()!==EXPECTED)return {ok:false,why:'the game wrote it for '+d._location};
 return {ok:true,doc:d};
}
async function getHandle(){
 for(const h of [handle,await idbGet()]){if(h&&await writable(h)){const c=await checkFile(h);if(c.ok){handle=h;return {h,c};}}}
 // An Open dialog: the existing file has to be picked, so a new file can't be made by mistake.
 status('Pick '+FOLDER+'\\'+FILE);
 const [h]=await window.showOpenFilePicker({startIn:'documents',types:PICKER_TYPES,multiple:false});
 if(!(await writable(h)))throw new Error('the browser did not allow writing to it');
 const c=await checkFile(h);
 if(!c.ok)throw Object.assign(new Error('that is not the game\'s file: '+c.why+'. Pick '+FOLDER+'\\'+FILE+' instead.'),{name:'WrongFile'});
 handle=h;idbPut(h);return {h,c};
}
// Keeps what the game added to the file since this page was made; rows marked user here replace the file's.
function mergeForSave(current){
 if(!current||typeof current!=='object')return doc;
 if(MODE==='emitters'){
  current.emitters=current.emitters||[];current.candidates=current.candidates||[];
  if(legacyChanged)current[LEGACY]=!!doc[LEGACY];
  for(const [e,listed] of entries()){if(e.source!=='user')continue;
   current.emitters=current.emitters.filter(x=>(x.id||x.key)!==e.id);current.candidates=current.candidates.filter(x=>(x.id||x.key)!==e.id);
   (listed?current.emitters:current.candidates).push(e);}
 }else{
  if(!Array.isArray(current.materials))current.materials=[];
  for(const [e] of entries()){if(e.source!=='user')continue;current.materials=current.materials.filter(x=>x.key!==e.key);current.materials.push(e);}
 }
 return current;
}
async function save(){
 if(!window.showOpenFilePicker){download();return;}
 try{
  const {h,c}=await getHandle();const current=c.doc;
  const newer=(current._saved&&doc._saved&&current._saved>doc._saved);
  const count=changed.size+(legacyChanged?1:0);
  const out=mergeForSave(current);const w=await h.createWritable();await w.write(JSON.stringify(out,null,2)+'\n');await w.close();
  changed.clear();legacyChanged=false;
  status('Saved '+count+' change(s) to '+current._location+' at '+new Date().toLocaleTimeString()+(newer?' (the file had newer changes from the game, kept)':'')+'. The game uses it the next time it starts.');
  render();
 }catch(e){if(e.name==='WrongFile'){status('Not saved: '+e.message);}else{handle=null;pickerError(e,'Saving');}}
}
function download(){const b=new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'});const a=el('a',{href:URL.createObjectURL(b),download:FILE});document.body.append(a);a.click();a.remove();status('Downloaded '+FILE+': replace '+FOLDER+'\\'+FILE+' with it.');}
function replaceDoc(d){for(const k of Object.keys(doc))delete doc[k];Object.assign(doc,d);changed.clear();fillAreas();render();}
async function load(){
 if(window.showOpenFilePicker){try{const [h]=await window.showOpenFilePicker({startIn:'documents',types:PICKER_TYPES});const c=await checkFile(h);
  if(c.ok){replaceDoc(c.doc);handle=h;idbPut(h);$('stale').hidden=true;status('Loaded '+c.doc._location);}
  else{const d=JSON.parse(await (await h.getFile()).text());replaceDoc(d);status('Loaded '+h.name+', but '+c.why+'; Save will still write to '+FOLDER+'\\'+FILE);}
 }catch(e){pickerError(e,'Loading');}}
 else $('file').click();
}
$('file').addEventListener('change',async ev=>{const f=ev.target.files[0];if(f){replaceDoc(JSON.parse(await f.text()));status('Loaded '+f.name);}});
$('save').onclick=save;$('download').onclick=download;$('load').onclick=load;
$('search').oninput=render;$('area').onchange=render;$('changed').onchange=render;
if(MODE==='emitters'){
 if(doc[LEGACY]===undefined)doc[LEGACY]=true;
 $('legacy').checked=!!doc[LEGACY];
 $('legacy').onchange=ev=>{doc[LEGACY]=ev.target.checked;legacyChanged=true;status('unsaved change: legacy glow '+(doc[LEGACY]?'on':'off'));render();};
 $('wouldglow').onchange=render;
}else{$('emitteronly').style.display='none';}
window.addEventListener('beforeunload',ev=>{if(changed.size){ev.preventDefault();ev.returnValue='';}});
fillAreas();render();
// When the page already has access to the game's file, warn if the file is newer than this page's copy.
(async()=>{try{const h=await idbGet();if(!h||((await h.queryPermission({mode:'read'}))!=='granted'))return;const c=await checkFile(h);
 if(c.ok&&c.doc._saved&&(!doc._saved||c.doc._saved>doc._saved)){const s=$('stale');s.hidden=false;
  $('stalemsg').textContent='The file on disk ('+c.doc._location+') is newer than this page. Load it before editing, or Save will merge your edits into it.';
  $('loadfresh').onclick=()=>{replaceDoc(c.doc);handle=h;s.hidden=true;status('Loaded '+c.doc._location);};}}catch(e){}})();
</script></body></html>
)EDITOR";

    static std::string editorHtml(const char *title, const char *mode, const char *fileName, const std::string &folder, const std::string &jsonText) {
        // The JSON sits in a script element; "</" would end it early.
        std::string data;
        data.reserve(jsonText.size());
        for (size_t i = 0; i < jsonText.size(); i++) {
            if ((jsonText[i] == '<') && (i + 1 < jsonText.size()) && (jsonText[i + 1] == '/')) {
                data += "<\\/";
                i++;
            }
            else {
                data += jsonText[i];
            }
        }

        std::string html = kEditorTemplate;
        auto replaceAll = [&html](const std::string &from, const std::string &to) {
            size_t position = 0;
            while ((position = html.find(from, position)) != std::string::npos) {
                html.replace(position, from.size(), to);
                position += to.size();
            }
        };

        replaceAll("%TITLE%", title);
        replaceAll("%MODE%", mode);
        replaceAll("%FILE%", fileName);
        replaceAll("%FOLDERJSON%", json(folder).dump());
        replaceAll("%FOLDER%", htmlEscape(folder));
        const size_t dataPosition = html.find("%DATA%");
        if (dataPosition != std::string::npos) {
            html.replace(dataPosition, 6, data);
        }

        return html;
    }

    void RtMaterialDatabase::writeFiles(const std::unordered_map<uint64_t, RtMaterialEntry> &snapshotIn, const std::unordered_map<uint64_t, RtEmitterEntry> &emitterSnapshotIn) {
        std::string path;
        {
            std::scoped_lock lock(mutex);
            path = localPath;
        }

        if (path.empty()) {
            return;
        }

        const std::filesystem::path jsonPath = fromUtf8(path);
        const std::filesystem::path emittersPath = jsonPath.parent_path() / "rt_emitters.json";

        // The user may have saved edits from the HTML editors or Eltale Studio while the game ran: entries the files
        // mark as the user's replace the game's before anything is written. They're applied in memory first, as the
        // write below hides the change from the next poll.
        // The legacy glow switch is only ever changed by the user, so the file's value is kept.
        if (kMaterialsHotReload) {
            loadEditedMaterials();
        }

        int diskLegacyGlow = -1;
        std::unordered_map<uint64_t, RtMaterialEntry> snapshot = snapshotIn;
        std::unordered_map<uint64_t, RtEmitterEntry> emitterSnapshot = emitterSnapshotIn;
        {
            std::unordered_map<uint64_t, RtMaterialEntry> onDisk;
            if (readEntries(jsonPath, onDisk)) {
                for (auto &pair : onDisk) {
                    if (pair.second.user) {
                        snapshot[pair.first] = pair.second;
                    }
                }
            }

            std::unordered_map<uint64_t, RtEmitterEntry> emittersOnDisk;
            if (readEmitters(emittersPath, emittersOnDisk, &diskLegacyGlow)) {
                for (auto &pair : emittersOnDisk) {
                    if (pair.second.source == "user") {
                        RtEmitterEntry entry = pair.second;
                        auto known = emitterSnapshot.find(pair.first);
                        if ((known != emitterSnapshot.end()) && !entry.hasObserved && known->second.hasObserved) {
                            entry.observed = known->second.observed;
                            entry.hasObserved = true;
                        }

                        emitterSnapshot[pair.first] = entry;
                    }
                }
            }
        }

        // Grouped by where each texture was first seen, then by class, so related textures sit together.
        std::vector<std::pair<uint64_t, const RtMaterialEntry *>> sorted;
        sorted.reserve(snapshot.size());
        for (const auto &pair : snapshot) {
            sorted.emplace_back(pair.first, &pair.second);
        }

        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
            const RtMaterialEntry &x = *a.second;
            const RtMaterialEntry &y = *b.second;
            if (x.map != y.map) return x.map < y.map;
            if (x.room != y.room) return x.room < y.room;
            if (x.materialClass != y.materialClass) return x.materialClass < y.materialClass;
            if (x.surface != y.surface) return x.surface < y.surface;
            return a.first < b.first;
        });

        ordered_json materials = ordered_json::array();
        for (const auto &pair : sorted) {
            const RtMaterialEntry &entry = *pair.second;
            ordered_json value = ordered_json::object();
            value["key"] = keyString(pair.first);
            value["label"] = entry.label;
            value["class"] = rtMaterialClassName(entry.materialClass);
            value["roughness"] = std::round(entry.roughness * 100.0f) / 100.0f;
            value["metalness"] = std::round(entry.metalness * 100.0f) / 100.0f;
            value["specular"] = std::round(entry.specular * 100.0f) / 100.0f;
            if ((entry.emission > 0.0f) || (kClassInfo[size_t(entry.materialClass)].emission > 0.0f)) {
                value["emission"] = std::round(entry.emission * 100.0f) / 100.0f;
                if (entry.emissionFromTexture) {
                    value["emission color"] = "albedo";
                }
                else {
                    value["emission color"] = { std::round(entry.emissionColor[0] * 1000.0f) / 1000.0f, std::round(entry.emissionColor[1] * 1000.0f) / 1000.0f,
                        std::round(entry.emissionColor[2] * 1000.0f) / 1000.0f };
                }
            }

            value["source"] = entry.user ? "user" : "guess";
            value["on"] = entry.surface;
            value["first seen"] = entry.area;
            value["map"] = entry.map;
            value["room"] = entry.room;
            value["texture"] = entry.texture;
            value["preview"] = entry.preview;
            value["stats"] = entry.stats;
            materials.push_back(value);
        }

        ordered_json document = ordered_json::object();
        document["_readme"] = {
            "Ray traced surface materials, one entry per texture, grouped by where each was first seen.",
            "Easiest: click the surface in Eltale Studio (Rooms or Characters, Alt+click), or open rt_materials.html next to",
            "this file in Edge or Chrome, edit there and press Save.",
            "class: matte, sprite, grass, dirt, wood, cloth, skin, stone, polished, metal, water, lava, foliage or window.",
            "roughness 0 (mirror) to 1 (matte); metalness 0 (plastic, stone) to 1 (metal); specular 0 (none) to 1 (full).",
            "Set source to \"user\" on an entry you edit, or the game may replace it with a new guess.",
            "The game reads the file when it starts, and its user entries again within a second of it changing. Only key,",
            "label, class, roughness, metalness, specular, emission and source matter; the other fields are notes the game fills in."
        };
        // Where the game reads the file and when it last wrote it: the editor page checks that it saves into this
        // file, and warns when the file on disk is newer than the page.
        const long long savedAt = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        document["_location"] = utf8(jsonPath);
        document["_saved"] = savedAt;
        document["materials"] = materials;

        std::error_code error;
        std::filesystem::create_directories(jsonPath.parent_path(), error);
        auto writeAtomically = [&](const std::filesystem::path &target, const std::string &contents) {
            std::filesystem::path temporary = target;
            temporary += ".tmp";
            {
                std::ofstream stream(temporary, std::ios::trunc | std::ios::binary);
                if (!stream.is_open()) {
                    return;
                }

                stream << contents;
            }

            std::filesystem::rename(temporary, target, error);
            if (error) {
                std::filesystem::remove(target, error);
                std::filesystem::rename(temporary, target, error);
            }
        };

        const std::string materialsText = document.dump(2) + "\n";
        writeAtomically(jsonPath, materialsText);
        {
            std::error_code timeError;
            const std::filesystem::file_time_type writeTime = std::filesystem::last_write_time(jsonPath, timeError);
            std::scoped_lock lock(mutex);
            if (!timeError) {
                materialsWriteTime = writeTime;
            }
        }
        writeAtomically(jsonPath.parent_path() / "rt_materials.html", editorHtml("Ray traced materials", "materials", "rt_materials.json", utf8(jsonPath.parent_path()), materialsText));

        // Emitters: the listed ones first, then the candidates the game found, each grouped by where first seen.
        std::vector<std::pair<uint64_t, const RtEmitterEntry *>> sortedEmitters;
        for (const auto &pair : emitterSnapshot) {
            sortedEmitters.emplace_back(pair.first, &pair.second);
        }

        std::sort(sortedEmitters.begin(), sortedEmitters.end(), [](const auto &a, const auto &b) {
            const RtEmitterEntry &x = *a.second;
            const RtEmitterEntry &y = *b.second;
            if (x.listed != y.listed) return x.listed;
            if (x.map != y.map) return x.map < y.map;
            if (x.room != y.room) return x.room < y.room;
            if (x.kind != y.kind) return x.kind < y.kind;
            return a.first < b.first;
        });

        ordered_json listedJson = ordered_json::array();
        ordered_json candidatesJson = ordered_json::array();
        for (const auto &pair : sortedEmitters) {
            (pair.second->listed ? listedJson : candidatesJson).push_back(emitterJson(pair.first, *pair.second));
        }

        ordered_json emitterDocument = ordered_json::object();
        emitterDocument["_readme"] = {
            "Draws that emit light in the ray traced lighting. Only entries under \"emitters\" with enabled true give light.",
            "Easiest: open rt_emitters.html next to this file in Edge or Chrome, press Enable as emitter on a candidate, then Save.",
            "\"match\" lists what a draw must have to use the entry; every field present must match: signature (one id for the",
            "draw's texture, vertex address, combiner, render mode and triangle count), texture (\"none\" for untextured),",
            "vertex address, combiner, other mode, prim color / env color / vertex color (within color tolerance) and",
            "triangles [min, max]. \"observed\" is what the game saw, to copy fields from. The old {\"key\": texture} form still works.",
            "color: \"sample\" or [r, g, b]. intensity about 1 for a spell, 0.1 for a faint glow. range scale multiplies the reach.",
            "max lights limits lights per entry (0 = no limit). fade in / fade out frames: 0 for the defaults.",
            "only kinds: additive, blended, cutout, untextured, opaque. only areas: [{\"map\": 4, \"room\": -1}] (room -1 = every room).",
            "The game reads this file when it starts; entries with source \"user\" always win.",
            "\"legacy glow for untextured effects\": when true, unlisted untextured polygon effects (item pillar, hit sparks)",
            "glow as they did before this list, as bright as the GPU finds them. A listed entry with enabled false keeps one dark.",
            "Candidates tagged \"old heuristic\": \"would glow\" are ones the old automatic detection made glow."
        };
        emitterDocument["_location"] = utf8(emittersPath);
        emitterDocument["_saved"] = savedAt;
        emitterDocument[kLegacyGlowSetting] = (diskLegacyGlow >= 0) ? (diskLegacyGlow != 0) : legacyGlow.load();
        emitterDocument["emitters"] = listedJson;
        emitterDocument["candidates"] = candidatesJson;
        const std::string emittersText = emitterDocument.dump(2) + "\n";
        writeAtomically(emittersPath, emittersText);
        writeAtomically(jsonPath.parent_path() / "rt_emitters.html", editorHtml("Ray traced light emitters", "emitters", "rt_emitters.json", utf8(jsonPath.parent_path()), emittersText));

        size_t userMaterials = 0;
        for (const auto &pair : snapshot) {
            userMaterials += pair.second.user ? 1 : 0;
        }

        size_t userEmitters = 0;
        for (const auto &pair : emitterSnapshot) {
            userEmitters += (pair.second.source == "user") ? 1 : 0;
        }

        rtDiagLog("materials: saved %zu entries to %s, kept %zu user entries; emitters: saved %zu entries, kept %zu user entries",
            snapshot.size(), utf8(jsonPath).c_str(), userMaterials, emitterSnapshot.size(), userEmitters);
    }

    void RtMaterialDatabase::setPreview(uint64_t textureKey, const std::string &preview) {
        auto entry = entries.find(textureKey);
        if ((entry != entries.end()) && (entry->second.preview != preview)) {
            entry->second.preview = preview;
            dirty = true;
        }

        for (auto &pair : emitters) {
            RtEmitterEntry &emitter = pair.second;
            const uint64_t texture = emitter.hasObserved ? emitter.observed.texture : (emitter.params.match.texture ? emitter.params.match.value.texture : 0);
            if ((texture == textureKey) && (emitter.preview != preview)) {
                emitter.preview = preview;
                dirty = true;
            }
        }
    }

    bool RtMaterialDatabase::matchEmitter(const RtEmitterSignature &draw, int map, int room, RtEmitterParams &params, bool &applies) {
        applies = false;
        if (!loaded) {
            return false;
        }

        std::scoped_lock lock(mutex);
        bool found = false;
        for (uint64_t id : listedEmitterIds) {
            auto it = emitters.find(id);
            if ((it == emitters.end()) || !it->second.params.match.matches(draw)) {
                continue;
            }

            const RtEmitterParams &candidate = it->second.params;
            const bool candidateApplies = candidate.enabled && candidate.appliesTo(draw.kind, map, room);
            if (!found || (candidateApplies && !applies)) {
                params = candidate;
                applies = candidateApplies;
                found = true;
                if (applies) {
                    break;
                }
            }
        }

        return found;
    }

    bool RtMaterialDatabase::emitterParams(uint64_t id, RtEmitterParams &params) {
        if (!loaded) {
            return false;
        }

        std::scoped_lock lock(mutex);
        auto it = emitters.find(id);
        if ((it == emitters.end()) || !it->second.listed) {
            return false;
        }

        params = it->second.params;
        return true;
    }

    bool RtMaterialDatabase::findEmitter(uint64_t id, RtEmitterEntry &entry) {
        std::scoped_lock lock(mutex);
        auto it = emitters.find(id);
        if (it == emitters.end()) {
            return false;
        }

        entry = it->second;
        return true;
    }

    void RtMaterialDatabase::recordEmitterCandidate(const RtEmitterSignature &draw, float extent, int map, int room, const char *areaName, bool wouldGlow) {
        if (!loaded || (draw.signature == 0)) {
            return;
        }

        std::scoped_lock lock(mutex);
        if (draw.texture != 0) {
            emitterTextureKeys.insert(draw.texture);
        }

        auto it = emitters.find(draw.signature);
        if ((it != emitters.end()) && it->second.hasObserved && (it->second.map >= 0)) {
            if (wouldGlow && !it->second.wouldGlow) {
                it->second.wouldGlow = true;
                dirty = true;
            }

            return;
        }

        if ((it == emitters.end()) && (emitters.size() >= kMaxMaterialEntries)) {
            return;
        }

        RtEmitterEntry &entry = emitters[draw.signature];
        if (it == emitters.end()) {
            entry.label = std::string(emitterKindName(draw.kind)) + ((draw.texture != 0) ? " effect" : " polygon effect (untextured)");
            entry.params.id = draw.signature;
            entry.params.enabled = false;
            entry.params.match.signature = true;
            entry.params.match.value = draw;
            entry.listed = false;
            entry.source = "game";
        }

        entry.observed = draw;
        entry.hasObserved = true;
        entry.wouldGlow = entry.wouldGlow || wouldGlow || ((draw.texture != 0) && (wouldGlowTextures.find(draw.texture) != wouldGlowTextures.end()));
        entry.kind = emitterKindName(draw.kind);
        if (entry.area.empty() || (entry.map < 0)) {
            entry.area = areaString(map, room, areaName);
            entry.map = map;
            entry.room = room;
        }

        char size[64];
        std::snprintf(size, sizeof(size), "%.1f across, %u triangles", extent * 2.0f, draw.triangles);
        entry.size = size;
        if (entry.preview.empty() && (draw.texture != 0)) {
            auto material = entries.find(draw.texture);
            if ((material != entries.end()) && !material->second.preview.empty()) {
                entry.preview = material->second.preview;
            }
        }

        dirty = true;
    }

    void RtMaterialDatabase::finishProbe(uint64_t key) {
        std::scoped_lock lock(mutex);
        pendingProbes.erase(key);
        probedThisSession.insert(key);
    }

    void RtMaterialDatabase::markTextureWouldGlow(uint64_t textureKey) {
        std::scoped_lock lock(mutex);
        if (!wouldGlowTextures.insert(textureKey).second) {
            return;
        }

        for (auto &pair : emitters) {
            RtEmitterEntry &entry = pair.second;
            if (entry.hasObserved && (entry.observed.texture == textureKey) && !entry.wouldGlow) {
                entry.wouldGlow = true;
                dirty = true;
            }
        }
    }

    void RtMaterialDatabase::emitterCounts(uint32_t &listed, uint32_t &enabled, uint32_t &candidates) {
        std::scoped_lock lock(mutex);
        listed = enabled = candidates = 0;
        for (const auto &pair : emitters) {
            listed += pair.second.listed ? 1 : 0;
            enabled += (pair.second.listed && pair.second.params.enabled) ? 1 : 0;
            candidates += pair.second.listed ? 0 : 1;
        }
    }

    void RtMaterialDatabase::saveIfDirty(uint32_t frame, bool force) {
        // Players (no authoring) keep everything in memory.
        if (!loaded || !authoring) {
            return;
        }

        std::unordered_map<uint64_t, RtMaterialEntry> snapshot;
        std::unordered_map<uint64_t, RtEmitterEntry> emitterSnapshot;
        {
            std::scoped_lock lock(mutex);
            if (!dirty || (!force && ((frame - lastSaveFrame) < kMaterialSaveIntervalFrames))) {
                return;
            }

            snapshot = entries;
            emitterSnapshot = emitters;
            dirty = false;
            lastSaveFrame = frame;
        }

        post([this, snapshot = std::move(snapshot), emitterSnapshot = std::move(emitterSnapshot)]() { writeFiles(snapshot, emitterSnapshot); });
    }

    uint32_t RtMaterialDatabase::packedMaterial(uint64_t key) {
        if (!loaded) {
            return 0;
        }

        std::scoped_lock lock(mutex);
        auto it = entries.find(key);
        return (it != entries.end()) ? it->second.packed : 0u;
    }

    bool RtMaterialDatabase::emissiveMaterial(uint64_t key, uint32_t &packedEmission, float radiance[3]) {
        if (!loaded) {
            return false;
        }

        std::scoped_lock lock(mutex);
        auto it = entries.find(key);
        if ((it == entries.end()) || (it->second.emission <= 0.0f)) {
            return false;
        }

        // Strength in 1/32 steps up to 8 in the top byte; the color below, or 0 for the texture's own color.
        const RtMaterialEntry &entry = it->second;
        const uint32_t strength = uint32_t(std::clamp(std::lround(entry.emission * 32.0f), 1L, 255L));
        uint32_t color = 0;
        if (!entry.emissionFromTexture) {
            for (int c = 0; c < 3; c++) {
                color |= uint32_t(std::clamp(std::lround(entry.emissionColor[c] * 255.0f), 0L, 255L)) << (c * 8);
            }

            color = std::max(color, 1u);
        }

        packedEmission = (strength << 24) | color;
        for (int c = 0; c < 3; c++) {
            radiance[c] = (entry.emissionFromTexture ? entry.averageColor[c] : entry.emissionColor[c]) * entry.emission;
        }

        return true;
    }

    bool RtMaterialDatabase::find(uint64_t key, RtMaterialEntry &entry) {
        std::scoped_lock lock(mutex);
        auto it = entries.find(key);
        if (it == entries.end()) {
            return false;
        }

        entry = it->second;
        return true;
    }

    bool RtMaterialDatabase::wantsProbe(uint64_t key, uint32_t frame) {
        if (!loaded || (key == 0)) {
            return false;
        }

        std::scoped_lock lock(mutex);
        if (probedThisSession.find(key) != probedThisSession.end()) {
            return false;
        }

        // Players use the shipped entries as they are; only textures the shipped list lacks are guessed (in memory).
        auto entry = entries.find(key);
        if ((entry != entries.end()) && (entry->second.user || !authoring)) {
            return false;
        }

        if ((entry == entries.end()) && (entries.size() >= kMaxMaterialEntries)) {
            return false;
        }

        auto pending = pendingProbes.find(key);
        return (pending == pendingProbes.end()) || ((frame - pending->second) > kMaterialProbeRetryFrames);
    }

    void RtMaterialDatabase::markProbePending(uint64_t key, uint32_t frame) {
        std::scoped_lock lock(mutex);
        pendingProbes[key] = frame;
    }

    bool RtMaterialDatabase::wantsPreview(uint64_t key) {
        // Previews are for the HTML editors, which only exist in authoring mode.
        if (!loaded || !authoring || (key == 0)) {
            return false;
        }

        std::scoped_lock lock(mutex);
        if (previewsRequested.find(key) != previewsRequested.end()) {
            return false;
        }

        auto entry = entries.find(key);
        const bool emitterTexture = (emitterTextureKeys.find(key) != emitterTextureKeys.end());
        if ((entry == entries.end()) && !emitterTexture) {
            return false;
        }

        // An existing preview file is kept; only missing ones are written.
        previewsRequested.insert(key);
        const std::string preview = "rt_materials/" + keyString(key) + ".png";
        if (std::filesystem::exists(fromUtf8(localPath).parent_path() / preview)) {
            setPreview(key, preview);
            return false;
        }

        return true;
    }

    void RtMaterialDatabase::submitPreview(uint64_t key, std::vector<uint8_t> &&pixels, uint32_t width, uint32_t height) {
        std::string directory;
        {
            std::scoped_lock lock(mutex);
            directory = previewDirectory;
        }

        if (directory.empty() || (width == 0) || (height == 0)) {
            return;
        }

        post([this, key, pixels = std::move(pixels), width, height, directory]() {
            std::error_code error;
            std::filesystem::create_directories(fromUtf8(directory), error);
            const std::string fileName = keyString(key) + ".png";
            const std::string filePath = utf8(fromUtf8(directory) / fileName);
            if (stbi_write_png(filePath.c_str(), int(width), int(height), 4, pixels.data(), int(width * 4)) == 0) {
                return;
            }

            std::scoped_lock lock(mutex);
            setPreview(key, "rt_materials/" + fileName);
        });
    }

    // A conservative guess from the texture's color and where it's drawn: anything unclear is matte.
    static RtMaterialClass guessClass(const RtMaterialStats &stats, const RtMaterialPlacement &placement) {
        const float r = stats.color[0], g = stats.color[1], b = stats.color[2];
        const float brightness = std::max(r, std::max(g, b));
        const float darkest = std::min(r, std::min(g, b));
        const float saturation = (brightness > 1e-3f) ? ((brightness - darkest) / brightness) : 0.0f;
        const float deviation = stats.luminanceDeviation;
        const bool green = (g > r) && (g > b) && (saturation > 0.2f);
        const bool brown = (r > g) && (g > b) && (saturation > 0.25f) && (brightness < 0.75f);
        const bool blue = (b > (r + 0.05f)) && (b >= g);
        const bool floor = placement.upFacing > 0.75f;
        const bool wall = placement.upFacing < 0.35f;
        if (placement.alphaTested) {
            return RtMaterialClass::Sprite;
        }

        if (placement.lit) {
            // Armor, blades and buckles: grey, fairly bright, with the highlights painted into the texture.
            if ((saturation < 0.12f) && (brightness >= 0.45f) && (brightness <= 0.95f) && (deviation >= 0.05f)) {
                return RtMaterialClass::Metal;
            }

            if ((r > g) && (g > b) && (saturation > 0.2f) && (saturation < 0.6f) && (brightness > 0.5f)) {
                return RtMaterialClass::Skin;
            }

            return RtMaterialClass::Cloth;
        }

        if (floor && blue && (saturation > 0.3f) && (brightness < 0.85f) && (deviation < 0.08f)) {
            return RtMaterialClass::Water;
        }

        if (green) {
            return RtMaterialClass::Grass;
        }

        if (brown) {
            return floor ? RtMaterialClass::Dirt : RtMaterialClass::Wood;
        }

        if (wall && (saturation < 0.2f) && (brightness >= 0.2f) && (brightness <= 0.8f) && (deviation < 0.15f)) {
            return RtMaterialClass::Stone;
        }

        if (floor && (saturation < 0.2f) && (brightness > 0.45f) && (deviation < 0.05f)) {
            return RtMaterialClass::PolishedFloor;
        }

        return RtMaterialClass::Matte;
    }

    static const char *surfaceName(const RtMaterialPlacement &placement) {
        if (placement.alphaTested) {
            return placement.lit ? "cutout on a character or object" : "cutout sprite or foliage";
        }

        if (placement.lit) {
            return "character, enemy or object model";
        }

        if (placement.upFacing > 0.75f) {
            return "level floor or ceiling";
        }

        return (placement.upFacing < 0.35f) ? "level wall" : "sloped level geometry";
    }

    void RtMaterialDatabase::noteWaterDraw(uint64_t key, const RtMaterialPlacement &placement) {
        if (!loaded || (key == 0)) {
            return;
        }

        std::scoped_lock lock(mutex);
        if ((entries.find(key) != entries.end()) || (entries.size() >= kMaxMaterialEntries)) {
            return;
        }

        RtMaterialEntry entry;
        entry.materialClass = RtMaterialClass::Water;
        const RtMaterialClassInfo &info = kClassInfo[size_t(RtMaterialClass::Water)];
        entry.roughness = info.roughness;
        entry.metalness = info.metalness;
        entry.specular = info.specular;
        entry.area = areaString(placement.map, placement.room, placement.areaName);
        entry.map = placement.map;
        entry.room = placement.room;
        entry.surface = "water surface (alpha blended, no depth write)";
        entry.texture = textureString(placement.textureWidth, placement.textureHeight, placement.textureFormat, placement.textureSize);
        entry.label = "water " + entry.surface + " in " + entry.area;
        char statsText[160];
        std::snprintf(statsText, sizeof(statsText), "found by Advanced Water Rendering: %u triangles, %.0f%% facing up; set another class to stop treating it as water",
            placement.triangleCount, placement.upFacing * 100.0f);
        entry.stats = statsText;
        packEntry(entry);
        entries[key] = entry;
        dirty = true;
    }

    void RtMaterialDatabase::addGuess(uint64_t key, const RtMaterialStats &stats, const RtMaterialPlacement &placement) {
        std::scoped_lock lock(mutex);
        pendingProbes.erase(key);
        probedThisSession.insert(key);
        // An opaque draw shows its texture whatever the alpha, so one that's see-through to the probe still gets an entry.
        if ((stats.coverage <= 0.0f) && placement.alphaTested) {
            return;
        }

        auto it = entries.find(key);
        if ((it != entries.end()) && it->second.user) {
            return;
        }

        RtMaterialEntry entry;
        if (it != entries.end()) {
            // Keep where it was first seen and its preview.
            entry = it->second;
        }

        if (entry.area.empty() || (entry.map < 0)) {
            entry.area = areaString(placement.map, placement.room, placement.areaName);
            entry.map = placement.map;
            entry.room = placement.room;
            entry.surface = surfaceName(placement);
            entry.texture = textureString(placement.textureWidth, placement.textureHeight, placement.textureFormat, placement.textureSize);
        }

        const RtMaterialClass previousClass = entry.materialClass;
        entry.materialClass = guessClass(stats, placement);
        const RtMaterialClassInfo &info = kClassInfo[size_t(entry.materialClass)];
        entry.roughness = info.roughness;
        entry.metalness = info.metalness;
        entry.specular = info.specular;
        entry.label = std::string(rtMaterialClassName(entry.materialClass)) + " " + entry.surface + " in " + entry.area;
        char statsText[160];
        const float brightness = std::max(stats.color[0], std::max(stats.color[1], stats.color[2]));
        const float darkest = std::min(stats.color[0], std::min(stats.color[1], stats.color[2]));
        const float saturation = (brightness > 1e-3f) ? ((brightness - darkest) / brightness) : 0.0f;
        std::snprintf(statsText, sizeof(statsText), "average color %.2f %.2f %.2f, saturation %.2f, variation %.2f, %u triangles",
            stats.color[0], stats.color[1], stats.color[2], saturation, stats.luminanceDeviation, placement.triangleCount);
        entry.stats = statsText;
        entry.emission = info.emission;
        for (int c = 0; c < 3; c++) {
            entry.averageColor[c] = std::clamp(stats.color[c], 0.0f, 1.0f);
        }

        packEntry(entry);

        const bool changed = (it == entries.end()) || (previousClass != entry.materialClass) || (it->second.label != entry.label);
        entries[key] = entry;
        dirty = dirty || changed;
    }

    void RtMaterialDatabase::classCounts(uint32_t counts[size_t(RtMaterialClass::Count)], uint32_t &userEntries) {
        std::scoped_lock lock(mutex);
        userEntries = 0;
        for (size_t c = 0; c < size_t(RtMaterialClass::Count); c++) {
            counts[c] = 0;
        }

        for (const auto &pair : entries) {
            counts[size_t(pair.second.materialClass)]++;
            userEntries += pair.second.user ? 1 : 0;
        }
    }

    size_t RtMaterialDatabase::size() {
        std::scoped_lock lock(mutex);
        return entries.size();
    }
};
