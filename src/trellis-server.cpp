// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate    multipart/form-data with an "image" file part (repeat it for
//                      more views of the same object, main view first; trellis model
//                      only, combined per "multi_mode" = stochastic|multidiffusion and
//                      "multi_stages" = comma list of ss,shape,tex or all, "view_weights"
//                      = comma list, main view first, "multi_interval" = LO,HI); optional text
//                      fields "seed", "seeds" (comma list: seed search), "resolution" (512/1024/1536), "bg_removal"
//                      (threshold|birefnet), "uv" (xatlas = default, unique
//                      chart space; box = faster projection), "band" (narrow-band
//                      DC remesh band width, default 1 — see --band), "model"
//                      (trellis|pixal3d — one directory holds both), "faces"
//                      (QEM decimation face target, >= 1000 or "auto" — see
//                      --faces), and for
//                      pixal3d the camera: "fov" (degrees), "mesh_scale",
//                      "extend_pixel", and "texture" (on|off; off = geometry
//                      only, like --no-texture). Returns model/gltf-binary.
//
// Launch-time defaults come from CLI flags (see trellis::parse_args);
// each request copies those defaults and applies its own overrides. The model
// directory is resolved once; each request runs the full pipeline via trellis_run()
// (per-stage load/free, like trellis-cli), serialized by a mutex. Keeping the
// process resident avoids re-initializing the Vulkan backend on every request.
#include "trellis_args.h"
#include "trellis_run.h"
#include "httplib.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

// std::tmpnam on MSVC yields drive-root paths ("\sXXX.N") that a non-elevated
// process cannot write; stage scratch files in the real temp directory instead.
std::string temp_stem(const std::string& workdir = {}) {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = workdir.empty() ? std::filesystem::temp_directory_path(ec)
                                                : std::filesystem::absolute(workdir);
    if (ec) dir = ".";
    std::filesystem::create_directories(dir,ec);
    if (ec) throw std::runtime_error("cannot create request workdir: " + ec.message());
    auto n = counter.fetch_add(1);
    return (dir / ("trellis-req-" + std::to_string(n))).string();
}

}  // namespace

int main(int argc, char** argv) {
    // Stage progress goes to stdout, which is fully buffered when piped (e.g.
    // under Lemonade's output capture) — keep it line-visible for diagnostics.
    setvbuf(stdout, nullptr, _IONBF, 0);

    trellis::TrellisParams base;
    if (!trellis::parse_args(argc, argv, base)) {
        trellis::print_usage(argv[0], /*server=*/true);
        return base.help ? 0 : 1;
    }

    std::mutex gen_mu;
    httplib::Server svr;

    // Trellis Studio (and any browser client) calls this server from a different
    // origin — a Tauri webview is tauri://localhost / http://tauri.localhost, and a
    // browser-served UI is another port — so every response needs permissive CORS
    // headers, and a multipart POST with non-simple headers may be preflighted with
    // OPTIONS. Applied to every route via the post-routing hook + a catch-all OPTIONS.
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "86400");
        res.set_header("Access-Control-Expose-Headers", "X-Trellis-Seed");
    });
    svr.Options(R"(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;  // headers added by the post-routing handler above
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'image' file part\"}", "application/json");
            return;
        }
        // Repeat the "image" part for further views of the same object; the first is the main
        // view (see TrellisParams::extra_images). Parts keep their upload order.
        const std::vector<httplib::MultipartFormData> images = req.get_file_values("image");

        // Per-request params start from the launch defaults, then apply overrides.
        trellis::TrellisParams p = base;
        if (req.has_file("seed")) p.seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());
        for (const char* field : {"seeds", "seed_orientations"}) {
            if (!req.has_file(field)) continue;
            std::string error;
            const std::string v = req.get_file_value(field).content;
            if (!(std::string(field) == "seeds" ? trellis::parse_seeds_arg(v.c_str(), p, error)
                                                : trellis::parse_seed_orientations_arg(v.c_str(), p, error))) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        if (req.has_file("resolution")) p.set_res(atoi(req.get_file_value("resolution").content.c_str()));
        if (req.has_file("bg_removal")) p.birefnet = (req.get_file_value("bg_removal").content == "birefnet") ? 1 : 0;
        if (req.has_file("uv")) p.xatlas = (req.get_file_value("uv").content == "xatlas");
        if (req.has_file("band")) p.band = atoi(req.get_file_value("band").content.c_str());
        if (req.has_file("faces")) {
            std::string error;
            if (!trellis::parse_faces_arg(req.get_file_value("faces").content.c_str(), p, error) ||
                !trellis::check_faces_compat(p, error)) {   // launch-time --decim + per-request faces
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        // Both families share one model directory (the Pixal3D flows carry a pixal3d_ prefix),
        // so a single resident server can serve either -- the client just has to say which.
        if (req.has_file("model")) {
            const std::string& m = req.get_file_value("model").content;
            if      (m == "pixal3d") p.family = trellis::ModelFamily::Pixal3D;
            else if (m == "trellis") p.family = trellis::ModelFamily::Trellis;
            else { res.status = 400; res.set_content("model must be trellis or pixal3d\n", "text/plain"); return; }
        }
        // Camera, for pixal3d. Unlike every other knob these describe the IMAGE rather than the
        // run, so a launch-time default is close to useless on a server taking arbitrary uploads.
        for (const auto& field : {std::pair<const char*, const char*>{"fov", "fov"},
                                  {"mesh_scale", "mesh-scale"}, {"extend_pixel", "extend-pixel"}}) {
            if (!req.has_file(field.first)) continue;
            std::string error;
            if (!trellis::parse_camera_arg(field.second, req.get_file_value(field.first).content.c_str(), p, error)) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        if (req.has_file("texture")) {
            const std::string& t = req.get_file_value("texture").content;
            if      (t == "off" || t == "0" || t == "false") p.texture = false;
            else if (t == "on"  || t == "1" || t == "true")  p.texture = true;
            else { res.status = 400; res.set_content("texture must be on or off\n", "text/plain"); return; }
        }
        if (req.has_file("webp")) {
            const std::string& w = req.get_file_value("webp").content;
            p.webp = (w == "off" || w == "0" || w == "false") ? 0
                   : (w == "on"  || w == "1" || w == "true")  ? 1 : -1;
        }
        for (const char* field : {"normal_map", "cull_hidden"}) {
            if (!req.has_file(field)) continue;
            const std::string& w = req.get_file_value(field).content;
            bool& target = std::string(field) == "normal_map" ? p.normal_map : p.cull_hidden;
            if      (w == "on"  || w == "1" || w == "true")  target = true;
            else if (w == "off" || w == "0" || w == "false") target = false;
            else { res.status = 400; res.set_content(std::string(field) + " must be on or off\n", "text/plain"); return; }
        }
        for (const char* field : {"ao_rays", "ao_radius"}) {
            if (!req.has_file(field)) continue;
            const std::string v = req.get_file_value(field).content;
            std::string error;
            if (!(std::string(field) == "ao_rays" ? trellis::parse_ao_rays_arg(v.c_str(), p, error)
                                                  : trellis::parse_ao_radius_arg(v.c_str(), p, error))) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        for (const char* field : {"multi_mode", "multi_stages", "view_weights", "multi_interval"}) {
            if (!req.has_file(field)) continue;
            const std::string v = req.get_file_value(field).content;
            const std::string f = field;
            std::string error;
            const bool ok = f == "multi_mode"   ? trellis::parse_multi_mode_arg(v.c_str(), p, error)
                          : f == "multi_stages" ? trellis::parse_multi_stages_arg(v.c_str(), p, error)
                          : f == "view_weights" ? trellis::parse_view_weights_arg(v.c_str(), p, error)
                                                : trellis::parse_multi_interval_arg(v.c_str(), p, error);
            if (!ok) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }

        std::string stem;
        try {
            stem = temp_stem(p.retopo ? p.retopo_workdir : std::string{});
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to stage request files\"}", "application/json");
            return;
        }
        p.image  = stem + ".png";
        p.extra_images.clear();   // launch-time -i views never leak into a request
        for (size_t v = 1; v < images.size(); ++v) p.extra_images.push_back(stem + "_view" + std::to_string(v) + ".png");
        p.output = stem + ".glb";
        {
            std::string error;
            if (!trellis::check_multi_image_compat(p, error)) {
                res.status = 400;
                res.set_content(error + "\n", "text/plain");
                return;
            }
        }
        auto remove_inputs = [&] {
            std::remove(p.image.c_str());
            for (const auto& e : p.extra_images) std::remove(e.c_str());
        };

        std::string glb;
        uint32_t seed_used = p.seed;
        std::string error_message = "3D reconstruction failed";
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            size_t total_bytes = 0;
            for (size_t v = 0; v < images.size(); ++v) {
                total_bytes += images[v].content.size();
                if (!write_file_bytes(v == 0 ? p.image : p.extra_images[v - 1], images[v].content)) {
                    remove_inputs();
                    res.status = 500;
                    res.set_content("{\"error\":\"failed to stage input image\"}", "application/json");
                    return;
                }
            }
            fprintf(stderr, "[trellis-server] generate: %zu image(s), %zu bytes, seed %u, res %s, bg %s, uv %s, faces %d, texture %s",
                    images.size(), total_bytes, p.seed, p.cascade ? std::to_string(p.hr_res).c_str() : "512",
                    p.birefnet < 0 ? "auto" : (p.birefnet ? "birefnet" : "threshold"), p.xatlas ? "xatlas" : "box",
                    trellis::qem_face_target(p), p.texture ? "on" : "off");
            if (images.size() > 1)
                fprintf(stderr, ", %s, stages%s%s%s",
                        p.multi_mode == trellis::MultiImageMode::MultiDiffusion ? "multidiffusion" : "stochastic",
                        p.multi_ss ? " ss" : "", p.multi_shape ? " shape" : "", p.multi_tex ? " tex" : "");
            fprintf(stderr, "\n");
            try {
                int rc = trellis_run(p, &seed_used);
                if (rc == 0) glb = read_file_bytes(p.output);
            } catch (const std::exception& e) {
                fprintf(stderr, "[trellis-server] generate failed: %s\n", e.what());
                error_message = e.what();
            }
            remove_inputs();
            std::remove(p.output.c_str());
            // trellis_run also writes sibling debug artifacts; clean them up too.
            std::remove((stem + ".ply").c_str());
            std::remove((stem + "_base.png").c_str());
            std::remove((stem + "_normal.png").c_str());   // --normal-map sidecars
            std::remove((stem + "_orm.png").c_str());
        }

        if (glb.empty()) {
            res.status = 500;
            std::string escaped;
            for (char c : error_message) {
                switch (c) {
                    case '"':  escaped += "\\\""; break;
                    case '\\': escaped += "\\\\"; break;
                    case '\n': escaped += "\\n";  break;
                    case '\r': escaped += "\\r";  break;
                    case '\t': escaped += "\\t";  break;
                    default:
                        if ((unsigned char)c < 0x20) break;
                        escaped += c;
                }
            }
            res.set_content("{\"error\":\"" + escaped + "\"}", "application/json");
            return;
        }
        // The seed actually used (a "seeds" search picks one): lets a client store a reproducible record.
        res.set_header("X-Trellis-Seed", std::to_string(seed_used));
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            base.models.c_str(), base.gpu, base.host.c_str(), base.port);
    if (!svr.listen(base.host, base.port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", base.host.c_str(), base.port);
        return 1;
    }
    return 0;
}
