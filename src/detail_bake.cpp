#include "detail_bake.h"
#include "uv_bake.h"
#include "tri_bvh.h"
#include "mesh_glb.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace trellis {

namespace {

struct V3 { float x, y, z; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline V3 normalized(V3 a, V3 fallback = {0, 0, 1}) {
    const float l = std::sqrt(dot(a, a));
    return l > 1e-20f ? a * (1.0f / l) : fallback;
}
inline V3 at(const std::vector<float>& v, int64_t i) { return {v[3*i], v[3*i+1], v[3*i+2]}; }

inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
inline float unit(uint32_t h) { return (h >> 8) * (1.0f / 16777216.0f); }

// Multi-source BFS: every unwritten texel takes the value of its nearest written one, so
// mip-mapping and bilinear filtering never pull the black gutter into the charts.
template <int C, typename T>
void dilate(std::vector<T>& img, std::vector<uint8_t> mask, int W) {
    std::vector<int> queue;
    queue.reserve((size_t)W * W / 2);
    for (int i = 0; i < W * W; ++i) if (mask[(size_t)i]) queue.push_back(i);
    static const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};
    for (size_t qi = 0; qi < queue.size(); ++qi) {
        const int t = queue[qi], x = t % W, y = t / W;
        for (int d = 0; d < 4; ++d) {
            const int xx = x + DX[d], yy = y + DY[d];
            if (xx < 0 || yy < 0 || xx >= W || yy >= W) continue;
            const size_t s = (size_t)yy * W + xx;
            if (mask[s]) continue;
            for (int c = 0; c < C; ++c) img[C*s + c] = img[C*(size_t)t + c];
            mask[s] = 1;
            queue.push_back((int)s);
        }
    }
}

}  // namespace

DetailBakeStats bake_detail_maps(BakedMesh& lo, const std::vector<float>& hi_verts,
                                 const std::vector<int32_t>& hi_faces, const TriBvh& hi_bvh,
                                 const DetailBakeParams& prm, std::vector<uint8_t>& normal_rgba) {
    const auto t_start = std::chrono::steady_clock::now();
    DetailBakeStats st;
    const int T = lo.T;
    const int64_t V = (int64_t)lo.verts.size() / 3, F = (int64_t)lo.faces.size() / 3;
    normal_rgba.assign((size_t)T * T * 4, 0);
    if (T <= 0 || F == 0) return st;

    // Smooth normals exactly as the GLB exports them (the weld is rotation-invariant, so
    // computing them in TRELLIS space gives the exported ones up to the global axis swap).
    const std::vector<float> lon = welded_normals_gltf(lo.verts.data(), V, lo.faces.data(), F);
    // Dense-surface vertex normals, area-weighted.
    const int64_t HV = (int64_t)hi_verts.size() / 3, HF = (int64_t)hi_faces.size() / 3;
    std::vector<float> hn((size_t)HV * 3, 0.0f);
    for (int64_t f = 0; f < HF; ++f) {
        const int32_t a = hi_faces[3*f], b = hi_faces[3*f+1], c = hi_faces[3*f+2];
        const V3 n = cross(at(hi_verts, b) - at(hi_verts, a), at(hi_verts, c) - at(hi_verts, a));
        for (int32_t v : {a, b, c}) { hn[3*v] += n.x; hn[3*v+1] += n.y; hn[3*v+2] += n.z; }
    }
    for (int64_t v = 0; v < HV; ++v) {
        const V3 n = normalized(at(hn, v));
        hn[3*v] = n.x; hn[3*v+1] = n.y; hn[3*v+2] = n.z;
    }

    // Per low triangle: T = dP/du, B = dP/dv (unit), the frame screen derivatives rebuild.
    std::vector<V3> ftan((size_t)F), fbit((size_t)F);
    for (int64_t f = 0; f < F; ++f) {
        const int32_t i0 = lo.faces[3*f], i1 = lo.faces[3*f+1], i2 = lo.faces[3*f+2];
        const V3 e1 = at(lo.verts, i1) - at(lo.verts, i0), e2 = at(lo.verts, i2) - at(lo.verts, i0);
        const float du1 = lo.uv[2*i1] - lo.uv[2*i0], dv1 = lo.uv[2*i1+1] - lo.uv[2*i0+1];
        const float du2 = lo.uv[2*i2] - lo.uv[2*i0], dv2 = lo.uv[2*i2+1] - lo.uv[2*i0+1];
        const float det = du1 * dv2 - du2 * dv1;
        if (std::fabs(det) < 1e-14f) { ftan[f] = {0, 0, 0}; fbit[f] = {0, 0, 0}; continue; }
        ftan[f] = normalized((e1 * dv2 - e2 * dv1) * (1.0f / det), {0, 0, 0});
        fbit[f] = normalized((e2 * du1 - e1 * du2) * (1.0f / det), {0, 0, 0});
    }

    // Rasterize the atlas at texel centres (same rule as uv_bake); first triangle wins.
    struct Texel { int32_t t, f; float w0, w1; };
    std::vector<Texel> texels;
    std::vector<uint8_t> covered((size_t)T * T, 0);
    for (int64_t f = 0; f < F; ++f) {
        const int32_t idx[3] = { lo.faces[3*f], lo.faces[3*f+1], lo.faces[3*f+2] };
        float px[3], py[3];
        for (int j = 0; j < 3; ++j) { px[j] = lo.uv[2*idx[j]] * T; py[j] = lo.uv[2*idx[j]+1] * T; }
        int x0 = (int)std::floor(std::min({px[0], px[1], px[2]})), x1 = (int)std::ceil(std::max({px[0], px[1], px[2]}));
        int y0 = (int)std::floor(std::min({py[0], py[1], py[2]})), y1 = (int)std::ceil(std::max({py[0], py[1], py[2]}));
        x0 = std::max(0, x0); y0 = std::max(0, y0); x1 = std::min(T - 1, x1); y1 = std::min(T - 1, y1);
        const float d = (py[1]-py[2])*(px[0]-px[2]) + (px[2]-px[1])*(py[0]-py[2]);
        if (std::fabs(d) < 1e-9f) continue;
        for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
            const float fx = x + 0.5f, fy = y + 0.5f;
            const float w0 = ((py[1]-py[2])*(fx-px[2]) + (px[2]-px[1])*(fy-py[2])) / d;
            const float w1 = ((py[2]-py[0])*(fx-px[2]) + (px[0]-px[2])*(fy-py[2])) / d;
            if (w0 < -0.001f || w1 < -0.001f || 1 - w0 - w1 < -0.001f) continue;
            const size_t t = (size_t)y * T + x;
            if (covered[t]) continue;
            covered[t] = 1;
            texels.push_back({(int32_t)t, (int32_t)f, w0, w1});
        }
    }
    st.texels = texels.size();

    std::vector<float> ao((size_t)T * T, 1.0f);
    std::atomic<size_t> next{0}, missed{0};
    std::atomic<long long> ao_sum{0};   // in 1/65536 units
    const int rays = std::max(0, prm.ao_rays);
    auto work = [&]() {
        size_t local_missed = 0; long long local_ao = 0;
        for (;;) {
            const size_t begin = next.fetch_add(4096);
            if (begin >= texels.size()) break;
            const size_t end = std::min(texels.size(), begin + 4096);
            for (size_t k = begin; k < end; ++k) {
                const Texel& tx = texels[k];
                const int32_t i0 = lo.faces[3*tx.f], i1 = lo.faces[3*tx.f+1], i2 = lo.faces[3*tx.f+2];
                const float w2 = 1.0f - tx.w0 - tx.w1;
                const V3 P = at(lo.verts, i0) * tx.w0 + at(lo.verts, i1) * tx.w1 + at(lo.verts, i2) * w2;
                const V3 N = normalized(at(lon, i0) * tx.w0 + at(lon, i1) * tx.w1 + at(lon, i2) * w2);
                V3 h = N, Q = P;
                const float p[3] = {P.x, P.y, P.z};
                const TriBvh::Hit hit = hi_bvh.closest(p, prm.max_dist);
                if (hit.face >= 0) {
                    const int32_t a = hi_faces[3*hit.face], b = hi_faces[3*hit.face+1], c = hi_faces[3*hit.face+2];
                    const V3 A = at(hi_verts, a), B = at(hi_verts, b), C = at(hi_verts, c);
                    Q = {hit.point[0], hit.point[1], hit.point[2]};
                    // Barycentrics of the closest point (it lies on the triangle).
                    const V3 e0 = B - A, e1 = C - A, ep = Q - A;
                    const float d00 = dot(e0, e0), d01 = dot(e0, e1), d11 = dot(e1, e1);
                    const float d20 = dot(ep, e0), d21 = dot(ep, e1);
                    const float den = d00 * d11 - d01 * d01;
                    float bv = 0, bw = 0;
                    if (std::fabs(den) > 1e-30f) { bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; }
                    const float bu = 1.0f - bv - bw;
                    h = normalized(at(hn, a) * bu + at(hn, b) * bv + at(hn, c) * bw, N);
                    if (dot(h, N) < 0.0f) h = h * -1.0f;   // the dense mesh's winding is not trusted
                } else {
                    ++local_missed;
                }

                // Encode: solve [T B N] c = h (Cramer), so normalize(T*cx + B*cy + N*cz) == h.
                V3 cvec = {0, 0, 1};
                const V3 Tt = ftan[(size_t)tx.f], Bt = fbit[(size_t)tx.f];
                const float det = dot(Tt, cross(Bt, N));
                if (std::fabs(det) > 1e-4f) {
                    cvec = { dot(h, cross(Bt, N)) / det, dot(Tt, cross(h, N)) / det, dot(Tt, cross(Bt, h)) / det };
                    cvec = normalized(cvec);
                }
                uint8_t* o = &normal_rgba[4 * (size_t)tx.t];
                auto enc = [](float v) { const float q = (v * 0.5f + 0.5f) * 255.0f + 0.5f; return (uint8_t)std::min(255.0f, std::max(0.0f, q)); };
                o[0] = enc(cvec.x); o[1] = enc(cvec.y); o[2] = enc(cvec.z); o[3] = 255;

                if (rays > 0) {
                    // Cosine-weighted, stratified in the polar term, rotated per texel.
                    const V3 axis = std::fabs(h.x) < 0.7f ? V3{1, 0, 0} : V3{0, 1, 0};
                    const V3 u = normalized(cross(h, axis)), v = cross(h, u);
                    const V3 origin = Q + h * prm.ray_eps;
                    const float org[3] = {origin.x, origin.y, origin.z};
                    const float rot = unit(hash32((uint32_t)tx.t * 9781u + 17u)) * 6.2831853f;
                    int occ = 0;
                    for (int r = 0; r < rays; ++r) {
                        const float r1 = (r + unit(hash32((uint32_t)tx.t * 31u + (uint32_t)r * 7919u + 1u))) / rays;
                        const float phi = rot + r * 2.39996323f;   // golden-angle spiral
                        const float rr = std::sqrt(r1);
                        const V3 dir = u * (rr * std::cos(phi)) + v * (rr * std::sin(phi)) + h * std::sqrt(std::max(0.0f, 1.0f - r1));
                        const float dd[3] = {dir.x, dir.y, dir.z};
                        if (hi_bvh.occluded(org, dd, prm.ao_radius)) ++occ;
                    }
                    const float a = 1.0f - (float)occ / rays;
                    ao[(size_t)tx.t] = a;
                    local_ao += (long long)(a * 65536.0f);
                } else {
                    local_ao += 65536;
                }
            }
        }
        missed += local_missed;
        ao_sum += local_ao;
    };
    unsigned nt = prm.threads > 0 ? (unsigned)prm.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < nt; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    st.missed = missed;
    st.mean_ao = st.texels ? (double)ao_sum / 65536.0 / st.texels : 1.0;

    // Gutters: nearest covered value, so filtering never reaches a flat or black texel.
    dilate<4>(normal_rgba, covered, T);
    for (size_t t = 0; t < covered.size(); ++t)
        if (!covered[t] && normal_rgba[4*t+3] == 0) { normal_rgba[4*t] = 128; normal_rgba[4*t+1] = 128; normal_rgba[4*t+2] = 255; normal_rgba[4*t+3] = 255; }
    if (rays > 0) {
        dilate<1>(ao, covered, T);
        for (size_t t = 0; t < ao.size(); ++t)
            lo.mr[4*t] = (uint8_t)std::min(255.0f, std::max(0.0f, ao[t] * 255.0f + 0.5f));
    }
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    return st;
}

int drop_enclosed_faces(std::vector<float>& verts, std::vector<int32_t>& faces, int rays, float reach,
                        double* area_dropped, int threads) {
    const int64_t V = (int64_t)verts.size() / 3, F = (int64_t)faces.size() / 3;
    if (area_dropped) *area_dropped = 0;
    if (F == 0 || rays <= 0) return 0;
    const TriBvh bvh = TriBvh::build(verts.data(), V, faces.data(), F);
    // Fibonacci hemisphere around +z, mapped onto each face's frame.
    std::vector<V3> dirs((size_t)rays);
    for (int i = 0; i < rays; ++i) {
        const float z = 1.0f - (i + 0.5f) / rays;            // uniform in solid angle
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = i * 2.39996323f;
        dirs[(size_t)i] = {r * std::cos(phi), r * std::sin(phi), z};
    }
    std::vector<uint8_t> keep((size_t)F, 1);
    std::vector<double> area((size_t)F, 0.0);
    std::atomic<int64_t> next{0};
    auto work = [&]() {
        for (;;) {
            const int64_t begin = next.fetch_add(256);
            if (begin >= F) break;
            const int64_t end = std::min(F, begin + 256);
            for (int64_t f = begin; f < end; ++f) {
                const V3 a = at(verts, faces[3*f]), b = at(verts, faces[3*f+1]), c = at(verts, faces[3*f+2]);
                const V3 cr = cross(b - a, c - a);
                const float len = std::sqrt(dot(cr, cr));
                area[(size_t)f] = 0.5 * len;
                if (len < 1e-14f) continue;                   // degenerate: invisible either way, keep
                const V3 n = cr * (1.0f / len);
                const V3 axis = std::fabs(n.x) < 0.7f ? V3{1, 0, 0} : V3{0, 1, 0};
                const V3 u = normalized(cross(n, axis)), v = cross(n, u);
                const V3 ctr = (a + b + c) * (1.0f / 3.0f);
                bool seen = false;
                for (int side = 0; side < 2 && !seen; ++side) {
                    const V3 ns = side ? n * -1.0f : n;
                    const V3 us = side ? u * -1.0f : u;
                    const V3 o = ctr + ns * 1e-5f;
                    const float org[3] = {o.x, o.y, o.z};
                    for (const V3& d : dirs) {
                        const V3 w = us * d.x + v * d.y + ns * d.z;
                        const float dd[3] = {w.x, w.y, w.z};
                        if (!bvh.occluded(org, dd, reach)) { seen = true; break; }
                    }
                }
                keep[(size_t)f] = seen;
            }
        }
    };
    unsigned nt = threads > 0 ? (unsigned)threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < nt; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();

    double total = 0, gone = 0;
    std::vector<int32_t> nf;
    nf.reserve(faces.size());
    for (int64_t f = 0; f < F; ++f) {
        total += area[(size_t)f];
        if (keep[(size_t)f]) { nf.insert(nf.end(), {faces[3*f], faces[3*f+1], faces[3*f+2]}); }
        else gone += area[(size_t)f];
    }
    const int dropped = (int)(F - (int64_t)nf.size() / 3);
    if (dropped == 0) return 0;
    std::vector<int32_t> remap((size_t)V, -1);
    std::vector<float> nv;
    nv.reserve(verts.size());
    for (int32_t& i : nf) {
        if (remap[(size_t)i] < 0) {
            remap[(size_t)i] = (int32_t)(nv.size() / 3);
            nv.insert(nv.end(), {verts[3*(size_t)i], verts[3*(size_t)i+1], verts[3*(size_t)i+2]});
        }
        i = remap[(size_t)i];
    }
    verts.swap(nv);
    faces.swap(nf);
    if (area_dropped) *area_dropped = total > 0 ? gone / total : 0.0;
    return dropped;
}

}  // namespace trellis
