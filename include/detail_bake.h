// Detail maps baked from the dense surface onto the simplified, unwrapped one: a tangent-space
// normal map and ambient occlusion. The QEM output keeps the silhouette but loses the decoded
// surface's small relief; these maps put it back for a game engine.
#pragma once
#include <cstdint>
#include <vector>

namespace trellis {

struct BakedMesh;
class TriBvh;

struct DetailBakeParams {
    int   ao_rays   = 32;      // cosine-weighted rays per texel; 0 = no occlusion
    float ao_radius = 0.05f;   // occlusion search distance, in the [-0.5,0.5] object cube
    float max_dist  = 0.02f;   // texels farther than this from the dense surface keep a flat normal
    float ray_eps   = 1e-3f;   // occlusion ray start offset off the dense surface
    int   threads   = 0;       // 0 = all hardware threads
};

struct DetailBakeStats {
    size_t texels = 0;         // covered texels
    size_t missed = 0;         // covered texels with no dense surface within max_dist
    double mean_ao = 1.0;      // mean occlusion term over covered texels (1 = open)
    double seconds = 0;
};

// lo: the unwrapped low mesh (TRELLIS space, uv in [0,1], v down the image like uv_bake).
// hi_verts / hi_faces + hi_bvh: the dense surface (any winding; normals are turned toward the
// low surface). Writes `normal_rgba` [T*T*4] and, when ao_rays > 0, the occlusion into the R
// channel of lo.mr (glTF: occlusionTexture sharing the metallic-roughness image).
//
// Tangent frame: per low triangle, T = normalize(dP/du), B = normalize(dP/dv) (v as stored,
// down the image), N = the interpolated smooth vertex normal that write_glb_textured exports.
// That is the frame an engine rebuilds from screen derivatives of position and uv
// (n = normalize(T*x + B*y + N*z)), so each texel stores the exact inverse of that sum; for
// a well-conditioned unwrap it is within a few degrees of MikkTSpace.
DetailBakeStats bake_detail_maps(BakedMesh& lo, const std::vector<float>& hi_verts,
                                 const std::vector<int32_t>& hi_faces, const TriBvh& hi_bvh,
                                 const DetailBakeParams& params, std::vector<uint8_t>& normal_rgba);

// Drop the faces nobody can see: from each face centre, `rays` directions per side (both
// sides, so the winding does not matter) are cast against the mesh itself; a face is kept as
// soon as one ray escapes past `reach` (the [-0.5,0.5] object fits well within 2). What goes
// are the closed inner shells and enclosed sheets the narrow-band remesh leaves inside thick
// parts -- invisible, yet they take their share of the UV atlas. Unused vertices are
// compacted away. In place; returns the number of faces dropped (`area_dropped` = their
// share of the total area when given).
int drop_enclosed_faces(std::vector<float>& verts, std::vector<int32_t>& faces, int rays = 64,
                        float reach = 2.0f, double* area_dropped = nullptr, int threads = 0);

} // namespace trellis
