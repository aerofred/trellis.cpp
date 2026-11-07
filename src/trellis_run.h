#pragma once
#include "trellis_args.h"

// Run the full TRELLIS.2 image->3D pipeline described by `p`: reconstruct
// `p.image` into a textured GLB at `p.output`. Returns 0 on success. All
// behavior (resolution, bg removal, texture, guidance, ...) comes from `p`,
// which trellis-cli / trellis-server populate via trellis::parse_args.
// `seed_used` (optional) receives the seed the model was generated with: the --seeds choice, the
// auto seed, or `p.seed`.
int trellis_run(const trellis::TrellisParams& p, uint32_t* seed_used = nullptr);
