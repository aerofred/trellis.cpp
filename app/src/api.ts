// HTTP client for the resident trellis-server (see src/trellis-server.cpp):
//   GET  /health   -> "ok"
//   POST /generate  multipart: image file(s) + seed/resolution/bg_removal/uv/texture
//                   (+ multi_mode/multi_stages with several views) fields
//                   -> model/gltf-binary, or JSON {"error": "..."} on failure.

import { apiBase, loadConfig } from "./config";
import type { GenParams } from "./types";

async function base(): Promise<string> {
  return apiBase(await loadConfig());
}

export async function health(timeoutMs = 2000): Promise<boolean> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${await base()}/health`, { signal: ctrl.signal });
    return res.ok && (await res.text()).trim() === "ok";
  } catch {
    return false;
  } finally {
    clearTimeout(t);
  }
}

/**
 * trellis-server maps bg_removal: "auto" keeps the server default (no field).
 * Several images = several "image" parts, main view first; the multi-view fields
 * are only sent then (the server rejects them with Pixal3D only when views > 1).
 */
function toForm(images: Blob[], p: GenParams): FormData {
  const fd = new FormData();
  images.forEach((image, i) => fd.append("image", image, i ? `view${i}.png` : "input.png"));
  fd.append("seed", String(p.seed));
  fd.append("resolution", String(p.resolution));
  if (p.bgRemoval !== "auto") fd.append("bg_removal", p.bgRemoval);
  fd.append("uv", p.uv);
  if (!p.texture) fd.append("texture", "off");
  if (images.length > 1) {
    fd.append("multi_mode", p.multiMode ?? "stochastic");
    fd.append("multi_stages", p.multiStages ?? "all");
    const w = p.viewWeight ?? 1;
    if (w !== 1) fd.append("view_weights", ["1", ...images.slice(1).map(() => String(w))].join(","));
  }
  const n = p.seedSearch ?? 0;
  if (n > 1) {
    // seed, seed+1, ... (the server wants seeds >= 1)
    const first = Math.max(1, p.seed);
    fd.append("seeds", Array.from({ length: n }, (_, i) => String(first + i)).join(","));
    fd.append("seed_orientations", p.seedOrientations ?? "any");
  }
  if (p.faces != null && p.faces > 0) fd.append("faces", String(p.faces));
  if (p.normalMap) fd.append("normal_map", "on");
  if (p.cullHidden) fd.append("cull_hidden", "on");
  return fd;
}

export interface GenerateResult {
  /** Seed the model was generated with (X-Trellis-Seed; the search's pick with seedSearch). */
  seedUsed?: number;
  glb: Blob;
}

export async function generate(
  images: Blob[],
  params: GenParams,
  signal?: AbortSignal,
): Promise<GenerateResult> {
  if (!images.length) throw new Error("no input image");
  const res = await fetch(`${await base()}/generate`, {
    method: "POST",
    body: toForm(images, params),
    signal,
  });
  if (!res.ok) {
    let msg = `generation failed (HTTP ${res.status})`;
    // JSON {"error": ...} from the pipeline, plain text from parameter validation.
    const body = await res.text().catch(() => "");
    try {
      const j = JSON.parse(body);
      if (j && typeof j.error === "string") msg = j.error;
    } catch {
      if (body.trim()) msg = body.trim();
    }
    throw new Error(msg);
  }
  const glb = await res.blob();
  if (glb.size === 0) throw new Error("server returned an empty model");
  const seedHeader = res.headers.get("X-Trellis-Seed");
  const seedUsed = seedHeader != null && /^\d+$/.test(seedHeader) ? parseInt(seedHeader, 10) : undefined;
  return { glb, seedUsed };
}
