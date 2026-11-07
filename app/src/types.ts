// Shared types for Trellis Studio.

export type Resolution = 512 | 1024 | 1536;
export type BgRemoval = "auto" | "birefnet" | "threshold";
export type Uv = "xatlas" | "box";
/** How several views of one object are combined (server field multi_mode). */
export type MultiMode = "stochastic" | "multidiffusion";
/** Which flow stages see every view (server field multi_stages); the others use the main view. */
export type MultiStages = "all" | "shape,tex" | "tex";

/** Generation knobs mapped onto trellis-server's POST /generate form fields. */
export interface GenParams {
  resolution: Resolution;
  seed: number;
  bgRemoval: BgRemoval;
  uv: Uv;
  /** false => geometry only (server field texture=off). */
  texture: boolean;
  /** Only sent with several input images. Absent on records saved before multi-view. */
  multiMode?: MultiMode;
  multiStages?: MultiStages;
}

export const DEFAULT_PARAMS: GenParams = {
  resolution: 1024,
  seed: 42,
  bgRemoval: "auto",
  uv: "xatlas",
  texture: true,
  multiMode: "stochastic",
  multiStages: "all",
};

/** Upper bound on input views: each one costs a bg-removal + DINOv3 pass. */
export const MAX_VIEWS = 8;

/** Written by the installer, read by the Tauri shell + surfaced to the UI. */
export interface AppConfig {
  serverBin: string;
  modelsDir: string;
  backend: string;
  gpu: number;
  host: string;
  port: number;
  /** where generated GLBs are auto-saved (Tauri only). */
  outputDir: string;
  /** true once the shell has a usable config.json; false => "setup needed". */
  configured: boolean;
}

/** One persisted generation (IndexedDB record). */
export interface GenRecord {
  id: string;
  ts: number;
  name: string;
  params: GenParams;
  input: Blob; // source image (the main view)
  extraInputs?: Blob[]; // further views of the same object, in order
  glb: Blob; // resulting model/gltf-binary
  thumb: Blob | null; // model-viewer snapshot for the gallery
}
