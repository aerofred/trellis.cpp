import "./ui.css";
import { generate, health } from "./api";
import { loadConfig } from "./config";
import { renderSettings } from "./settings";
import { all, clear as clearStore, del as removeRecord, get as getRecord, isEphemeral, newId, put } from "./store";
import { isTauri, listen, saveBytes, saveToOutputDir } from "./tauri";
import { Viewer } from "./viewer";
import { DEFAULT_PARAMS, MAX_VIEWS, type GenParams, type GenRecord } from "./types";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

// ---- element refs ----
const dropzone = $("dropzone");
const fileInput = $<HTMLInputElement>("file-input");
const inputPreview = $<HTMLImageElement>("input-preview");
const dropHint = $("dropzone-hint");
const viewsEl = $("views");
const viewsList = $("views-list");
const viewsCaption = $("views-caption");
const multiCtl = $("multi-ctl");
const generateBtn = $<HTMLButtonElement>("generate-btn");
const progress = $("progress");
const progressStage = $("progress-stage");
const progressElapsed = $("progress-elapsed");
const cancelBtn = $<HTMLButtonElement>("cancel-btn");
const resetViewBtn = $<HTMLButtonElement>("reset-view");
const saveGlbBtn = $<HTMLButtonElement>("save-glb");
const viewerCaption = $("viewer-caption");
const galleryEl = $("gallery");
const clearGalleryBtn = $("clear-gallery");
const backendBadge = $("backend-badge");
const serverDot = $("server-dot");
const serverLabel = $("server-label");
const setupBanner = $("setup-banner");

// ---- state ----
const viewer = new Viewer($("viewer-mount"));
/** Input views of one object; inputs[0] is the main view (shown large, named after). */
interface InputView {
  blob: Blob;
  name: string;
  url: string;
}
let inputs: InputView[] = [];
let inputName = "input.png";
let currentGlb: Blob | null = null;
let activeId: string | null = null;
let serverOnline = false;
let generating = false;
let abort: AbortController | null = null;
let elapsedTimer: number | null = null;
let galleryUrls: string[] = [];
let warnedEphemeral = false;

// ---- controls -> params ----
function readParams(): GenParams {
  const res = parseInt(($("ctl-res") as HTMLSelectElement).value, 10);
  const seed = parseInt(($("ctl-seed") as HTMLInputElement).value, 10);
  return {
    resolution: (res === 512 || res === 1536 ? res : 1024) as GenParams["resolution"],
    seed: isNaN(seed) ? DEFAULT_PARAMS.seed : seed,
    bgRemoval: ($("ctl-bg") as HTMLSelectElement).value as GenParams["bgRemoval"],
    uv: ($("ctl-uv") as HTMLSelectElement).value as GenParams["uv"],
    texture: ($("ctl-tex") as HTMLSelectElement).value !== "off",
    multiMode: ($("ctl-multi-mode") as HTMLSelectElement).value as GenParams["multiMode"],
    multiStages: ($("ctl-multi-stages") as HTMLSelectElement).value as GenParams["multiStages"],
  };
}

function applyParams(p: GenParams): void {
  ($("ctl-res") as HTMLSelectElement).value = String(p.resolution);
  ($("ctl-seed") as HTMLInputElement).value = String(p.seed);
  ($("ctl-bg") as HTMLSelectElement).value = p.bgRemoval;
  ($("ctl-uv") as HTMLSelectElement).value = p.uv;
  // records saved before this option existed have no field: treat as textured
  ($("ctl-tex") as HTMLSelectElement).value = p.texture === false ? "off" : "on";
  ($("ctl-multi-mode") as HTMLSelectElement).value = p.multiMode ?? DEFAULT_PARAMS.multiMode!;
  ($("ctl-multi-stages") as HTMLSelectElement).value = p.multiStages ?? DEFAULT_PARAMS.multiStages!;
}

// ---- toasts ----
function toast(msg: string, kind: "" | "ok" | "err" = ""): void {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

// ---- input images ----
// Several images are several views of the SAME object (front, back, side…): the
// server conditions the flow stages on all of them. Dropping/picking on the big
// zone replaces the set; the "+" tile and drops onto the strip add views.
function renderInputs(): void {
  const main = inputs[0];
  inputName = main?.name || "input.png";
  inputPreview.classList.toggle("hidden", !main);
  dropHint.classList.toggle("hidden", !!main);
  if (main) inputPreview.src = main.url;
  else inputPreview.removeAttribute("src");

  viewsEl.classList.toggle("hidden", !main);
  multiCtl.classList.toggle("hidden", inputs.length < 2);
  viewsList.innerHTML = "";
  inputs.forEach((v, i) => {
    const item = document.createElement("div");
    item.className = "vitem" + (i === 0 ? " main" : "");
    item.title = i === 0 ? `${v.name} (main view)` : `${v.name} — click to make it the main view`;
    const img = document.createElement("img");
    img.src = v.url;
    item.appendChild(img);
    if (i === 0 && inputs.length > 1) {
      const tag = document.createElement("div");
      tag.className = "vtag";
      tag.textContent = "main";
      item.appendChild(tag);
    }
    const del = document.createElement("button");
    del.className = "vdel";
    del.textContent = "✕";
    del.title = "Remove this view";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      URL.revokeObjectURL(v.url);
      inputs.splice(i, 1);
      renderInputs();
    });
    item.appendChild(del);
    item.addEventListener("click", () => {
      if (i === 0) return;
      inputs.unshift(...inputs.splice(i, 1));
      renderInputs();
    });
    viewsList.appendChild(item);
  });
  if (inputs.length < MAX_VIEWS) {
    const add = document.createElement("div");
    add.className = "vadd";
    add.textContent = "+";
    add.title = "Add another view of the same object";
    add.addEventListener("click", () => pickFiles(true));
    viewsList.appendChild(add);
  }
  viewsCaption.textContent =
    inputs.length > 1
      ? `${inputs.length} views of one object · click a view to make it the main one`
      : "Add views of the same object (back, side…) with +";
  updateGenerateEnabled();
}

function setInputs(files: { blob: Blob; name: string }[], append = false): void {
  if (!files.length) return;
  if (!append) {
    inputs.forEach((v) => URL.revokeObjectURL(v.url));
    inputs = [];
  }
  const room = MAX_VIEWS - inputs.length;
  if (files.length > room) toast(`At most ${MAX_VIEWS} views — extra images ignored.`, "err");
  for (const f of files.slice(0, Math.max(0, room))) {
    inputs.push({ blob: f.blob, name: f.name || "input.png", url: URL.createObjectURL(f.blob) });
  }
  renderInputs();
}

function setInput(blob: Blob, name: string): void {
  setInputs([{ blob, name }]);
}

let pickAppend = false;
function pickFiles(append: boolean): void {
  pickAppend = append;
  fileInput.value = "";
  fileInput.click();
}

dropzone.addEventListener("click", () => pickFiles(false));
dropzone.addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") pickFiles(false);
});
fileInput.addEventListener("change", () => {
  const files = [...(fileInput.files ?? [])].filter(isImage);
  setInputs(files.map((f) => ({ blob: f, name: f.name })), pickAppend);
});

/** Image files of a drop; toasts about the rest. */
function droppedImages(e: Event): { blob: Blob; name: string }[] {
  const files = [...((e as DragEvent).dataTransfer?.files ?? [])];
  const bad = files.filter((f) => !isImage(f));
  if (bad.length) toast(`"${bad[0].name}" isn't an image — drop a PNG, JPEG or WebP.`, "err");
  return files.filter(isImage).map((f) => ({ blob: f, name: f.name }));
}
// HTML5 drag-and-drop. In Tauri this needs `dragDropEnabled: false` on the
// window (tauri.conf.json): otherwise the webview's native file-drop handler
// swallows the drop on Windows/WebView2 and the page never sees the files.

// A file dropped anywhere outside the drop zone must not make the webview
// navigate to it (which would replace the whole app with the image).
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|gif|tiff?)$/i;
const isImage = (f: File) => f.type.startsWith("image/") || IMAGE_EXT.test(f.name);

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  }),
);
// dragleave also fires when the pointer moves onto a child (the preview <img>),
// which made the highlight flicker; only clear it when actually leaving the zone.
dropzone.addEventListener("dragleave", (e) => {
  if (!dropzone.contains((e as DragEvent).relatedTarget as Node | null)) dropzone.classList.remove("drag");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  setInputs(droppedImages(e));
});
// Dropping onto the view strip adds views instead of replacing them.
["dragenter", "dragover"].forEach((ev) =>
  viewsEl.addEventListener(ev, (e) => {
    e.preventDefault();
    viewsEl.classList.add("drag");
  }),
);
viewsEl.addEventListener("dragleave", (e) => {
  if (!viewsEl.contains((e as DragEvent).relatedTarget as Node | null)) viewsEl.classList.remove("drag");
});
viewsEl.addEventListener("drop", (e) => {
  e.preventDefault();
  viewsEl.classList.remove("drag");
  setInputs(droppedImages(e), true);
});
window.addEventListener("paste", async (e: ClipboardEvent) => {
  for (const item of e.clipboardData?.items ?? []) {
    const image = item.type.startsWith("image/") && item.getAsFile();
    if (image) return setInput(image, image.name || "pasted.png");
  }
  // WebKitGTK can omit image data from ClipboardEvent. Since paste was
  // user-triggered, try reading the clipboard item list for an image.
  const items = await navigator.clipboard.read().catch(() => []);
  for (const item of items) {
    const type = item.types.find((type) => type.startsWith("image/"));
    if (type) {
      const image = await item.getType(type).catch(() => null);
      if (image) return setInput(image, "pasted.png");
    }
  }
});

// ---- generate ----
function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function updateGenerateEnabled(): void {
  generateBtn.disabled = !(serverOnline && inputs.length && !generating);
  generateBtn.textContent = inputs.length > 1 ? `Generate 3D from ${inputs.length} views` : "Generate 3D";
}

async function doGenerate(): Promise<void> {
  if (!inputs.length || generating) return;
  const params = readParams();
  const images = inputs.map((v) => v.blob);
  const nViews = images.length;
  const viewsTag = nViews > 1 ? ` · ${nViews} views` : "";
  generating = true;
  updateGenerateEnabled();
  progress.classList.remove("hidden");
  progressStage.textContent = "starting…";
  const started = Date.now();
  progressElapsed.textContent = "0:00";
  elapsedTimer = window.setInterval(() => {
    progressElapsed.textContent = fmtElapsed(Date.now() - started);
  }, 1000);
  abort = new AbortController();
  try {
    const { glb } = await generate(images, params, abort.signal);
    currentGlb = glb;

    const rec: GenRecord = {
      id: newId(),
      ts: Date.now(),
      name: inputName,
      params,
      input: images[0],
      extraInputs: images.slice(1),
      glb,
      thumb: null,
    };

    // 1) Write the GLB to the output folder FIRST. This on-disk file is the real
    // deliverable and must survive any later failure — a dead gallery DB (some
    // WebKitGTK builds can't open IndexedDB) or a WebGL/model-viewer crash must
    // never cost the user a successful generation.
    let savedPath: string | null = null;
    if (isTauri()) {
      try {
        const bytes = new Uint8Array(await glb.arrayBuffer());
        const base = inputName.replace(/\.[^.]+$/, "") || "model";
        const views = nViews > 1 ? `_${nViews}views` : "";
        const fname = `${base}_${params.resolution}${views}_seed${params.seed}_${rec.id}.glb`;
        savedPath = await saveToOutputDir(fname, bytes);
      } catch (e) {
        toast(`Auto-save to output folder failed: ${(e as Error).message}`, "err");
      }
    }
    toast(savedPath ? `Saved to ${savedPath}` : "Generation complete", "ok");

    // 2) Add it to the gallery. The store falls back to in-memory if IndexedDB
    // is unavailable, so this never throws and never blocks the save above.
    await put(rec);
    if (isEphemeral() && !warnedEphemeral) {
      warnedEphemeral = true;
      toast(
        "Gallery won't persist across restarts (IndexedDB unavailable on this system) — " +
          "but every generation is still saved to your output folder.",
        "err",
      );
    }
    activeId = rec.id;
    setViewerTools(true);
    viewerCaption.textContent = `${params.resolution}${viewsTag} · seed ${params.seed} · ${(glb.size / 1e6).toFixed(1)} MB`;
    await refreshGallery();

    // 3) Best-effort 3D preview + gallery thumbnail — never blocks the save above.
    try {
      await viewer.load(glb);
      const thumb = await viewer.thumbnail();
      if (thumb) {
        rec.thumb = thumb;
        await put(rec);
        await refreshGallery();
      }
    } catch (e) {
      toast(`3D preview couldn't render (your result is still saved): ${(e as Error).message}`, "err");
    }
  } catch (e) {
    if (abort?.signal.aborted) toast("Generation cancelled");
    else toast((e as Error).message || "generation failed", "err");
  } finally {
    generating = false;
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    progress.classList.add("hidden");
    updateGenerateEnabled();
  }
}

generateBtn.addEventListener("click", doGenerate);
cancelBtn.addEventListener("click", () => abort?.abort());

// ---- viewer tools ----
function setViewerTools(on: boolean): void {
  resetViewBtn.disabled = !on;
  saveGlbBtn.disabled = !on;
}
resetViewBtn.addEventListener("click", () => viewer.resetView());
saveGlbBtn.addEventListener("click", async () => {
  if (!currentGlb) return;
  const bytes = new Uint8Array(await currentGlb.arrayBuffer());
  const base = inputName.replace(/\.[^.]+$/, "") || "model";
  const ok = await saveBytes(`${base}.glb`, bytes);
  if (ok) toast("Saved", "ok");
});

// ---- gallery ----
async function loadRecord(id: string): Promise<void> {
  const rec = await getRecord(id);
  if (!rec) return;
  try {
    await viewer.load(rec.glb);
  } catch (e) {
    toast((e as Error).message, "err");
    return;
  }
  const extra = rec.extraInputs ?? [];
  setInputs([
    { blob: rec.input, name: rec.name },
    ...extra.map((blob, i) => ({ blob, name: `view${i + 1}` })),
  ]);
  applyParams(rec.params);
  currentGlb = rec.glb;
  activeId = rec.id;
  setViewerTools(true);
  const viewsTag = extra.length ? ` · ${extra.length + 1} views` : "";
  viewerCaption.textContent = `${rec.name} · ${rec.params.resolution}${viewsTag} · seed ${rec.params.seed}`;
  await refreshGallery();
}

async function refreshGallery(): Promise<void> {
  galleryUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryUrls = [];
  const recs = await all();
  galleryEl.innerHTML = "";
  if (!recs.length) {
    galleryEl.innerHTML = `<div class="gallery-empty">No generations yet.</div>`;
    return;
  }
  for (const r of recs) {
    const item = document.createElement("div");
    item.className = "gitem" + (r.id === activeId ? " active" : "");
    item.title = `${r.name} · ${new Date(r.ts).toLocaleString()}`;

    const img = document.createElement("img");
    const src = r.thumb ?? r.input;
    const url = URL.createObjectURL(src);
    galleryUrls.push(url);
    img.src = url;
    item.appendChild(img);

    const meta = document.createElement("div");
    meta.className = "gmeta";
    const nv = 1 + (r.extraInputs?.length ?? 0);
    meta.textContent = nv > 1 ? `${r.params.resolution} · ${nv}v` : `${r.params.resolution}`;
    item.appendChild(meta);

    const delBtn = document.createElement("button");
    delBtn.className = "gdel";
    delBtn.textContent = "✕";
    delBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await removeRecord(r.id);
      if (activeId === r.id) activeId = null;
      await refreshGallery();
    });
    item.appendChild(delBtn);

    item.addEventListener("click", () => loadRecord(r.id));
    galleryEl.appendChild(item);
  }
}

clearGalleryBtn.addEventListener("click", async () => {
  if (!confirm("Delete all saved generations?")) return;
  await clearStore();
  activeId = null;
  await refreshGallery();
});

// ---- settings modal ----
const modal = $("settings-modal");
async function openSettings(): Promise<void> {
  await renderSettings($("settings-body"), () => {
    pollHealth();
    modal.classList.add("hidden");
    toast("Settings applied");
  });
  modal.classList.remove("hidden");
}
$("settings-btn").addEventListener("click", openSettings);
$("banner-settings").addEventListener("click", openSettings);
$("settings-close").addEventListener("click", () => modal.classList.add("hidden"));
modal.addEventListener("click", (e) => {
  if (e.target === modal) modal.classList.add("hidden");
});

// ---- server status ----
async function pollHealth(): Promise<void> {
  const cfg = await loadConfig(true);
  backendBadge.textContent = cfg.backend !== "unknown" ? cfg.backend : "—";
  const ok = await health();
  serverOnline = ok;
  serverDot.className = "dot " + (ok ? "ok" : "err");
  serverLabel.textContent = ok ? "ready" : cfg.configured ? "offline" : "setup needed";
  const needSetup = !ok && !cfg.configured;
  setupBanner.classList.toggle("hidden", !needSetup);
  if (needSetup) {
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Trellis Studio isn't set up yet — run the installer or point it at your models directory.";
  } else if (!ok && cfg.configured) {
    setupBanner.classList.remove("hidden");
    (setupBanner.querySelector("span") as HTMLElement).textContent =
      "Server is offline — it may still be loading, or check the models directory in settings.";
  }
  updateGenerateEnabled();
}

// ---- server log -> progress (Tauri only) ----
listen<string>("server-log", (line) => {
  const t = String(line).trim();
  if (generating && t) progressStage.textContent = t;
});

// ---- boot ----
async function boot(): Promise<void> {
  setViewerTools(false);
  await refreshGallery();
  await pollHealth();
  window.setInterval(pollHealth, 4000);
  if (!isTauri()) {
    // Browser mode: no shell to report a backend.
    backendBadge.textContent = "browser";
  }
}
boot();
