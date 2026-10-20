// Openjutsu — selection canvas for Object Swap.
//
// Scumble's flow is: draw a region on the image, and that region is what
// gets replaced. Ours asked for a hand-made mask PNG, which is a bad
// experience and hid the actual contract (white = replace).
//
// This paints the source frame on a canvas, lets you drag a rectangle over
// the thing to swap, and renders that rectangle as a mask image the backend
// already understands. The rectangle is also the seed for the per-frame
// tracker, so the mask follows the subject instead of drifting.

// Both files are classic scripts sharing one global scope, so these must
// NOT be named $ and $$: app.js already declares those at the top level,
// and a second `const $` throws a SyntaxError that kills this file
// outright.
const sel1 = (sel) => document.querySelector(sel);

// The selection state, in source-frame pixels.
const selection = {
  x: 0, y: 0, w: 0, h: 0,   // null until the user draws
  frame: null,                // ImageData of the source frame
  naturalW: 0, naturalH: 0,
  dragging: false,
  startX: 0, startY: 0,
  hoverHandle: null,
};

const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const HANDLE_PX = 7;

const canvas = sel1("#select-canvas");
const ctx = canvas ? canvas.getContext("2d") : null;

// -- helpers ---------------------------------------------------------------

function hasSelection() {
  return selection.w > 2 && selection.h > 2;
}

function displayScale() {
  if (!canvas || !selection.naturalW) return 1;
  return canvas.width / selection.naturalW;
}

// The mask the backend wants: white where the region is, black elsewhere,
// at the source frame's native resolution.
//
// The edge is SOFT on purpose. A hard-edged rectangle tells the model
// "here is a frame" and it renders the replacement as a picture inside
// that frame, with a visible seam around it. A feathered edge gives it a
// gradual transition and it blends into the scene instead.
function selectionMaskCanvas() {
  const c = document.createElement("canvas");
  c.width = selection.naturalW || 1;
  c.height = selection.naturalH || 1;
  const cx = c.getContext("2d");
  cx.fillStyle = "#000";
  cx.fillRect(0, 0, c.width, c.height);

  const feather = Math.max(4, Math.round(Math.min(selection.w, selection.h) * 0.06));
  const pad = feather * 2;
  const tmp = document.createElement("canvas");
  tmp.width = c.width;
  tmp.height = c.height;
  const tx = tmp.getContext("2d");
  tx.fillStyle = "#000";
  tx.fillRect(0, 0, tmp.width, tmp.height);
  tx.filter = `blur(${feather}px)`;
  tx.fillStyle = "#fff";
  tx.fillRect(selection.x, selection.y, selection.w, selection.h);
  tx.filter = "none";

  cx.drawImage(tmp, 0, 0);
  return c;
}

function selectionMaskBlob() {
  return new Promise((resolve) => {
    selectionMaskCanvas().toBlob((b) => resolve(b), "image/png");
  });
}

// -- painting --------------------------------------------------------------

function drawCanvas() {
  if (!ctx) return;
  const { width: W, height: H } = canvas;
  ctx.clearRect(0, 0, W, H);

  // Checkerboard, so a transparent image still reads as an image.
  ctx.fillStyle = "#141419";
  ctx.fillRect(0, 0, W, H);

  if (selection.frame) {
    // putImageData cannot scale, and it does not take the 5-argument form
    // — draw the frame through an offscreen canvas at the display size.
    if (!selection._scaled) {
      const off = document.createElement("canvas");
      off.width = selection.naturalW;
      off.height = selection.naturalH;
      off.getContext("2d").putImageData(selection.frame, 0, 0);
      selection._scaled = off;
    }
    ctx.drawImage(selection._scaled, 0, 0, W, H);
  }

  if (!hasSelection()) return;

  const s = displayScale();
  const x = selection.x * s, y = selection.y * s;
  const w = selection.w * s, h = selection.h * s;

  // Dim everything outside the selection.
  ctx.fillStyle = "rgba(8, 8, 12, 0.55)";
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.rect(x, y, w, h);
  ctx.fill("evenodd");

  // Marching ants: a dashed white/black outline.
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = "#000";
  ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  ctx.lineDashOffset = 0;
  ctx.strokeStyle = "#fff";
  ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  ctx.setLineDash([]);

  // Corner handles.
  ctx.fillStyle = "#fff";
  ctx.strokeStyle = "#000";
  for (const p of handlePositions(x, y, w, h)) {
    ctx.beginPath();
    ctx.rect(p.x - HANDLE_PX / 2, p.y - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
    ctx.fill();
    ctx.stroke();
  }
}

function handlePositions(x, y, w, h) {
  const mx = x + w / 2, my = y + h / 2;
  return [
    { id: "nw", x: x, y: y }, { id: "n", x: mx, y: y }, { id: "ne", x: x + w, y: y },
    { id: "e", x: x + w, y: my }, { id: "se", x: x + w, y: y + h },
    { id: "s", x: mx, y: y + h }, { id: "sw", x: x, y: y + h }, { id: "w", x, y: my },
  ];
}

function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const s = displayScale();
  return {
    x: (evt.clientX - rect.left) * (selection.naturalW / rect.width) / s * s,
    y: (evt.clientY - rect.top) * (selection.naturalH / rect.height) / s * s,
  };
}

function hitHandle(pt) {
  if (!hasSelection()) return null;
  const s = displayScale();
  const x = selection.x * s, y = selection.y * s;
  const w = selection.w * s, h = selection.h * s;
  for (const p of handlePositions(x, y, w, h)) {
    if (Math.abs(pt.x * s - p.x) <= HANDLE_PX && Math.abs(pt.y * s - p.y) <= HANDLE_PX) {
      return p.id;
    }
  }
  return null;
}

// -- loading the source frame ----------------------------------------------

// Ask the server for a still of the clip. The server always has ffmpeg,
// so this works even when the browser cannot decode the file's codec.
async function fetchServerFrame(file) {
  const fd = new FormData();
  fd.append("source_video", file);
  const resp = await fetch("/api/preview-frame", { method: "POST", body: fd });
  if (!resp.ok) throw new Error(`preview failed (${resp.status})`);
  return await resp.blob();
}

async function paintFrame(blob) {
  const bitmap = await createImageBitmap(blob);
  selection.naturalW = bitmap.width;
  selection.naturalH = bitmap.height;

  const maxW = 640;
  const scale = Math.min(1, maxW / selection.naturalW);
  canvas.width = Math.round(selection.naturalW * scale);
  canvas.height = Math.round(selection.naturalH * scale);

  const tmp = document.createElement("canvas");
  tmp.width = selection.naturalW;
  tmp.height = selection.naturalH;
  const cx = tmp.getContext("2d");
  cx.drawImage(bitmap, 0, 0);
  selection.frame = cx.getImageData(0, 0, tmp.width, tmp.height);
  selection._scaled = null;   // invalidate the cached scaled copy
  drawCanvas();
  updateSelectionStatus();
}

async function loadSourceFrame(file) {
  if (!canvas) return;
  // Server-side first: it is the only path that works for every codec.
  try {
    await paintFrame(await fetchServerFrame(file));
    return;
  } catch (err) {
    // Fall back to decoding locally.
  }
  await loadFrameLocally(file);
}

async function loadFrameLocally(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.playsInline = true;

  await new Promise((resolve, reject) => {
    let settled = false;
    const ok = () => { if (!settled) { settled = true; resolve(); } };
    const bad = () => {
      if (!settled) { settled = true; reject(new Error("could not read that video")); }
    };
    video.addEventListener("loadeddata", ok, { once: true });
    video.addEventListener("error", bad, { once: true });
    // Never hang the UI on a codec the browser cannot decode.
    setTimeout(ok, 6000);
  });

  // loadeddata can fire before the dimensions are known, and drawing a
  // zero-sized video throws. Wait for a frame that actually has pixels.
  if (!video.videoWidth || !video.videoHeight) {
    await new Promise((resolve) => {
      let settled = false;
      const done = () => { if (!settled) { settled = true; resolve(); } };
      video.addEventListener("loadeddata", done, { once: true });
      video.addEventListener("canplay", done, { once: true });
      setTimeout(done, 4000);
    });
  }
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error("that video has no readable frames");
  }

  // Seek a little way in: the first frame is often a cut or a black frame.
  // Guard the wait — a short clip may not fire "seeked" at all, and without
  // this the whole load hangs and the canvas stays empty.
  try {
    video.currentTime = Math.min(0.5, (video.duration || 1) / 4);
    await new Promise((resolve) => {
      let settled = false;
      const done = () => { if (!settled) { settled = true; resolve(); } };
      video.addEventListener("seeked", done, { once: true });
      setTimeout(done, 1500);
    });
  } catch (e) { /* keep the first frame */ }

  selection.naturalW = video.videoWidth;
  selection.naturalH = video.videoHeight;

  const maxW = 640;
  const scale = Math.min(1, maxW / selection.naturalW);
  canvas.width = Math.round(selection.naturalW * scale);
  canvas.height = Math.round(selection.naturalH * scale);

  const tmp = document.createElement("canvas");
  tmp.width = selection.naturalW;
  tmp.height = selection.naturalH;
  tmp.getContext("2d").drawImage(video, 0, 0);
  selection.frame = tmp.getContext("2d").getImageData(0, 0, tmp.width, tmp.height);
  selection._scaled = null;   // invalidate the cached scaled copy

  URL.revokeObjectURL(url);
  drawCanvas();
  updateSelectionStatus();
}

// -- pointer interaction ---------------------------------------------------

if (canvas) {
  canvas.addEventListener("pointerdown", (e) => {
    if (!selection.frame) return;
    const pt = canvasPoint(e);
    const handle = hitHandle(pt);
    canvas.setPointerCapture(e.pointerId);
    selection.dragging = true;
    selection.startX = pt.x;
    selection.startY = pt.y;
    selection.hoverHandle = handle;
    if (handle) {
      selection.dragMode = "resize";
    } else {
      selection.dragMode = "new";
      selection.x = pt.x; selection.y = pt.y; selection.w = 0; selection.h = 0;
    }
    drawCanvas();
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!selection.dragging) {
      const h = hitHandle(canvasPoint(e));
      canvas.style.cursor = h ? `${h}-resize` : "crosshair";
      return;
    }
    const pt = canvasPoint(e);
    if (selection.dragMode === "new") {
      selection.x = Math.min(selection.startX, pt.x);
      selection.y = Math.min(selection.startY, pt.y);
      selection.w = Math.abs(pt.x - selection.startX);
      selection.h = Math.abs(pt.y - selection.startY);
    } else {
      resizeSelection(selection.hoverHandle, pt);
    }
    drawCanvas();
    updateSelectionStatus();
  });

  canvas.addEventListener("pointerup", (e) => {
    selection.dragging = false;
    canvas.releasePointerCapture(e.pointerId);
    drawCanvas();
    updateSelectionStatus();
  });

  canvas.addEventListener("pointerleave", () => {
    if (!selection.dragging) { canvas.style.cursor = "crosshair"; }
  });
}

function resizeSelection(handle, pt) {
  let { x, y, w, h } = selection;
  if (handle.includes("w")) { w = selection.x + selection.w - pt.x; x = pt.x; }
  if (handle.includes("e")) { w = pt.x - selection.x; }
  if (handle.includes("n")) { h = selection.y + selection.h - pt.y; y = pt.y; }
  if (handle.includes("s")) { h = pt.y - selection.y; }
  if (w < 0) { x += w; w = -w; }
  if (h < 0) { y += h; h = -h; }
  selection.x = x; selection.y = y; selection.w = w; selection.h = h;
}

function updateSelectionStatus() {
  const el = sel1("#selection-status");
  if (!el) return;
  if (!hasSelection()) {
    el.textContent = "No region selected yet — drag a box over the thing to replace.";
    return;
  }
  const pct = ((selection.w * selection.h) /
    (selection.naturalW * selection.naturalH) * 100).toFixed(1);
  el.textContent =
    `Region: ${Math.round(selection.w)}×${Math.round(selection.h)} px ` +
    `at (${Math.round(selection.x)}, ${Math.round(selection.y)}) · ${pct}% of the frame`;
}

function clearSelection() {
  selection.x = selection.y = selection.w = selection.h = 0;
  drawCanvas();
  updateSelectionStatus();
}

// Expose to the rest of the app.
window.openjutsuSelection = {
  has: hasSelection,
  blob: selectionMaskBlob,
  rect: () => ({
    x: Math.round(selection.x), y: Math.round(selection.y),
    w: Math.round(selection.w), h: Math.round(selection.h),
    frameW: selection.naturalW, frameH: selection.naturalH,
  }),
  clear: clearSelection,
  loadFrame: loadSourceFrame,
  draw: drawCanvas,
};
