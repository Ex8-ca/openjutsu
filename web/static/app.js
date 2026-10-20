// Openjutsu frontend — minimal vanilla JS.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// refs is now {person?: File, scene?: File, outfit?: File}
const refs = {};

// ---------------------------------------------------------------------------
// Access token
// ---------------------------------------------------------------------------
// The server can gate the expensive /api/* endpoints behind a shared secret
// (OPENJUTSU_TOKEN). When that is set, the browser needs the token as a
// cookie or a query parameter. A 401 is recoverable: ask for the token,
// store it, and retry - do not dead-end the user with a raw error string.
const TOKEN_KEY = "openjutsu_token";

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setToken(value) {
  const t = (value || "").trim();
  if (t) {
    localStorage.setItem(TOKEN_KEY, t);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

// The server sets the cookie when the app is opened as /?token=...; the
// stored token is sent as a query parameter as well so the app still works
// when cookies are blocked.
function withToken(url) {
  const t = getToken();
  if (!t) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(t)}`;
}

async function apiFetch(url, options) {
  const opts = options || {};
  const headers = new Headers((opts && opts.headers) || {});
  const stored = getToken();
  if (stored) headers.set("X-Openjutsu-Token", stored);
  opts.headers = headers;
  let resp = await fetch(withToken(url), opts);
  if (resp.status === 401) {
    const token = await promptForToken();
    if (token) {
      setToken(token);
      resp = await fetch(withToken(url), opts);
    }
  }
  return resp;
}

// Ask for the token. Resolves with the token string, or null if cancelled.
function promptForToken() {
  return new Promise((resolve) => {
    const existing = document.getElementById("token-prompt");
    if (existing) existing.remove();
    const wrap = document.createElement("div");
    wrap.id = "token-prompt";
    wrap.className = "token-prompt";
    wrap.innerHTML = `
      <div class="token-prompt-card">
        <h3>This server is access-controlled</h3>
        <p>The <code>/api/*</code> endpoints need an access token.
           Open the app URL with <code>?token=YOUR_TOKEN</code> once, or paste
           the token here.</p>
        <input id="token-input" type="password" placeholder="access token" autocomplete="off">
        <div class="token-prompt-actions">
          <button id="token-cancel" class="btn-secondary" type="button">Cancel</button>
          <button id="token-ok" class="btn-primary" type="button">Unlock</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);
    const input = wrap.querySelector("#token-input");
    const done = (value) => { wrap.remove(); resolve(value); };
    wrap.querySelector("#token-ok").addEventListener("click", () => done(input.value.trim()));
    wrap.querySelector("#token-cancel").addEventListener("click", () => done(null));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") done(input.value.trim());
      if (e.key === "Escape") done(null);
    });
    input.focus();
  });
}

// -- reference image upload (per role slot) -----------------------------
function renderThumb(slot) {
  const slotEl = document.querySelector(`.ref-slot[data-role="${slot}"]`);
  const thumb = slotEl.querySelector(".thumb");
  thumb.innerHTML = "";
  if (refs[slot]) {
    const url = URL.createObjectURL(refs[slot]);
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `
      <img src="${url}" alt="${refs[slot].name}">
      <button class="x" type="button" aria-label="remove">×</button>
    `;
    div.querySelector(".x").addEventListener("click", () => {
      URL.revokeObjectURL(url);
      refs[slot] = null;
      renderThumb(slot);
    });
    thumb.appendChild(div);
  }
}

$$(".ref-slot").forEach((slotEl) => {
  const role = slotEl.dataset.role;
  const input = slotEl.querySelector('input[type="file"]');
  const btn = slotEl.querySelector(".add-btn");
  btn.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) {
      refs[role] = f;
      renderThumb(role);
    }
    e.target.value = "";
  });
});

// -- source video -> selection canvas --------------------------------------
const sourceInput = $("#source_video");
if (sourceInput) {
  sourceInput.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f && window.openjutsuSelection) {
      window.openjutsuSelection.loadFrame(f).catch((err) => {
        const el = $("#selection-status");
        if (el) el.textContent = `could not read that video: ${err}`;
      });
    }
  });
}

const trackBox = $("#track-region");
const trackNote = $("#track-note");
if (trackBox && trackNote) {
  trackBox.addEventListener("change", () => {
    trackNote.hidden = !trackBox.checked;
  });
}

const clearBtn = $("#selection-clear");
if (clearBtn) {
  clearBtn.addEventListener("click", () => {
    if (window.openjutsuSelection) window.openjutsuSelection.clear();
  });
}

// -- mode switch ----------------------------------------------------------
const swapCard = $("#swap-mask-card");
const maskInput = $("#swap_mask");
$$('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isSwap = $('input[name="mode"]:checked').value === "swap";
    swapCard.hidden = !isSwap;
    if (isSwap && !maskInput.files.length) maskInput.setAttribute("required", "required");
    else maskInput.removeAttribute("required");
  });
});
maskInput.addEventListener("change", () => {
  const f = maskInput.files[0];
  $("#swap-mask-name").textContent = f ? f.name : "";
});

// -- prompt toggle --------------------------------------------------------
const promptEl = $("#prompt");
$("#prompt-on").addEventListener("change", (e) => {
  promptEl.disabled = !e.target.checked;
});

// -- prompt enhance --------------------------------------------------------
$("#enhance-prompt").addEventListener("click", async () => {
  const btn = $("#enhance-prompt");
  const orig = promptEl.value.trim();
  if (!orig) {
    promptEl.value = "A natural-looking scene";
  }
  btn.disabled = true;
  btn.textContent = "✨ Enhancing…";
  try {
    const resp = await apiFetch("/api/enhance-prompt", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: `prompt=${encodeURIComponent(promptEl.value)}`
    });
    if (resp.ok) {
      const d = await resp.json();
      promptEl.value = d.enhanced || promptEl.value;
    }
  } catch (e) {
    // Silent: keep what user typed
  } finally {
    btn.disabled = false;
    btn.textContent = "✨ Enhance";
  }
});

// -- generate ------------------------------------------------------------
const goBtn = $("#go");
goBtn.addEventListener("click", async () => {
  const video = $("#source_video").files[0];
  if (!video) return alert("Add a reference video first.");

  goBtn.disabled = true;
  goBtn.textContent = "Submitting…";

  const fd = new FormData();
  fd.append("source_video", video);
  const mode = $('input[name="mode"]:checked').value;
  fd.append("mode", mode);
  if (mode === "swap") {
    // A drawn region is the primary path: it is exactly what the user
    // pointed at. Fall back to a mask file only if nothing was drawn.
    const sel = window.openjutsuSelection;
    if (sel && sel.has()) {
      const blob = await sel.blob();
      if (blob) {
        fd.append("mask_image", blob, "selection.png");
        const rect = sel.rect();
        fd.append("mask_rect", JSON.stringify(rect));
        if (document.querySelector("#track-region")?.checked) {
          fd.append("track_region", "1");
        }
      }
    } else {
      const mask = maskInput.files[0];
      if (!mask) {
        alert("Draw a region over the thing to replace (or provide a mask image).");
        return;
      }
      fd.append("mask_image", mask);
    }
  }
  for (const role of ["person", "scene", "outfit"]) {
    if (refs[role]) fd.append("ref_images", refs[role]);
    if (refs[role]) fd.append("refs_role", role);
  }
  fd.append("prompt", $("#prompt-on").checked ? $("#prompt").value : "");
  fd.append("quality", $("#quality").value);

  $("#status").classList.remove("hidden");
  $("#status-line").textContent = "uploading…";
  $("#outputs").innerHTML = "";
  $("#log").hidden = true;

  // Submit. The public tunnel can drop a response to a large multipart
  // upload, so retry a few times before giving up — the request usually
  // reached the app and the response was lost on the way back.
  let resp = null;
  let lastErr = "";
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      resp = await apiFetch("/api/generate", { method: "POST", body: fd });
      break;
    } catch (e) {
      lastErr = String(e);
      if (attempt < 4) {
        $("#status-line").textContent = `uploading… retrying (${attempt}/3)`;
        await new Promise((r) => setTimeout(r, 2500 * attempt));
      }
    }
  }
  if (!resp) {
    $("#status-line").classList.add("error");
    $("#status-line").textContent =
      `network error after 4 attempts: ${lastErr}. The connection to the server was dropped — try again, or run Openjutsu locally.`;
    goBtn.disabled = false; goBtn.textContent = "Generate ✨";
    return;
  }

  if (!resp.ok) {
    const t = await resp.text();
    $("#status-line").classList.add("error");
    if (resp.status === 401) {
      $("#status-line").textContent =
        `unauthorized: this server needs an access token. Reload with ?token=YOUR_TOKEN, or enter it when prompted.`;
    } else if (resp.status === 429) {
      $("#status-line").textContent =
        `busy: ${t} — this box renders one clip at a time.`;
    } else {
      $("#status-line").textContent = `submit failed (${resp.status}): ${t}`;
    }
    goBtn.disabled = false; goBtn.textContent = "Generate ✨";
    return;
  }

  const { job_id, status } = await resp.json();
  $("#status-line").textContent = `${status} (job ${job_id.slice(0, 8)})`;

  // Poll status
  const start = Date.now();
  const poll = async () => {
    let r;
    try { r = await apiFetch(`/api/jobs/${job_id}`); } catch { return setTimeout(poll, 4000); }
    if (!r.ok) return setTimeout(poll, 4000);
    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    $("#status-line").textContent = `${data.status} · ${elapsed}s`;
    $("#status-line").classList.toggle("error", data.status === "failed");
    if (data.log && data.log.length) {
      $("#log").hidden = false;
      $("#log").textContent = data.log.join("\n");
    }
    if (data.status === "complete") {
      // Triptych: source + ref + output (Task 3)
      const comp = document.createElement("div");
      comp.className = "comparison";
      for (const o of data.outputs) {
        const wrap = document.createElement("figure");
        const cap = o.label || o.name;
        wrap.innerHTML = `<img src="${o.url}"><figcaption>${cap}</figcaption>`;
        comp.appendChild(wrap);
      }
      $("#outputs").appendChild(comp);

      // Inline playback of the final video
      const videoOut = data.outputs.find(o => o.name.endsWith(".mp4"));
      if (videoOut) {
        const v = document.createElement("video");
        v.src = videoOut.url;
        v.controls = true;
        v.autoplay = false;
        v.style.marginTop = "16px";
        $("#outputs").appendChild(v);
      }
      goBtn.disabled = false; goBtn.textContent = "Generate ✨";
      return;
    }
    if (data.status === "failed") {
      goBtn.disabled = false; goBtn.textContent = "Generate ✨";
      return;
    }
    setTimeout(poll, 2000);
  };
  poll();
});