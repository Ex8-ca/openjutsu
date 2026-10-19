// Openjutsu frontend — minimal vanilla JS.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// refs is now {person?: File, scene?: File, outfit?: File}
const refs = {};

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
    const resp = await fetch("/api/enhance-prompt", {
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

  let resp;
  try {
    resp = await fetch("/api/generate", { method: "POST", body: fd });
  } catch (e) {
    $("#status-line").classList.add("error");
    $("#status-line").textContent = `network error: ${e}`;
    goBtn.disabled = false; goBtn.textContent = "Generate ✨";
    return;
  }

  if (!resp.ok) {
    const t = await resp.text();
    $("#status-line").classList.add("error");
    $("#status-line").textContent = `submit failed (${resp.status}): ${t}`;
    goBtn.disabled = false; goBtn.textContent = "Generate ✨";
    return;
  }

  const { job_id, status } = await resp.json();
  $("#status-line").textContent = `${status} (job ${job_id.slice(0, 8)})`;

  // Poll status
  const start = Date.now();
  const poll = async () => {
    let r;
    try { r = await fetch(`/api/jobs/${job_id}`); } catch { return setTimeout(poll, 4000); }
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