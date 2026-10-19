// Openjutsu frontend — minimal vanilla JS.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const refs = []; // File[]

// -- reference image upload + thumbnail grid -------------------------------
const thumbGrid = $("#ref-list");
const renderThumbs = () => {
  thumbGrid.innerHTML = "";
  refs.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const div = document.createElement("div");
    div.className = "thumb";
    div.innerHTML = `
      <img src="${url}" alt="${f.name}">
      <button class="x" type="button" aria-label="remove">×</button>
    `;
    div.querySelector(".x").addEventListener("click", () => {
      URL.revokeObjectURL(url);
      refs.splice(i, 1);
      renderThumbs();
    });
    thumbGrid.appendChild(div);
  });
};
$("#ref-add").addEventListener("change", (e) => {
  for (const f of e.target.files) {
    if (refs.length >= 9) break;
    refs.push(f);
  }
  e.target.value = "";
  renderThumbs();
});

// -- prompt toggle --------------------------------------------------------
const promptEl = $("#prompt");
$("#prompt-on").addEventListener("change", (e) => {
  promptEl.disabled = !e.target.checked;
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
  refs.forEach((f) => fd.append("ref_images", f));
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
      for (const o of data.outputs) {
        const v = document.createElement("video");
        v.src = o.url;
        v.controls = true;
        v.autoplay = false;
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