/* ══════════════════════════════════════════════════════
   NeuroPulse · app.js
   Neural canvas animation + Inference logic
══════════════════════════════════════════════════════ */

/* ────────────────────────────────────────────
   1.  NEURAL CANVAS BACKGROUND ANIMATION
──────────────────────────────────────────── */
(function initNeuralCanvas() {
  const canvas = document.getElementById("neural-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const COLORS = ["#00c8ff", "#00ff9d", "#4ab5ff", "#c77dff", "#ffb347"];
  const NODE_COUNT = 55;
  const MAX_DIST = 180;

  let W, H, nodes;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeNode() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 2.2 + 0.8,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      pulse: Math.random() * Math.PI * 2,
    };
  }

  function init() {
    resize();
    nodes = Array.from({ length: NODE_COUNT }, makeNode);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Update
    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      n.pulse += 0.025;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    }

    // Draw edges
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > MAX_DIST) continue;

        const alpha = (1 - dist / MAX_DIST) * 0.18;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(0,200,255,${alpha})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    // Draw nodes
    for (const n of nodes) {
      const pAlpha = 0.5 + 0.5 * Math.sin(n.pulse);
      // Outer glow
      const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.r * 6);
      grd.addColorStop(0, n.color + "40");
      grd.addColorStop(1, n.color + "00");
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r * 6, 0, Math.PI * 2);
      ctx.fillStyle = grd;
      ctx.fill();

      // Core dot
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color;
      ctx.globalAlpha = 0.5 + 0.5 * pAlpha;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", () => {
    resize();
    nodes = Array.from({ length: NODE_COUNT }, makeNode);
  });

  init();
  draw();
})();


/* ────────────────────────────────────────────
   2.  DOM ELEMENT REFS
──────────────────────────────────────────── */
const form        = document.getElementById("predict-form");
const videoInput  = document.getElementById("video-input");
const videoMeta   = document.getElementById("video-meta");
const dzTitle     = document.getElementById("dz-title");
const dzIcon      = document.getElementById("dz-icon");
const submitBtn   = document.getElementById("submit-btn");
const btnText     = document.getElementById("btn-text");
const statusPill  = document.getElementById("status-pill");
const uploadZone  = document.getElementById("upload-zone");

// Result elements
const probabilityEl = document.getElementById("probability");
const labelEl       = document.getElementById("label");
const predictionEl  = document.getElementById("prediction");
const confidenceEl  = document.getElementById("confidence");
const modelEl       = document.getElementById("model");
const debugEl       = document.getElementById("debug");
const ringArc       = document.getElementById("ring-arc");

// Stream bars
const barRppg  = document.getElementById("bar-rppg");
const barEff   = document.getElementById("bar-eff");
const barXcep  = document.getElementById("bar-xcep");
const barSwin  = document.getElementById("bar-swin");
const valRppg  = document.getElementById("val-rppg");
const valEff   = document.getElementById("val-eff");
const valXcep  = document.getElementById("val-xcep");
const valSwin  = document.getElementById("val-swin");

/* ────────────────────────────────────────────
   3.  CONFIG
──────────────────────────────────────────── */
const RING_CIRCUMFERENCE = 2 * Math.PI * 62; // r=62 → ≈389.56

const API_BASE =
  window.location.port === "8000"
    ? window.location.origin
    : "http://127.0.0.1:8000";

/* ────────────────────────────────────────────
   4.  STATUS PILL
──────────────────────────────────────────── */
function setStatus(text, type = "idle") {
  statusPill.textContent = text;
  statusPill.className = `pill pill-${type}`;
}

/* ────────────────────────────────────────────
   5.  RING METER
──────────────────────────────────────────── */
function updateRing(probability) {
  const p = Math.max(0, Math.min(1, Number(probability) || 0));
  const offset = RING_CIRCUMFERENCE * (1 - p);
  ringArc.style.strokeDashoffset = String(offset);

  // Color transitions: green → amber → red
  let color;
  if (p < 0.4) {
    color = "#00ff9d";  // safe green
  } else if (p < 0.65) {
    color = "#ffb347";  // warning amber
  } else {
    color = "#ff5c5c";  // danger red
  }
  ringArc.style.stroke = color;
}

/* ────────────────────────────────────────────
   6.  STREAM BARS
──────────────────────────────────────────── */
function updateStreamBars(data) {
  const streams = [
    { bar: barRppg,  val: valRppg,  key: "P_rPPG" },
    { bar: barEff,   val: valEff,   key: "P_efficientnet" },
    { bar: barXcep,  val: valXcep,  key: "P_xception" },
    { bar: barSwin,  val: valSwin,  key: "P_swin" },
  ];

  for (const s of streams) {
    const raw = data[s.key];
    if (raw !== undefined && raw !== null) {
      const p = Math.max(0, Math.min(1, Number(raw)));
      s.bar.style.width = `${(p * 100).toFixed(1)}%`;
      s.val.textContent = p.toFixed(3);
    } else {
      // If individual stream probs not returned, show ensemble spread
      s.bar.style.width = "0%";
      s.val.textContent = "—";
    }
  }
}

/* ────────────────────────────────────────────
   7.  UPDATE RESULT PANEL
──────────────────────────────────────────── */
function updateResult(data) {
  const p = Number(data.probability);

  // Ring & probability
  probabilityEl.textContent = p.toFixed(3);
  updateRing(p);

  // KV fields
  labelEl.textContent      = data.label      ?? "—";
  predictionEl.textContent = data.prediction != null ? String(data.prediction) : "—";
  confidenceEl.textContent = data.confidence != null
    ? `${Number(data.confidence).toFixed(2)}%`
    : "—";
  modelEl.textContent = data.model_name ?? "—";

  // Stream bars
  updateStreamBars(data);

  // Debug JSON
  debugEl.textContent = JSON.stringify(data, null, 2);

  // Color-code probability display
  probabilityEl.style.color =
    p >= 0.65 ? "var(--red)" :
    p >= 0.4  ? "var(--amber)" :
    "var(--green)";
}

/* ────────────────────────────────────────────
   8.  RESET RESULT PANEL
──────────────────────────────────────────── */
function resetResult() {
  probabilityEl.textContent = "—";
  probabilityEl.style.color = "var(--text-bright)";
  labelEl.textContent      = "—";
  predictionEl.textContent = "—";
  confidenceEl.textContent = "—";
  modelEl.textContent      = "—";
  debugEl.textContent      = "{}";
  updateRing(0);
  for (const bar of [barRppg, barEff, barXcep, barSwin]) bar.style.width = "0%";
  for (const v of [valRppg, valEff, valXcep, valSwin]) v.textContent = "—";
}

/* ────────────────────────────────────────────
   9.  VIDEO FILE METADATA
──────────────────────────────────────────── */
function updateFileMeta(file) {
  if (!file) {
    dzTitle.textContent = "Drop video here or click to browse";
    videoMeta.textContent = "mp4 · avi · mov · mkv · webm · max 500MB";
    submitBtn.disabled = true;
    btnText.textContent = "Select a Video First";
    return;
  }

  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  const ext    = file.name.split(".").pop().toUpperCase();

  dzTitle.textContent    = `✔ ${file.name}`;
  videoMeta.textContent  = `${ext} · ${sizeMB} MB`;
  submitBtn.disabled     = false;
  btnText.textContent    = "Analyze Video";
  setStatus("Video ready · Click Analyze to run ensemble", "idle");
}

/* ────────────────────────────────────────────
   10.  EVENT LISTENERS — FILE INPUT
──────────────────────────────────────────── */
videoInput.addEventListener("change", () => {
  updateFileMeta(videoInput.files?.[0] ?? null);
});

/* Drag and drop */
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  videoInput.files = dt.files;
  updateFileMeta(file);
});

/* ────────────────────────────────────────────
   11.  FORM SUBMIT — INFERENCE
──────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = videoInput.files?.[0];
  if (!file) {
    setStatus("Please upload a video file", "error");
    return;
  }

  // Validate size (500 MB)
  if (file.size > 500 * 1024 * 1024) {
    setStatus("File exceeds 500 MB limit", "error");
    return;
  }

  // UI: start loading
  submitBtn.disabled    = true;
  btnText.textContent   = "Running Ensemble Inference…";
  resetResult();
  setStatus("Extracting features across 4 streams…", "loading");

  const formData = new FormData();
  formData.append("video", file);

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      body: formData,
    });

    const payload = await res.json();

    if (!res.ok) {
      const detail = payload.detail || `HTTP ${res.status}`;
      throw new Error(detail);
    }

    updateResult(payload);
    setStatus("Ensemble inference complete", "ok");

  } catch (err) {
    setStatus(`Inference failed: ${err.message}`, "error");
    debugEl.textContent = String(err);
    console.error("[NeuroPulse] Inference error:", err);
  } finally {
    submitBtn.disabled  = false;
    btnText.textContent = "Analyze Video";
  }
});

/* ────────────────────────────────────────────
   12.  SCROLL REVEAL
──────────────────────────────────────────── */
(function initScrollReveal() {
  const revealTargets = document.querySelectorAll(
    ".glass-card, .mcard, .stream, .ensemble-node, .callout-box, .pipe-row"
  );

  if (!("IntersectionObserver" in window)) {
    revealTargets.forEach(el => {
      el.style.opacity = "1";
      el.style.transform = "none";
    });
    return;
  }

  // Set initial state
  revealTargets.forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(16px)";
    el.style.transition = "opacity 600ms ease, transform 600ms ease";
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "none";
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );

  revealTargets.forEach(el => observer.observe(el));
})();

/* ────────────────────────────────────────────
   13.  INIT
──────────────────────────────────────────── */
setStatus("Upload a video to start", "idle");
updateRing(0);
submitBtn.disabled  = true;
btnText.textContent = "Select a Video First";