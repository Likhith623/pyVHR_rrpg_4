const form = document.getElementById("predict-form");
const videoInput = document.getElementById("video-input");
const videoMeta = document.getElementById("video-meta");
const submitBtn = document.getElementById("submit-btn");
const statusPill = document.getElementById("status-pill");
const uploadZone = document.getElementById("upload-zone");

const probabilityEl = document.getElementById("probability");
const labelEl = document.getElementById("label");
const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const modelEl = document.getElementById("model");
const debugEl = document.getElementById("debug");
const meterFg = document.getElementById("meter-fg");

const CIRCUMFERENCE = 2 * Math.PI * 50;
const API_BASE = window.location.port === "8000"
  ? window.location.origin
  : "http://127.0.0.1:8000";

function setStatus(text, type = "idle") {
  statusPill.textContent = text;
  const map = {
    idle: ["#f0f6ff", "#234f83", "#b6c9e5"],
    loading: ["#fff6e5", "#84560e", "#e8c88b"],
    ok: ["#e7f8ef", "#12653e", "#8fd2af"],
    error: ["#fdeeed", "#9f2e2b", "#edb3b0"],
  };
  const [bg, ink, border] = map[type] || map.idle;
  statusPill.style.background = bg;
  statusPill.style.color = ink;
  statusPill.style.borderColor = border;
}

function updateMeter(probability) {
  const p = Math.max(0, Math.min(1, Number(probability) || 0));
  meterFg.style.strokeDasharray = String(CIRCUMFERENCE);
  meterFg.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - p));
  meterFg.style.stroke = p >= 0.5 ? "#b73934" : "#1a7b62";
}

function updateResult(data) {
  probabilityEl.textContent = Number(data.probability).toFixed(3);
  labelEl.textContent = data.label;
  predictionEl.textContent = String(data.prediction);
  confidenceEl.textContent = `${Number(data.confidence).toFixed(2)}%`;
  modelEl.textContent = data.model_name;
  debugEl.textContent = JSON.stringify(data, null, 2);
  updateMeter(data.probability);
}

function updateVideoMeta(file) {
  if (!file) {
    videoMeta.textContent = "Supported: mp4, avi, mov, mkv, webm · max 500MB";
    return;
  }
  const sizeMb = file.size / (1024 * 1024);
  videoMeta.textContent = `${file.name} · ${sizeMb.toFixed(2)} MB`;
}

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  updateVideoMeta(file);
});

uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadZone.style.borderColor = "#1f67b4";
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.style.borderColor = "#87a5cc";
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadZone.style.borderColor = "#87a5cc";
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;

  const dt = new DataTransfer();
  dt.items.add(file);
  videoInput.files = dt.files;
  updateVideoMeta(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = videoInput.files?.[0];
  if (!file) {
    setStatus("Please upload a video", "error");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  submitBtn.disabled = true;
  submitBtn.textContent = "Analyzing...";
  setStatus("Running automatic ensemble inference...", "loading");

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      body: formData,
    });

    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || `HTTP ${res.status}`);
    }

    updateResult(payload);
    setStatus("Inference complete", "ok");
  } catch (error) {
    setStatus("Inference failed", "error");
    debugEl.textContent = String(error);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Video";
  }
});

setStatus("Upload a video to start", "idle");
updateMeter(0);
