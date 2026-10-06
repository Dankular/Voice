import { OmniVoice, SAMPLE_RATE } from "./omnivoice.js";
import { loadAsr } from "./asr.js";
import { decodeToMono, resample, trimEdges, capLength, wavBlob } from "./audio.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const END = new Set(";:,.!?…)]}\"'“”‘’；：，。！？、……）】");
const addPunctuation = (t) => { t = t.trim(); return t && !END.has(t[t.length - 1]) ? t + (/[一-鿿]/.test(t) ? "。" : ".") : t; };

let omni = null, transcribe = null, picked = null, page = 0, last = new FormData();
const refs = new Map();   // voice_id -> prepared reference (transcribed + encoded once)

// ---------- models: load at page load ----------
const bars = {};
function status(label, got, total, note) {
  let row = bars[label];
  if (!row) {
    row = bars[label] = document.createElement("div");
    row.className = "m"; $("models").appendChild(row);
  }
  const pct = total ? Math.round(100 * got / total) : 0;
  row.textContent = `${label}: ${note || (total ? pct + "%" : "loading…")}`;
}

async function loadModels() {
  if (!("gpu" in navigator)) status("WebGPU", 0, 0, "not available in this browser — falling back to wasm (very slow)");
  try {
    [omni, transcribe] = await Promise.all([
      OmniVoice.load({ onStatus: status }),
      loadAsr({ onProgress: (p) => p.status === "progress" && status("Whisper " + (p.file || ""), p.loaded, p.total) }),
    ]);
    $("models").textContent = `Models ready (${omni.providers[0]}).`;
    updateGo();
  } catch (e) { $("err").textContent = "Model load failed: " + (e.message || e); console.error(e); }
}

// ---------- voice list ----------
async function loadVoices(reset) {
  if (reset) { page = 0; $("list").innerHTML = ""; $("more").hidden = true; }
  const q = new URLSearchParams([...last].filter(([, v]) => v)); q.set("page", page);
  try {
    const r = await fetch("/api/voices?" + q);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const { voices, has_more } = await r.json();
    if (reset && !voices.length) $("list").textContent = "No voices found.";
    for (const v of voices) {
      const el = document.createElement("div"); el.className = "voice";
      const meta = [v.gender, v.accent, v.age, v.language].filter(Boolean).join(" · ");
      el.innerHTML = `<div class="info"><div class="n">${esc(v.name)}</div>
        <div class="d">${esc(meta)}${meta && v.description ? " — " : ""}${esc(v.description)}</div></div>
        <audio controls preload="none" src="/api/preview/${encodeURIComponent(v.voice_id)}" style="width:200px;margin:0"></audio>`;
      el.querySelector("audio").addEventListener("click", (e) => e.stopPropagation());
      el.onclick = () => {
        document.querySelectorAll(".voice.sel").forEach((x) => x.classList.remove("sel"));
        el.classList.add("sel"); picked = v; $("picked").textContent = v.name; updateGo();
      };
      $("list").appendChild(el);
    }
    $("more").hidden = !has_more;
  } catch (e) { $("err").textContent = String(e.message || e); }
}
$("filters").onsubmit = (e) => { e.preventDefault(); last = new FormData($("filters")); loadVoices(true); };
$("more").onclick = () => { page++; loadVoices(false); };

// ---------- pick -> transcribe on the fly -> clone ----------
async function prepare(voice) {
  if (refs.has(voice.voice_id)) return refs.get(voice.voice_id);
  $("status").textContent = "Fetching voice sample…";
  const r = await fetch(`/api/preview/${encodeURIComponent(voice.voice_id)}`);
  if (!r.ok) throw new Error("could not fetch the voice sample");
  const bytes = new Uint8Array(await r.arrayBuffer());
  let x24 = trimEdges(capLength(await decodeToMono(bytes, SAMPLE_RATE), SAMPLE_RATE, 20), SAMPLE_RATE);
  if (x24.length < SAMPLE_RATE) throw new Error("voice sample is too short");
  $("status").textContent = "Transcribing the sample…";
  const text = await transcribe(await resample(x24, SAMPLE_RATE, 16000));
  if (!text) throw new Error("could not transcribe the sample");
  $("status").textContent = "Encoding the sample…";
  const enc = await omni.encodeReference(x24);
  const ref = { ...enc, text: addPunctuation(text) };
  refs.set(voice.voice_id, ref);
  return ref;
}

function updateGo() { $("go").disabled = !(omni && transcribe && picked); }

$("go").onclick = async () => {
  const text = $("text").value.trim();
  if (!text || !picked) return;
  $("go").disabled = true; $("err").textContent = "";
  try {
    const ref = await prepare(picked);
    const wav = await omni.synthesize(text, ref, { onStep: (s, n) => { $("status").textContent = `Generating… step ${s}/${n}`; } });
    const a = $("audio"); a.src = URL.createObjectURL(wavBlob(wav, SAMPLE_RATE)); a.hidden = false; a.play();
    $("status").textContent = `Sample transcript: “${ref.text}”`;
  } catch (e) { $("err").textContent = String(e.message || e); $("status").textContent = ""; console.error(e); }
  updateGo();
};

loadModels();
loadVoices(true);
