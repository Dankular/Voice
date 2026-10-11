import { decodeToMono, resample, trimEdges, capLength, wavBlob } from "./audio.js";

const SAMPLE_RATE = 24000;
const REF_MAX_S = 30;   // the whole sample is used; 30 s is Whisper's window, so longer audio could not be transcribed in full
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const END = new Set(";:,.!?…)]}\"'“”‘’；：，。！？、……）】");
const addPunctuation = (t) => { t = t.trim(); return t && !END.has(t[t.length - 1]) ? t + (/[一-鿿]/.test(t) ? "。" : ".") : t; };

// Gapless playback of chunks as they are generated (Web Audio), plus a concatenated WAV at the end.
class Player {
  constructor() { this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE }); this.t = 0; this.parts = []; this.nodes = []; this.stopped = false; }
  push(f32) {
    const b = this.ctx.createBuffer(1, f32.length, SAMPLE_RATE); b.copyToChannel(f32, 0);
    const n = this.ctx.createBufferSource(); n.buffer = b; n.connect(this.ctx.destination);
    const at = Math.max(this.ctx.currentTime + 0.05, this.t); n.start(at); this.t = at + b.duration;
    this.parts.push(f32); this.nodes.push(n);
  }
  stop() { this.stopped = true; this.nodes.forEach((n) => { try { n.stop(); } catch {} }); this.ctx.close(); }
  done() { return new Promise((res) => { const w = () => (this.stopped || this.ctx.currentTime >= this.t - 0.02 ? res() : setTimeout(w, 50)); w(); }); }   // resolves once everything pushed has been played (or stop())
  wav() { const all = new Float32Array(this.parts.reduce((a, p) => a + p.length, 0)); let o = 0; for (const p of this.parts) { all.set(p, o); o += p.length; } return wavBlob(all, SAMPLE_RATE); }
}

// sentence-sized chunks; the first one may be short so audio starts sooner
function chunkText(text) {
  const parts = text.split(/(?<=[.!?。！？;；])\s*/).map((x) => x.trim()).filter(Boolean);
  const out = []; let cur = "";
  for (const p of parts) { cur = cur ? cur + " " + p : p; if (cur.length >= (out.length ? 60 : 25)) { out.push(cur); cur = ""; } }
  if (cur) { if (out.length && cur.length < 25) out[out.length - 1] += " " + cur; else out.push(cur); }
  return out.length ? out : [text];
}

let kvMod = null, engineName = "", omni = null, transcribe = null, picked = null, page = 0, last = new FormData();
const refs = new Map();   // voice_id -> prepared reference (transcribed + encoded once)

// ---------- models: load at page load (a full-screen loading page covers the UI until they are ready) ----------
const EXPECTED_BYTES = 1190e6;       // ~LM 309 + text emb 155 + audio emb 34 + codec 480 + Whisper ~210 MB; progress is measured against this
const bars = {}, gotBytes = {};
function status(label, got, total, note) {
  let row = bars[label];
  if (!row) { row = bars[label] = document.createElement("div"); $("loadrows").appendChild(row); }
  const pct = total ? Math.round(100 * got / total) : 0;
  row.textContent = `${label}: ${note || (total ? pct + "%" : "loading…")}`;
  if (total) gotBytes[label] = got;
  const done = Object.values(gotBytes).reduce((a, b) => a + b, 0);
  $("loadbar").firstElementChild.style.width = Math.min(99, Math.round(100 * done / EXPECTED_BYTES)) + "%";
  $("loadstage").textContent = note ? `${label} — ${note}` : (/warm/i.test(label) ? "Compiling GPU kernels…" : `Downloading ${label}…`);
}
function loadingDone() {
  $("loadbar").firstElementChild.style.width = "100%";
  $("loading").classList.add("done");
  setTimeout(() => $("loading").remove(), 400);
}
$("loaddismiss").onclick = () => $("loading").remove();

async function loadModels() {
  if (!("gpu" in navigator)) status("WebGPU", 0, 0, "not available in this browser — falling back to wasm (very slow)");
  try {
    // dynamic imports: the voice list and filters keep working even if the model libraries fail to load
    const { loadAsr } = await import("./asr.js");
    const asrP = loadAsr({ onProgress: (p) => p.status === "progress" && status("Whisper " + (p.file || ""), p.loaded, p.total) });
    const kvBase = new URLSearchParams(location.search).get("kv") || undefined;
    try {                                        // fast engine: cached prefix + 4-bit LM (needs the files from export/build_all.sh)
      kvMod = await import("./load_kv.js");
      omni = await kvMod.loadKV({ kvBase, onStatus: status });
      if (kvMod.PROFILE) kvMod.profileRows.length = 0;
      engineName = omni.fused ? "cached-prefix, fused 4-bit" : "cached-prefix 4-bit";
    } catch (e) {                                // fall back to the slower full-recompute engine
      console.warn("fast engine unavailable, falling back:", e);
      $("err").textContent = "Fast engine files not found — using the slower fallback engine. (" + (e.message || e) + ")";
      omni = await (await import("./omnivoice.js")).OmniVoice.load({ onStatus: status });
      engineName = "full-recompute int8";
    }
    $("loadstage").textContent = "Preparing speech recognition…";
    transcribe = await asrP;
    $("models").textContent = `Models ready — ${engineName} on ${omni.providers[0]}.`;
    updateGo(); prefetch(); loadingDone();
  } catch (e) {
    $("err").textContent = "Model load failed: " + (e.message || e); console.error(e);
    $("loaderr").textContent = "Model load failed: " + (e.message || e); $("loadstage").textContent = "Could not load the models."; $("loaddismiss").hidden = false;
  }
}

// ---------- voice list + filters built from the voices' own tags ----------
const NICE = { gender: "Gender", age: "Age", accent: "Accent", descriptive: "Style", use_case: "Use case", language: "Language",
  category: "Type" };
const HIDE = new Set(["voice_id", "name", "description", "preview_url", "public_owner_id", "cloned_by_count", "usage_character_count_1y",
  "usage_character_count_7d", "play_api_usage_character_count_1y", "rate", "notice_period", "created_date", "date_unix", "image_url",
  "free_users_allowed", "live_moderation_enabled", "featured", "is_added_by_user", "financial_rewards_enabled", "verified_languages"]);
const pretty = (v) => { const t = String(v).replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };

let allVoices = [];
const active = new Map();       // tag -> Set(selected values); OR within a tag, AND across tags

function facetFields() {         // simple string tags with a usable number of distinct values
  const vals = new Map();
  for (const v of allVoices) for (const [k, x] of Object.entries(v)) {
    if (HIDE.has(k) || typeof x !== "string" || !x || x.length > 40) continue;
    if (!vals.has(k)) vals.set(k, new Set());
    vals.get(k).add(x);
  }
  return [...vals].filter(([, s]) => s.size >= 2 && s.size <= 60).map(([k]) => k)
    .sort((a, b) => (Object.keys(NICE).indexOf(a) + 1 || 99) - (Object.keys(NICE).indexOf(b) + 1 || 99));
}

const matches = (v, skip) => [...active].every(([k, set]) => k === skip || !set.size || set.has(v[k]));

const MAX_FACETS = 6;            // a few compact dropdowns, so the filters never take more than a row or two
function renderFacets() {
  const box = $("facets"); box.innerHTML = "";
  for (const k of facetFields().slice(0, MAX_FACETS)) {
    const counts = new Map();
    for (const v of allVoices) if (v[k] && matches(v, k)) counts.set(v[k], (counts.get(v[k]) || 0) + 1);   // counts respect the other filters
    const all = [...new Set(allVoices.map((v) => v[k]).filter(Boolean))].sort((a, b) => (counts.get(b) || 0) - (counts.get(a) || 0));
    const sel = document.createElement("select"); sel.setAttribute("aria-label", NICE[k] || pretty(k));
    sel.innerHTML = `<option value="">${esc(NICE[k] || pretty(k))}: any</option>` +
      all.map((val) => `<option value="${esc(val)}"${counts.get(val) ? "" : " disabled"}>${esc(pretty(val))} (${counts.get(val) || 0})</option>`).join("");
    sel.value = [...(active.get(k) || [])][0] || "";
    sel.onchange = () => { if (sel.value) active.set(k, new Set([sel.value])); else active.delete(k); listPage = 0; renderAll(); };
    box.appendChild(sel);
  }
}

// one shared preview player (instead of an <audio> element per row)
const preview = new Audio(); let previewing = null;
preview.onended = preview.onpause = () => { previewing = null; document.querySelectorAll(".voice .play").forEach((b) => (b.textContent = "▶")); };
function togglePreview(v, btn) {
  const same = previewing === v.voice_id; preview.pause();
  if (same) return;
  preview.src = `/api/preview/${encodeURIComponent(v.voice_id)}`; previewing = v.voice_id;
  preview.play().then(() => { btn.textContent = "■"; }).catch(() => { previewing = null; });
}

// The list never scrolls: it shows as many rows as fit, and the pager steps through the rest.
const ROW_H = 54, ROW_GAP = 6;
let listPage = 0, hasMore = false;
const pageSize = () => Math.max(3, Math.floor(($("list").clientHeight + ROW_GAP) / (ROW_H + ROW_GAP)));

function renderList() {
  const shown = allVoices.filter((v) => matches(v)), size = pageSize();
  const pages = Math.max(1, Math.ceil(shown.length / size));
  listPage = Math.min(listPage, pages - 1);
  const from = listPage * size, view = shown.slice(from, from + size);
  $("list").innerHTML = "";
  for (const v of view) {
    const el = document.createElement("div"); el.className = "voice" + (picked?.voice_id === v.voice_id ? " sel" : "");
    const meta = ["gender", "age", "accent", "language"].map((k) => v[k]).filter(Boolean).map(pretty).join(" · ");
    el.innerHTML = `<button type="button" class="play" aria-label="Preview ${esc(v.name)}">${previewing === v.voice_id ? "■" : "▶"}</button>
      <div class="info"><div class="n">${esc(v.name)}</div>
      <div class="d">${esc(meta)}${meta && v.description ? " — " : ""}${esc(v.description)}</div></div>`;
    el.querySelector(".play").onclick = (e) => { e.stopPropagation(); togglePreview(v, e.currentTarget); };
    el.onclick = () => {
      document.querySelectorAll(".voice.sel").forEach((x) => x.classList.remove("sel"));
      el.classList.add("sel"); picked = v; $("picked").textContent = v.name; updateGo(); $("err").textContent = ""; prefetch();
    };
    $("list").appendChild(el);
  }
  if (allVoices.length && !shown.length) $("list").innerHTML = '<div class="listmsg">No voices match these filters.</div>';
  const n = [...active.values()].reduce((a, s) => a + s.size, 0);
  $("count").textContent = shown.length ? `${from + 1}–${from + view.length} of ${shown.length}${hasMore ? "+" : ""}` : "";
  $("prev").disabled = listPage === 0;
  $("next").disabled = listPage >= pages - 1 && !hasMore;
  $("clear").hidden = !n;
}
const renderAll = () => { renderFacets(); renderList(); };
$("clear").onclick = () => { active.clear(); listPage = 0; renderAll(); };
$("prev").onclick = () => { listPage = Math.max(0, listPage - 1); renderList(); };
$("next").onclick = async () => {
  const size = pageSize(), shown = allVoices.filter((v) => matches(v)).length;
  if ((listPage + 1) * size >= shown && hasMore) { page++; await loadVoices(false); }   // reached the end of what is loaded: fetch more
  if ((listPage + 1) * size < allVoices.filter((v) => matches(v)).length) listPage++;
  renderList();
};
new ResizeObserver(() => allVoices.length && renderList()).observe($("list"));

async function loadVoices(reset) {
  if (reset) { page = 0; listPage = 0; hasMore = false; allVoices = []; active.clear(); $("list").innerHTML = ""; $("facets").innerHTML = ""; }
  const q = new URLSearchParams([...last].filter(([, v]) => v)); q.set("page", page);
  try {
    const r = await fetch("/api/voices?" + q);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const { voices, has_more } = await r.json();
    const seen = new Set(allVoices.map((v) => v.voice_id));
    allVoices.push(...voices.filter((v) => !seen.has(v.voice_id)));
    hasMore = !!has_more;
    if (reset && !voices.length) $("list").innerHTML = '<div class="listmsg">No voices found.</div>';
    else renderAll();
  } catch (e) { $("err").textContent = String(e.message || e); }
}
$("filters").onsubmit = (e) => { e.preventDefault(); last = new FormData($("filters")); loadVoices(true); };

// ---------- pick -> transcribe on the fly -> clone ----------
// Preparation (fetch sample -> transcribe -> codec-encode) does not depend on the text, so it starts as soon as a voice is
// selected, runs ASR and the encoder concurrently, and is remembered across visits (IndexedDB; the codes are ~10 KB).
const SIG = "ct03-enc+whisper-base-v1";                       // bump when the codec encoder or ASR model changes
const inflight = new Map();
let prepRuns = 0;                                          // 0 = first real preparation in this page session
let lastPrep = "";

const idb = () => new Promise((res, rej) => { const r = indexedDB.open("omnivoice-refs", 1); r.onupgradeneeded = () => r.result.createObjectStore("refs"); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
async function cacheGet(key) { try { const db = await idb(); return await new Promise((res) => { const q = db.transaction("refs").objectStore("refs").get(key); q.onsuccess = () => res(q.result); q.onerror = () => res(null); }); } catch { return null; } }
async function cachePut(key, v) { try { const db = await idb(); db.transaction("refs", "readwrite").objectStore("refs").put(v, key); } catch { /* storage blocked: fine */ } }

async function doPrepare(voice) {
  const key = `${SIG}:${voice.voice_id}`, t = {};
  const saved = await cacheGet(key);
  if (saved) { lastPrep = "prepared from cache"; return { ...saved, codes: new Int32Array(saved.codes) }; }
  let t0 = performance.now();
  $("status").textContent = "Fetching voice sample…";
  const r = await fetch(`/api/preview/${encodeURIComponent(voice.voice_id)}`);
  if (!r.ok) throw new Error("could not fetch the voice sample");
  const bytes = new Uint8Array(await r.arrayBuffer());
  const x24 = trimEdges(capLength(await decodeToMono(bytes, SAMPLE_RATE), SAMPLE_RATE, REF_MAX_S), SAMPLE_RATE);
  if (x24.length < SAMPLE_RATE) throw new Error("voice sample is too short");
  const x16 = await resample(x24, SAMPLE_RATE, 16000);
  t.fetchDecode = performance.now() - t0;
  $("status").textContent = "Transcribing and encoding the sample…";
  const timed = async (name, f) => { const s0 = performance.now(); const v = await f(); t[name] = performance.now() - s0; return v; };
  const t1 = performance.now();
  const [text, enc] = await Promise.all([timed("asr", () => transcribe(x16)), timed("encode", () => omni.encodeReference(x24))]);   // independent: run concurrently
  t.both = performance.now() - t1;
  if (!text) throw new Error("could not transcribe the sample");
  const ref = { ...enc, text: addPunctuation(text) };
  lastPrep = `${prepRuns++ === 0 ? "[first preparation this session] " : "[later preparation] "}prepared in ${((performance.now() - t0) / 1000).toFixed(1)} s — fetch+decode ${t.fetchDecode.toFixed(0)} ms, Whisper ${t.asr.toFixed(0)} ms, codec encode ${t.encode.toFixed(0)} ms (ran together: ${t.both.toFixed(0)} ms)`;
  cachePut(key, { ...ref, codes: ref.codes.buffer.slice(0) });
  return ref;
}

function prepare(voice) {
  const key = voice.voice_id;
  if (refs.has(key)) return Promise.resolve(refs.get(key));
  if (!inflight.has(key)) {
    inflight.set(key, doPrepare(voice).then((ref) => { refs.set(key, ref); return ref; }).finally(() => inflight.delete(key)));
  }
  return inflight.get(key);
}
// start preparing as soon as a voice is picked (or as soon as the models finish loading)
function prefetch() {
  if (!(omni && transcribe && picked)) return;
  const v = picked;
  prepare(v).then(() => { if (picked === v) $("status").textContent = "Voice ready."; $("perf").textContent = lastPrep; if (kvMod?.PROFILE) showProfile("codec encoder"); })
            .catch((e) => { if (picked === v) $("err").textContent = String(e.message || e); });
}

function updateGo() { $("go").disabled = !(omni && transcribe && picked); }

// ?profile=1: where does the GPU time go? Aggregates the per-kernel timings collected since the last call.
function showProfile(label = "synthesis") {
  const rows = kvMod.profileRows.splice(0), by = new Map();
  let total = 0;
  for (const r of rows) { const ms = (r.endTime - r.startTime) / 1e6, k = r.kernelType; total += ms; const e = by.get(k) || { n: 0, ms: 0 }; e.n++; e.ms += ms; by.set(k, e); }
  const top = [...by].sort((a, b) => b[1].ms - a[1].ms).slice(0, 12).map(([k, e]) => `${k.padEnd(34)} ${String(e.n).padStart(6)} kernels  ${e.ms.toFixed(1).padStart(8)} ms`);
  $("prof").textContent = `[${label}] GPU kernel time ${total.toFixed(0)} ms over ${rows.length} kernels\n` + top.join("\n");
}

let player = null, cancelled = false;
$("stop").onclick = () => { cancelled = true; player?.stop(); };

$("go").onclick = async () => {
  const text = $("text").value.trim();
  if (!text || !picked) return;
  $("go").disabled = true; $("stop").hidden = false; $("err").textContent = ""; $("perf").textContent = ""; $("dl").hidden = true;
  cancelled = false; player?.stop(); player = new Player();
  try {
    const ref = await prepare(picked);
    const opts = { guidance: $("cfg").checked ? 2.0 : 0 };   // steps: engine default
    const pieces = chunkText(text), lines = [];
    for (let i = 0; i < pieces.length && !cancelled; i++) {
      const t = performance.now();
      const wav = await omni.synthesize(pieces[i], ref, { ...opts, onStep: (st) => { $("status").textContent = `Chunk ${i + 1}/${pieces.length} — step ${st.step}/${st.of}`; } });
      if (cancelled) break;
      player.push(wav);
      if (kvMod?.PROFILE) showProfile();
      const secs = (performance.now() - t) / 1000, audio = wav.length / SAMPLE_RATE, st = wav.timing.steps;
      const first = st[0], rest = st.slice(1);
      const avg = (k) => (rest.reduce((a, x) => a + x[k], 0) / Math.max(1, rest.length)).toFixed(0);
      lines.push(`chunk ${i + 1}: ${secs.toFixed(1)} s for ${audio.toFixed(1)} s audio (RTF ${(secs / audio).toFixed(2)}; <1 = faster than real time) · ` +
        `step 1: ${first.lmMs.toFixed(0)}+${first.postMs.toFixed(0)} ms, later steps avg LM ${avg("lmMs")} + post ${avg("postMs")} + js ${avg("jsMs")} ms · S=${wav.timing.shapes.full ?? wav.timing.shapes.cond}`);
      $("perf").textContent = lines[lines.length - 1];
    }
    if (!cancelled) {
      $("dl").href = URL.createObjectURL(player.wav()); $("dl").hidden = false;
      $("status").textContent = `Sample transcript: “${ref.text}”`;
    }
  } catch (e) { $("err").textContent = String(e.message || e); $("status").textContent = ""; console.error(e); }
  $("stop").hidden = true; updateGo();
};

// ---------- talk: microphone -> VAD -> Whisper -> local LLM -> OmniVoice, with barge-in ----------
// FireRedVAD (listen.js) finds the user's utterances; agent.js runs the turn loop. Every utterance is also announced as a
// `user-utterance` event on document ({ audio: Float32Array 16 kHz mono, durationMs }), and `user-speech-start` when it begins.
let listener = null, agent = null, llmP = null;
const chat = [];                                            // last few messages shown in the conversation view (no scrolling: older ones fall off)
function renderChat() {
  $("chat").innerHTML = chat.filter((m) => m.text).slice(-6).map((m) => `<div class="msg ${m.who}">${esc(m.text)}</div>`).join("");
}
function loadLlmOnce() {
  return (llmP ??= (async () => {
    const [tf, { loadLlm, DEFAULT_LLM }] = await Promise.all([import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1"), import("./llm.js")]);
    const q = new URLSearchParams(location.search), gpu = "gpu" in navigator;
    const got = {}, onProgress = (p) => {
      if (p.status !== "progress" || !p.total) return;
      got[p.file] = [p.loaded, p.total]; const v = Object.values(got), l = v.reduce((a, x) => a + x[0], 0), t = v.reduce((a, x) => a + x[1], 0);
      $("status").textContent = `Downloading the language model… ${Math.round((100 * l) / t)}%`;
    };
    $("status").textContent = "Loading the language model…";
    return loadLlm(tf, { model: q.get("llm") || DEFAULT_LLM.model, dtype: q.get("llmdtype") || (gpu ? DEFAULT_LLM.dtype : "q4"), device: gpu ? "webgpu" : "wasm", onProgress });
  })().catch((e) => { llmP = null; throw e; }));
}

// Speaks sentences with OmniVoice as they arrive and plays them gaplessly; stop() cuts the audio immediately (barge-in).
function makeSpeaker() {
  let pl = null, gen = 0;
  return {
    async speak(text) {
      const my = gen; pl ??= new Player(); const out = pl;
      const ref = await prepare(picked);
      for (const piece of chunkText(text)) {
        if (my !== gen) throw new Error("stopped");
        const wav = await omni.synthesize(piece, ref, { guidance: 0 });   // a chunk in flight cannot be aborted; its audio is dropped below
        if (my !== gen) throw new Error("stopped");
        out.push(wav);
      }
    },
    finish: () => (pl ? pl.done() : Promise.resolve()),
    stop() { gen++; pl?.stop(); pl = null; },
  };
}

function setTalkUi(on) {
  const btn = $("listen");
  btn.classList.toggle("on", on); btn.textContent = on ? "🎤 Stop talking" : "🎤 Talk";
  $("vadbar").hidden = !on; $("text").hidden = on; $("chat").hidden = !on; $("go").hidden = on; $("cfg-l").hidden = on;
}
const STATE_TEXT = { thinking: "Thinking…", speaking: "Speaking… (talk to interrupt)", idle: "Listening…" };

async function toggleTalk() {
  const btn = $("listen");
  if (listener?.running) { listener.stop(); agent?.interrupt(); setTalkUi(false); $("status").textContent = ""; return; }
  if (!(omni && transcribe)) { $("err").textContent = "The speech models are still loading."; return; }
  if (!picked) { $("err").textContent = "Pick a voice first."; return; }
  btn.disabled = true; $("err").textContent = "";
  try {
    $("status").textContent = "Loading voice activity model…";
    const [{ Listener, loadVad }, ort, { Agent }] = await Promise.all([import("./listen.js"), import("/vendor/ort/ort.webgpu.min.mjs"), import("./agent.js")]);
    ort.env.wasm.wasmPaths = "/vendor/ort/";
    const [vad, llm] = await Promise.all([loadVad(ort), loadLlmOnce()]);
    await prepare(picked);                                      // the voice must be ready before the first reply
    const bar = $("vadbar");
    if (!agent) {
      agent = new Agent({
        transcribe: (a) => transcribe(a), llm, tts: makeSpeaker(),
        on: {
          state: (st) => { $("status").textContent = STATE_TEXT[st] || ""; if (listener) listener.turn.o.startMs = st === "speaking" ? 300 : 120; },   // harder to trigger while we are the ones talking
          user: (t) => { chat.push({ who: "u", text: t }, { who: "a", text: "" }); renderChat(); },
          agent: (t) => { const m = chat.at(-1); if (m?.who === "a") { m.text = t; renderChat(); } },
          error: (e) => { $("err").textContent = "Agent error: " + (e.message || e); console.error(e); },
        },
      });
    }
    listener ??= new Listener(vad, {
      prob: (p) => { bar.firstElementChild.style.width = Math.round(p * 100) + "%"; bar.classList.toggle("hot", p >= 0.5); },
      start: () => { agent.interrupt(); $("status").textContent = "Hearing you…"; document.dispatchEvent(new CustomEvent("user-speech-start")); },   // barge-in: stop the agent as soon as the user starts
      end: (u) => { document.dispatchEvent(new CustomEvent("user-utterance", { detail: u })); agent.utterance(u.audio); },
      error: (e) => { $("err").textContent = "Listening failed: " + (e.message || e); console.error(e); },
    });
    await listener.start();
    setTalkUi(true); renderChat(); $("status").textContent = STATE_TEXT.idle;
  } catch (e) {
    $("err").textContent = (e.name === "NotAllowedError" ? "Microphone permission was denied." : "Could not start: " + (e.message || e)); $("status").textContent = ""; console.error(e);
  }
  btn.disabled = false;
}
$("listen").onclick = toggleTalk;

if (new URLSearchParams(location.search).get("profile")) document.body.classList.add("profile");
loadModels();
loadVoices(true);
