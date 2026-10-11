// Microphone capture as 16 kHz mono float32 blocks. Echo cancellation / noise suppression are requested so the agent's own
// voice coming out of the speakers is less likely to be heard as the user (needed for barge-in).
const TARGET = 16000;

// Streaming resampler for browsers that will not run the AudioContext at 16 kHz: box-filter average (anti-aliasing) with fractional edges.
class Resampler {
  constructor(inRate) { this.r = inRate / TARGET; this.pos = 0; this.carry = new Float32Array(0); }
  push(x) {
    const s = new Float32Array(this.carry.length + x.length); s.set(this.carry); s.set(x, this.carry.length);
    const out = []; let p = this.pos;
    while (p + this.r <= s.length) {
      const a = p, b = p + this.r; let acc = 0;
      for (let i = Math.floor(a); i < Math.ceil(b); i++) acc += s[i] * (Math.min(b, i + 1) - Math.max(a, i));
      out.push(acc / this.r); p = b;
    }
    const keep = Math.floor(p); this.carry = s.slice(keep); this.pos = p - keep;
    return Float32Array.from(out);
  }
}

export async function startMic(onAudio, { echoCancellation = true, noiseSuppression = true, autoGainControl = true } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation, noiseSuppression, autoGainControl } });
  let ctx;
  try { ctx = new AudioContext({ sampleRate: TARGET }); } catch { ctx = new AudioContext(); }
  try {
    await ctx.audioWorklet.addModule(new URL("./mic-worklet.js", import.meta.url));
    const src = ctx.createMediaStreamSource(stream), node = new AudioWorkletNode(ctx, "mic-capture", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
    const rs = ctx.sampleRate === TARGET ? null : new Resampler(ctx.sampleRate);
    node.port.onmessage = (e) => { const x = rs ? rs.push(e.data) : e.data; if (x.length) onAudio(x); };
    const mute = ctx.createGain(); mute.gain.value = 0;               // keep the graph pulled without playing the mic back
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
    if (ctx.state === "suspended") await ctx.resume();
    return { sampleRate: ctx.sampleRate, stop() { node.port.onmessage = null; stream.getTracks().forEach((t) => t.stop()); ctx.close(); } };
  } catch (e) { stream.getTracks().forEach((t) => t.stop()); ctx.close(); throw e; }
}
