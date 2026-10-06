// Browser audio helpers: decode any format the browser supports, resample, trim, write WAV.
export async function decodeToMono(bytes, sampleRate) {
  // decodeAudioData resamples to the context rate
  const ctx = new OfflineAudioContext(1, 1, sampleRate);
  const buf = await ctx.decodeAudioData(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

export async function resample(x, from, to) {
  if (from === to) return x;
  const ctx = new OfflineAudioContext(1, Math.ceil(x.length * to / from), to);
  const b = ctx.createBuffer(1, x.length, from);
  b.copyToChannel(x, 0);
  const s = ctx.createBufferSource();
  s.buffer = b; s.connect(ctx.destination); s.start();
  return (await ctx.startRendering()).getChannelData(0);
}

// numpy-equivalent of upstream's silence-edge trim (-50 dBFS, keep 100 ms lead / 200 ms tail)
export function trimEdges(x, sr, leadMs = 100, trailMs = 200, threshDb = -50) {
  const hop = Math.floor(sr * 0.01), n = Math.floor(x.length / hop);
  let first = -1, last = -1;
  for (let f = 0; f < n; f++) {
    let e = 0;
    for (let i = f * hop; i < (f + 1) * hop; i++) e += x[i] * x[i];
    if (20 * Math.log10(Math.sqrt(e / hop) + 1e-12) > threshDb) { if (first < 0) first = f; last = f; }
  }
  if (first < 0) return x;
  const a = Math.max(0, first * hop - Math.floor(sr * leadMs / 1000));
  const b = Math.min(x.length, (last + 1) * hop + Math.floor(sr * trailMs / 1000));
  return x.subarray(a, b);
}

// if longer than maxS, cut at the quietest 50 ms frame between 60% and 100% of maxS
export function capLength(x, sr, maxS) {
  if (x.length <= maxS * sr) return x;
  const hop = Math.floor(sr * 0.05), lo = Math.floor(0.6 * maxS * sr / hop), hi = Math.floor(maxS * sr / hop);
  let best = lo, bestE = Infinity;
  for (let f = lo; f < hi; f++) {
    let e = 0;
    for (let i = f * hop; i < (f + 1) * hop; i++) e += x[i] * x[i];
    if (e < bestE) { bestE = e; best = f; }
  }
  return x.subarray(0, best * hop);
}

export function rms(x) { let e = 0; for (let i = 0; i < x.length; i++) e += x[i] * x[i]; return Math.sqrt(e / Math.max(1, x.length)); }

export function wavBlob(f32, sr) {
  const dv = new DataView(new ArrayBuffer(44 + f32.length * 2));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); dv.setUint32(4, 36 + f32.length * 2, true); w(8, "WAVEfmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  w(36, "data"); dv.setUint32(40, f32.length * 2, true);
  for (let i = 0; i < f32.length; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, f32[i])) * 32767, true);
  return new Blob([dv], { type: "audio/wav" });
}
