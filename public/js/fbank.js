// Streaming Kaldi-style log-mel filterbank (80 bins, 25 ms window / 10 ms shift, 16 kHz), as used by FireRedVAD.
// Port of kaldi_native_fbank with the options FireRedVAD sets: snip_edges, dither 0, 80 bins; everything else at the library
// defaults (Povey window, pre-emphasis 0.97, DC removal, 512-point FFT, power spectrum, 20 Hz .. Nyquist, log with float-eps floor).
// Samples go in as float32 in [-1, 1]; they are scaled to int16 range because FireRedVAD feeds int16 samples to the extractor.
// Verified numerically against kaldi_native_fbank (see scripts/test_vad_fbank.mjs).

const SR = 16000, WIN = 400, HOP = 160, NFFT = 512, BINS = 80, LOW_HZ = 20, PREEMPH = 0.97, EPS = 1.1920928955078125e-7;

const mel = (f) => 1127 * Math.log(1 + f / 700);
const hann = new Float64Array(WIN);                       // "povey" window: hann ^ 0.85
for (let i = 0; i < WIN; i++) hann[i] = Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1)), 0.85);

// triangular filters in the mel domain; the Nyquist FFT bin is not used (as in Kaldi)
const banks = (() => {
  const nBins = NFFT / 2, binHz = SR / NFFT, lo = mel(LOW_HZ), hi = mel(SR / 2), d = (hi - lo) / (BINS + 1);
  return Array.from({ length: BINS }, (_, b) => {
    const left = lo + b * d, center = lo + (b + 1) * d, right = lo + (b + 2) * d, w = new Float64Array(nBins);
    let first = -1, last = -1;
    for (let i = 0; i < nBins; i++) {
      const m = mel(binHz * i);
      if (m > left && m < right) { w[i] = m <= center ? (m - left) / (center - left) : (right - m) / (right - center); if (first < 0) first = i; last = i; }
    }
    return { first, w: w.subarray(first, last + 1) };
  });
})();

// iterative radix-2 FFT on separate re/im arrays
const cos = new Float64Array(NFFT / 2), sin = new Float64Array(NFFT / 2), rev = new Uint16Array(NFFT);
for (let i = 0; i < NFFT / 2; i++) { cos[i] = Math.cos((2 * Math.PI * i) / NFFT); sin[i] = -Math.sin((2 * Math.PI * i) / NFFT); }
for (let i = 0, bits = Math.log2(NFFT); i < NFFT; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
function fft(re, im) {
  for (let i = 0; i < NFFT; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= NFFT; size <<= 1) {
    const half = size >> 1, step = NFFT / size;
    for (let s = 0; s < NFFT; s += size) for (let k = 0; k < half; k++) {
      const a = s + k, b = a + half, wr = cos[k * step], wi = sin[k * step];
      const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}

export class Fbank {
  constructor() { this.buf = new Float64Array(0); this.re = new Float64Array(NFFT); this.im = new Float64Array(NFFT); }

  // Append samples (float32 in [-1,1]); returns the new complete frames as Float32Array(80)[].
  push(samples) {
    const old = this.buf.length, buf = new Float64Array(old + samples.length);
    buf.set(this.buf); for (let i = 0; i < samples.length; i++) buf[old + i] = samples[i] * 32768;
    const out = []; let start = 0;
    while (start + WIN <= buf.length) { out.push(this.frame(buf, start)); start += HOP; }
    this.buf = buf.slice(start);                           // keep only what the next frame still needs
    return out;
  }
  reset() { this.buf = new Float64Array(0); }

  frame(buf, start) {
    const { re, im } = this; let mean = 0;
    for (let i = 0; i < WIN; i++) mean += buf[start + i]; mean /= WIN;
    const x = new Float64Array(WIN); for (let i = 0; i < WIN; i++) x[i] = buf[start + i] - mean;
    for (let i = WIN - 1; i > 0; i--) x[i] -= PREEMPH * x[i - 1];
    x[0] -= PREEMPH * x[0];
    re.fill(0); im.fill(0); for (let i = 0; i < WIN; i++) re[i] = x[i] * hann[i];
    fft(re, im);
    const out = new Float32Array(BINS);
    for (let b = 0; b < BINS; b++) {
      const { first, w } = banks[b]; let e = 0;
      for (let k = 0; k < w.length; k++) { const i = first + k; e += w[k] * (re[i] * re[i] + im[i] * im[i]); }
      out[b] = Math.log(Math.max(e, EPS));
    }
    return out;
  }
}
