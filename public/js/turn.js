// Turns per-frame speech probabilities (10 ms each, from StreamingVad) into "user started / stopped speaking" events and
// captures the utterance audio (with a little pre-roll) for the speech recogniser.
// Frame i of the VAD covers samples [160 i, 160 i + 400); a frame is treated as "now" when it is reported.
const HOP = 160, SR = 16000;

export class TurnDetector {
  // opts (ms unless noted): threshold (probability), startMs (speech needed to open a turn), endMs (silence that closes it),
  // minSpeechMs (shorter turns are discarded as noise), preRollMs (audio kept before the detected start), maxMs (force-close).
  constructor(on = {}, opts = {}) {
    this.on = on; this.o = { threshold: 0.5, startMs: 120, endMs: 700, minSpeechMs: 250, preRollMs: 300, maxMs: 30000, ...opts };
    this.reset();
  }
  reset() { this.chunks = []; this.base = 0; this.seen = 0; this.frame = 0; this.run = 0; this.silence = 0; this.speaking = false; this.startSample = 0; this.speechFrames = 0; }

  // audio: the samples that were fed to the VAD in this call; probs: the probabilities it returned for them
  push(audio, probs) {
    this.chunks.push(audio); this.seen += audio.length;
    const { threshold, startMs, endMs, minSpeechMs, preRollMs, maxMs } = this.o, f = (ms) => Math.round(ms / 10);
    for (const p of probs) {
      const hit = p >= threshold, atSample = this.frame * HOP; this.frame++;
      this.on.prob?.(p);
      if (!this.speaking) {
        this.run = hit ? this.run + 1 : 0;
        if (this.run >= f(startMs)) {                                  // speech began `run` frames ago
          this.speaking = true; this.silence = 0; this.speechFrames = this.run;
          this.startSample = Math.max(0, atSample - (this.run - 1) * HOP - Math.round((preRollMs * SR) / 1000));
          this.on.start?.();
        }
      } else {
        if (hit) { this.silence = 0; this.speechFrames++; } else this.silence++;
        const lenMs = ((this.frame * HOP - this.startSample) / SR) * 1000;
        if (this.silence >= f(endMs) || lenMs >= maxMs) this.close(this.speechFrames * 10 >= minSpeechMs);
      }
    }
    if (!this.speaking) this.trim(Math.round((preRollMs * SR) / 1000) + HOP * (f(startMs) + 2) + 400);
  }

  // end the current turn (also callable from outside, e.g. when the mic stops)
  close(keep = true) {
    if (!this.speaking) return;
    const end = Math.min(this.seen, this.frame * HOP + 400 - this.silence * HOP);   // drop the trailing silence, keep the last frame's window
    const audio = this.slice(this.startSample, Math.max(end, this.startSample));
    this.speaking = false; this.run = 0; this.silence = 0;
    this.trim(0);
    if (keep) this.on.end?.({ audio, durationMs: (audio.length / SR) * 1000 }); else this.on.discard?.();
  }

  slice(from, to) {
    const out = new Float32Array(Math.max(0, to - from)); let pos = this.base;
    for (const c of this.chunks) { const a = Math.max(from, pos), b = Math.min(to, pos + c.length); if (b > a) out.set(c.subarray(a - pos, b - pos), a - from); pos += c.length; }
    return out;
  }
  trim(keepSamples) {                                                    // idle: only the pre-roll window is needed
    const drop = this.seen - keepSamples; let pos = this.base;
    while (this.chunks.length && pos + this.chunks[0].length <= drop) { pos += this.chunks[0].length; this.chunks.shift(); }
    this.base = pos;
  }
}
