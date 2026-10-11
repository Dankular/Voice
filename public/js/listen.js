// Microphone -> FireRedVAD -> turn detector. Calls: on.prob(p) per 10 ms, on.start() when the user starts speaking,
// on.end({ audio, durationMs }) with the utterance (16 kHz mono float32) when they stop, on.discard() for blips too short to count.
import { FireRedVAD, StreamingVad } from "./firered_vad.js";
import { TurnDetector } from "./turn.js";
import { startMic } from "./mic.js";

let vadP = null;
export function loadVad(ort, base = "/models/vad") { return (vadP ??= FireRedVAD.fetch(ort, base).catch((e) => { vadP = null; throw e; })); }

export class Listener {
  constructor(vad, on = {}, turnOpts = {}) {
    this.stream = new StreamingVad(vad); this.on = on;
    this.turn = new TurnDetector(on, turnOpts); this.mic = null; this.chain = Promise.resolve();
  }
  get running() { return !!this.mic; }
  async start() {
    if (this.mic) return;
    this.stream.reset(); this.turn.reset();
    // blocks arrive faster than the model runs at times; StreamingVad serialises, this keeps audio and probabilities paired in order
    this.mic = await startMic((a) => { this.chain = this.chain.then(async () => { if (!this.mic) return; this.turn.push(a, await this.stream.push(a)); }).catch((e) => this.on.error?.(e)); });
  }
  stop() { const m = this.mic; this.mic = null; m?.stop(); this.turn.close(); }
}
