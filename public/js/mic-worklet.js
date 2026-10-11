// AudioWorklet: forwards the microphone's mono samples to the main thread in BLOCK-sample blocks.
const BLOCK = 512;
class MicCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(BLOCK); this.n = 0; }
  process(inputs) {
    const ch = inputs[0]?.[0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === BLOCK) { this.port.postMessage(this.buf, [this.buf.buffer]); this.buf = new Float32Array(BLOCK); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("mic-capture", MicCapture);
