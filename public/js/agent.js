// The voice-agent turn loop: user utterance -> transcribe -> LLM (streamed) -> sentences -> speech, with barge-in.
// Everything it needs is injected, so it is testable without a browser:
//   transcribe(audio16k) -> Promise<string>
//   llm.chat(messages, {onText}) -> Promise<string>;  llm.interrupt()
//   tts.speak(text) -> Promise (resolves when that sentence has been synthesised and handed to the player; rejects/returns early after tts.stop())
//   tts.finish() -> Promise (resolves when everything queued has been heard);  tts.stop() -> cut the audio now
// on: state(s) 'idle'|'thinking'|'speaking', user(text), agent(textSoFar), error(e)

export const SYSTEM_PROMPT = "You are a friendly voice assistant. The user is talking to you out loud and your reply will be spoken aloud. " +
  "Answer in one to three short, natural sentences. Use plain spoken language only: no markdown, lists, emojis, code or special symbols. " +
  "If you did not catch something, ask the user to repeat it.";

// Splits streamed text into speakable sentences; the first is released early (short) so speech starts sooner.
export class SentenceChunker {
  constructor(firstMin = 20, restMin = 50) { this.buf = ""; this.n = 0; this.firstMin = firstMin; this.restMin = restMin; }
  push(delta) {
    this.buf += delta; const out = [];
    for (;;) {
      const min = this.n ? this.restMin : this.firstMin;
      // a sentence end counts only once whitespace follows it (so "3.5" is not split); CJK full stops need no whitespace
      const re = /(?:[.!?…]+["'”’)\]]*\s+|[。！？]+["'”’)\]]*)/g; let cut = -1, m;
      while ((m = re.exec(this.buf))) if (re.lastIndex >= min) { cut = re.lastIndex; break; }
      if (cut < 0) break;
      const s = this.buf.slice(0, cut).trim(); this.buf = this.buf.slice(cut);
      if (s) { out.push(s); this.n++; }
    }
    return out;
  }
  flush() { const s = this.buf.trim(); this.buf = ""; if (s) this.n++; return s ? [s] : []; }
}

const clean = (t) => t.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/[*_`#>]+/g, "").replace(/\s+/g, " ").trim();

export class Agent {
  constructor({ transcribe, llm, tts, on = {}, system = SYSTEM_PROMPT, maxHistory = 12, minChars = 2 }) {
    Object.assign(this, { transcribe, llm, tts, on, system, maxHistory, minChars });
    this.history = []; this.state = "idle"; this.turn = null; this.chain = Promise.resolve();
  }
  set(s) { if (s !== this.state) { this.state = s; this.on.state?.(s); } }

  // A finished user utterance. Supersedes a turn still in progress.
  utterance(audio) {
    this.interrupt();
    const turn = (this.turn = { cancelled: false, spoken: [] });
    this.chain = this.chain.then(() => this.run(turn, audio)).catch((e) => this.on.error?.(e));
    return this.chain;
  }
  // The user started talking over the agent (or thinking): stop everything that is in flight.
  interrupt() {
    const t = this.turn; if (!t || t.cancelled) return;
    t.cancelled = true; this.llm.interrupt(); this.tts.stop();
  }

  async run(turn, audio) {
    this.set("thinking");
    try {
      const text = clean(await this.transcribe(audio));
      if (turn.cancelled) return;
      if (text.replace(/[^\p{L}\p{N}]/gu, "").length < this.minChars) { return; }       // noise / empty transcript
      this.history.push({ role: "user", content: text }); this.on.user?.(text);
      const messages = [{ role: "system", content: this.system }, ...this.history.slice(-this.maxHistory)];
      const chunker = new SentenceChunker(); let reply = ""; const queue = []; let wake = null, closed = false;
      const say = async () => {                                       // speaks queued sentences in order while the LLM keeps generating
        for (;;) {
          if (turn.cancelled) return;
          const s = queue.shift();
          if (s === undefined) { if (closed) return; await new Promise((r) => (wake = r)); continue; }
          this.set("speaking");
          try { await this.tts.speak(s); } catch (e) { if (!turn.cancelled) throw e; return; }
          if (!turn.cancelled) turn.spoken.push(s);
        }
      };
      const speaker = say();
      const enqueue = (arr) => { queue.push(...arr.map(clean).filter(Boolean)); wake?.(); wake = null; };
      const full = await this.llm.chat(messages, { onText: (d) => { reply += d; this.on.agent?.(clean(reply)); if (!turn.cancelled) enqueue(chunker.push(d)); } });
      if (!turn.cancelled) enqueue(chunker.flush());
      closed = true; wake?.(); wake = null;
      await speaker;
      if (!turn.cancelled) await this.tts.finish();
      // remember what was actually said: the whole reply, or only the sentences that were spoken before an interruption
      const said = turn.cancelled ? turn.spoken.join(" ") : clean(full);
      if (said) this.history.push({ role: "assistant", content: turn.cancelled ? said + " …" : said });
    } finally { if (this.turn === turn) this.set("idle"); }
  }
}
