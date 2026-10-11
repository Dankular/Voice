// Unit tests for the agent turn loop and sentence chunker (mocked ASR / LLM / TTS). Run: node scripts/test_agent.mjs
import assert from "node:assert/strict";
import { Agent, SentenceChunker } from "../public/js/agent.js";

// ---- SentenceChunker
{
  const c = new SentenceChunker(), got = [];
  for (const d of ["Hello there. ", "It is 3", ".5 degrees out", "side! Do you want ", "more? Sure.", " Okay"]) got.push(...c.push(d));
  got.push(...c.flush());
  // first chunk needs >= 20 chars, later ones >= 50; "3.5" is never split; whatever is left comes out on flush()
  assert.deepEqual(got, ["Hello there. It is 3.5 degrees outside!", "Do you want more? Sure. Okay"]);
  const k = new SentenceChunker(); assert.deepEqual([...k.push("你好。"), ...k.flush()], ["你好。"]);
  const q = new SentenceChunker(); assert.deepEqual(q.push("Short. "), []); assert.deepEqual(q.flush(), ["Short."]);   // below the minimum: held back
  console.log("chunker ok:", JSON.stringify(got));
}

// ---- Agent with mocks
const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
function rig({ transcript = "what time is it", reply = ["It is noon. ", "Anything else I can do for you today? ", "Just ask."], speakMs = 20, tokenMs = 5 } = {}) {
  const log = []; let stopped = false, interrupted = false;
  const tts = { speak: async (s) => { log.push("speak:" + s); await tick(speakMs); if (stopped) throw new Error("stopped"); }, finish: async () => { log.push("finish"); },
    stop: () => { stopped = true; log.push("tts.stop"); } };
  const llm = { interrupt: () => { interrupted = true; log.push("llm.interrupt"); },
    chat: async (m, { onText }) => { log.push("chat:" + m.at(-1).content); let full = ""; interrupted = false;
      for (const d of reply) { await tick(tokenMs); if (interrupted) break; full += d; onText(d); } return full; } };
  const states = []; const agent = new Agent({ transcribe: async () => transcript, llm, tts, on: { state: (s) => states.push(s), error: (e) => log.push("error:" + e.message) } });
  return { agent, log, states, reset: () => { stopped = false; } };
}
{ // normal turn
  const { agent, log, states } = rig(); await agent.utterance(new Float32Array(1600));
  const spoken = log.filter((l) => l.startsWith("speak:")).map((l) => l.slice(6));
  assert.deepEqual(spoken, ["It is noon. Anything else I can do for you today?", "Just ask."]); assert.ok(log.includes("finish"));
  assert.deepEqual(agent.history.map((h) => h.role), ["user", "assistant"]); assert.deepEqual(states, ["thinking", "speaking", "idle"]);
  console.log("normal turn ok:", JSON.stringify(spoken));
}
{ // barge-in while speaking: only spoken sentences are kept, later ones never spoken
  const r = rig({ reply: ["First sentence is long enough here. ", "Second sentence follows right after it. ", "Third one should never be heard at all."], speakMs: 60 });
  const done = r.agent.utterance(new Float32Array(1600)); await tick(90); r.agent.interrupt(); await done;
  const spoken = r.log.filter((l) => l.startsWith("speak:")); assert.ok(spoken.length >= 1 && spoken.length < 3, "spoke " + spoken.length);
  assert.ok(r.log.includes("tts.stop") && r.log.includes("llm.interrupt")); assert.ok(!r.log.includes("finish"));
  const last = r.agent.history.at(-1); assert.equal(last.role, "assistant"); assert.ok(last.content.endsWith("…") && !last.content.includes("Third"));
  assert.equal(r.states.at(-1), "idle"); console.log("barge-in ok; history keeps:", JSON.stringify(last.content));
}
{ // new utterance supersedes a turn in progress
  const r = rig({ speakMs: 50 }); const a = r.agent.utterance(new Float32Array(1600)); await tick(60); r.reset();
  const b = r.agent.utterance(new Float32Array(1600)); await Promise.all([a, b]);
  assert.equal(r.log.filter((l) => l.startsWith("chat:")).length, 2); assert.equal(r.states.at(-1), "idle"); console.log("supersede ok");
}
{ // noise / empty transcript: no LLM call, no history
  const r = rig({ transcript: " . " }); await r.agent.utterance(new Float32Array(1600));
  assert.equal(r.log.filter((l) => l.startsWith("chat:")).length, 0); assert.equal(r.agent.history.length, 0); assert.equal(r.states.at(-1), "idle"); console.log("empty transcript ok");
}
{ // errors surface via on.error and leave the agent usable
  const r = rig(); r.agent.transcribe = async () => { throw new Error("asr down"); }; await r.agent.utterance(new Float32Array(1600));
  assert.ok(r.log.includes("error:asr down")); assert.equal(r.states.at(-1), "idle"); console.log("error path ok");
}
console.log("ALL OK");
