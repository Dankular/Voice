// In-browser chat LLM (transformers.js, WebGPU) with token streaming and interruption (for barge-in).
// `tf` is the @huggingface/transformers module (passed in so the same code runs in the browser and under Node tests).
// Qwen3 "thinks" by default; the chat template is applied with enable_thinking:false so replies start immediately.
export const DEFAULT_LLM = { model: "onnx-community/Qwen3-0.6B-ONNX", dtype: "q4f16" };

export async function loadLlm(tf, { model = DEFAULT_LLM.model, dtype = DEFAULT_LLM.dtype, device = "webgpu", onProgress } = {}) {
  const [tokenizer, lm] = await Promise.all([
    tf.AutoTokenizer.from_pretrained(model, { progress_callback: onProgress }),
    tf.AutoModelForCausalLM.from_pretrained(model, { dtype, device, progress_callback: onProgress }),
  ]);
  const stopping = new tf.InterruptableStoppingCriteria();
  let busy = null;

  // messages: [{role, content}]; onText(delta) is called as text is generated. Resolves with the full reply (partial if interrupted).
  async function chat(messages, { onText, maxNewTokens = 160, temperature = 0.7, topP = 0.8, topK = 20 } = {}) {
    await busy;                                                       // one generation at a time
    let done; busy = new Promise((r) => (done = r)); stopping.reset();
    try {
      const inputs = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true, enable_thinking: false });
      let text = "";
      const streamer = new tf.TextStreamer(tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: (t) => { text += t; onText?.(t); } });
      await lm.generate({ ...inputs, max_new_tokens: maxNewTokens, do_sample: true, temperature, top_p: topP, top_k: topK, streamer, stopping_criteria: stopping });
      return text;
    } finally { done(); }
  }
  return { chat, interrupt: () => stopping.interrupt(), tokenizer, model: lm };
}
