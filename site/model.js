// The serving performance model, in the browser.
//
// A port of src/fastserve/perfmodel/serving.py, function for function. The Python file is the reference:
// site/data/dashboard.json carries inputs with the Python model's outputs ("checks"), and both the page and
// the test suite recompute them here and compare (site/tests/parity.mjs).
//
// One decode step for `batch` sequences at mean context `context`:
//   weights   bytes / bandwidth + FLOPs / peak
//   KV cache  batch * context * bytes per token / (bandwidth * efficiency of the attention kernel)
//   overhead  fixed per step + per sequence
// A closed loop of N users: time per request = prefill + output tokens * time per token / batch.

export const BYTES_PER_WEIGHT = { bf16: 2.0, fp8: 1.0, int8: 1.0, int4: 4.125 / 8 }; // INT4: + a scale per 128
export const KV_BYTES_PER_ELEMENT = { bf16: 2.0, fp8: 1.0 };
export const FLASH_ATTN = "flash_attn";
export const FLASHINFER = "flashinfer";

// INT4 weights are dequantized to BF16 before the multiply (Marlin), so they use BF16's peak.
export function matmulPeak(hw, weights) {
  return hw.peak[weights === "int4" ? "bf16" : weights];
}

// vLLM 0.30 on a GPU older than Hopper: FlashInfer with speculation loses the full CUDA graph.
export function piecewise(stack) {
  return stack.backend === FLASHINFER && stack.speculation != null;
}

export function kvBytesPerToken(cfg, kv) {
  return (cfg.kv_bytes_per_token_bf16 / 2) * KV_BYTES_PER_ELEMENT[kv];
}

// Bytes a decode step streams: the quantized layers, and the LM head, which stays BF16.
export function weightBytes(cfg, weights) {
  return cfg.linear_params * BYTES_PER_WEIGHT[weights] + cfg.vocab_size * cfg.hidden_size * 2;
}

// Seconds to read the weights once and multiply them for `tokens` token positions.
export function weightTime(cfg, hw, stack, tokens) {
  const head = cfg.vocab_size * cfg.hidden_size;
  let compute = (2 * cfg.linear_params * tokens) / matmulPeak(hw, stack.weights);
  compute += (2 * head * tokens) / hw.peak.bf16;
  return weightBytes(cfg, stack.weights) / hw.bandwidth + compute;
}

// Share of the memory bandwidth at which the attention kernel reads the KV cache.
export function attentionEfficiency(stack, cal, batch) {
  if (stack.backend === FLASHINFER) return cal.flashinfer_efficiency[stack.kv];
  return 1 / (1 + cal.flash_attn_batch_penalty * Math.log2(Math.max(batch, 1)));
}

// Cached tokens that must come from memory in one step. A prefix shared by a group is read once per group
// if one layer's slice of it fits in the GPU's L2 cache.
export function kvTokensRead(cfg, hw, stack, batch, context, load) {
  if (!load || !stack.prefix_caching || load.sharers <= 1 || load.shared_len <= 0) return batch * context;
  const perLayer = (load.shared_len * kvBytesPerToken(cfg, stack.kv)) / cfg.num_layers;
  if (perLayer > hw.l2_bytes) return batch * context;
  const shared = Math.min(load.shared_len, context);
  return batch * (context - shared) + (batch / Math.min(load.sharers, batch)) * shared;
}

export function kvTime(cfg, hw, stack, cal, batch, context, load = null) {
  const tokens = kvTokensRead(cfg, hw, stack, batch, context, load);
  const bytes = tokens * kvBytesPerToken(cfg, stack.kv);
  return bytes / (hw.bandwidth * attentionEfficiency(stack, cal, batch));
}

// Seconds for one decode pass: `batch` sequences, each adding `newTokens` positions.
export function stepTime(cfg, hw, stack, cal, batch, context, newTokens = 1, load = null) {
  let overhead = cal.step_s + cal.per_seq_s * batch;
  if (stack.weights === "fp8" || stack.weights === "int8") overhead += cal.act_quant_s;
  const weights = weightTime(cfg, hw, stack, batch * newTokens);
  return overhead + weights + kvTime(cfg, hw, stack, cal, batch, context, load);
}

// (linear, attention) FLOPs to prefill `newTokens` after `cached` tokens already in the cache.
export function prefillFlops(cfg, newTokens, cached = 0) {
  const linear = 2 * cfg.linear_params * newTokens;
  const pairs = newTokens * (cached + newTokens / 2);
  return [linear, 4 * pairs * cfg.head_dim * cfg.num_heads * cfg.num_layers];
}

// Seconds a prefill adds: its FLOPs, plus one read of the weights if it runs as a pass of its own.
export function prefillTime(cfg, hw, stack, cal, newTokens, cached = 0, ownPass = true) {
  if (newTokens <= 0) return 0;
  const [linear, attention] = prefillFlops(cfg, newTokens, cached);
  const efficiency = cal.prefill_linear[stack.weights] ?? 1.0;
  let seconds = linear / (matmulPeak(hw, stack.weights) * efficiency);
  seconds += attention / (hw.peak.bf16 * cal.prefill_attention);
  return seconds + (ownPass ? weightBytes(cfg, stack.weights) / hw.bandwidth : 0);
}

// Seconds of GPU time per output token of each sequence in the batch (one step serves them all).
export function timePerToken(cfg, hw, stack, cal, batch, context, load = null) {
  const spec = stack.speculation;
  if (spec == null) return stepTime(cfg, hw, stack, cal, batch, context, 1, load);
  const verify = stepTime(cfg, hw, stack, cal, batch, context, spec.k + 1, load);
  const draft = spec.k * (cal.draft_step_s + spec.draft_bytes / hw.bandwidth + cal.draft_per_seq_s * batch);
  const gpuPass = verify + draft;
  // On piecewise graphs the host works beside the GPU: a pass takes whichever of the two is slower.
  const passTime = piecewise(stack) ? Math.max(gpuPass, cal.host_step_s) : gpuPass;
  return passTime / spec.tokens_per_pass;
}

// Sequences decoding at once: the users, unless vLLM's limit or the KV cache holds fewer.
export function runningBatch(load, cal) {
  let batch = Math.min(load.users, load.max_batch);
  if (load.kv_tokens != null) {
    batch = Math.min(batch, load.kv_tokens / ((load.prompt_len + load.output_len) * cal.kv_slack));
  }
  return Math.max(batch, 1);
}

// Throughput, TPOT and (for one user) TTFT of a closed-loop workload on one configuration.
export function predict(cfg, hw, stack, cal, load) {
  const batch = runningBatch(load, cal);
  const context = load.context != null ? load.context : load.prompt_len + load.output_len / 2;
  const cached = stack.prefix_caching ? Math.min(load.cached_len, load.prompt_len) : 0;
  const prefill = prefillTime(cfg, hw, stack, cal, load.prompt_len - cached, cached, batch <= 1);
  const perToken = timePerToken(cfg, hw, stack, cal, batch, context, load);
  let request = prefill + (load.output_len * perToken) / batch;
  if (load.users <= 1) request += cal.request_s; // one user: the GPU idles while the next request is handled
  const step = stepTime(cfg, hw, stack, cal, batch, context, 1, load);
  return {
    batch,
    tok_s: load.output_len / request,
    tpot_ms: 1e3 * (perToken + (prefill * (batch - 1)) / load.output_len),
    ttft_ms: load.users <= 1 ? 1e3 * (cal.request_s + prefill + perToken) : null,
    step_ms: 1e3 * step,
    prefill_ms: 1e3 * prefill,
    kv_share: kvTime(cfg, hw, stack, cal, batch, context, load) / step,
    host_bound: piecewise(stack) && cal.host_step_s > 0 && perToken * stack.speculation.tokens_per_pass <= cal.host_step_s,
  };
}

// The KV cache's capacity in tokens: vLLM gives the cache what is left of its memory budget.
export function kvCapacity(cfg, stack, starts) {
  let gib = starts.budget_gib - starts.weights_gib[stack.weights];
  if (stack.backend === FLASHINFER) gib -= starts.flashinfer_gib;
  if (stack.speculation != null) gib -= starts.drafter_gib;
  return (gib * 2 ** 30) / kvBytesPerToken(cfg, stack.kv);
}

export function dollarsPerMillion(tokS, dollarsPerHour) {
  return (dollarsPerHour / (tokS * 3600)) * 1e6;
}

// A stack from the calculator's switches. An FP8 cache is served by FlashInfer on this GPU.
export function makeStack({ weights = "bf16", fp8Kv = false, prefix = false, speculation = null }) {
  return {
    weights,
    kv: fp8Kv ? "fp8" : "bf16",
    backend: fp8Kv ? FLASHINFER : FLASH_ATTN,
    prefix_caching: prefix,
    speculation,
  };
}

export function makeLoad(fields) {
  return {
    cached_len: 0,
    context: null,
    shared_len: 0,
    sharers: 1,
    kv_tokens: null,
    max_batch: 256,
    ...fields,
  };
}

// Recompute every check in the data file; returns the cases that differ from the Python model.
export function runChecks(data, tolerance = 1e-9) {
  const failures = [];
  for (const check of data.checks) {
    const cfg = data.models[check.model];
    const cal = data.calibrations[check.calibration];
    const got = predict(cfg, data.hardware, check.stack, cal, check.load);
    got.kv_capacity = kvCapacity(cfg, check.stack, data.startup_memory[check.model]);
    const expected = { ...check.expected, kv_capacity: check.kv_capacity };
    for (const [key, want] of Object.entries(expected)) {
      const error = Math.abs(got[key] - want) / Math.max(Math.abs(want), 1e-300);
      if (!(error <= tolerance)) failures.push({ check, key, want, got: got[key], error });
    }
  }
  return failures;
}
