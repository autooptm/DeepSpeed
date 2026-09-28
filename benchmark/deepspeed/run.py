# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Single-GPU DeepSpeed inference benchmark: greedy text generation.

Usage follows the library's own inference tutorial
(docs/_tutorials/inference-tutorial.md): load a HuggingFace causal LM, wrap it
with `deepspeed.init_inference()`, and generate.

One unit of work is one request, and the key loop below carries it end to end:
tokenize the prompt, generate a fixed number of new tokens, decode the
completion, append it to the output file.

    python benchmark/deepspeed/run.py
"""
import argparse
import json
import os
import random
import time

import torch

import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
OPT_WINDOW = int(os.environ.get("DS_BENCH_OPT_WINDOW", 256))


def _single_process_dist_env():
    """deepspeed.init_inference() expects the usual launcher variables; this
    benchmark is a single process, so supply single-process defaults."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")


def load_records(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_engine(model_path, dtype="fp16", kernel_inject=False, device="cuda"):
    """Load the model and wrap it in the DeepSpeed inference engine."""
    _single_process_dist_env()
    torch_dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    engine = deepspeed.init_inference(model,
                                      tensor_parallel={"tp_size": 1},
                                      dtype=torch_dtype,
                                      replace_with_kernel_inject=kernel_inject)
    module = engine.module
    module.to(device)
    if torch_dtype is not torch.float32 and next(module.parameters()).dtype != torch_dtype:
        module.to(torch_dtype)
    return tokenizer, module


class StepDecoder:

    def __init__(self, model, device, window=OPT_WINDOW):
        self.model = model
        self.device = device
        self.window = window
        kw = dict(config=model.config, max_cache_len=window, device=device, dtype=model.dtype)
        try:
            self.cache = StaticCache(max_batch_size=1, **kw)
        except TypeError:
            self.cache = StaticCache(batch_size=1, **kw)
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.cache_pos = torch.zeros((1,), dtype=torch.long, device=device)
        self.pos_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.mask = torch.zeros((1, window), dtype=torch.long, device=device)
        self._g = None
        self.logits = None

    def _step(self):
        out = self.model(input_ids=self.ids, attention_mask=self.mask,
                         position_ids=self.pos_ids, past_key_values=self.cache,
                         cache_position=self.cache_pos, use_cache=True)
        return out.logits[:, -1, :]

    def prefill(self, input_ids):
        prompt_len = int(input_ids.shape[1])
        self.cache.reset()
        self.mask.zero_()
        self.mask[:, :prompt_len] = 1
        pos = torch.arange(prompt_len, device=self.device)
        out = self.model(input_ids=input_ids, attention_mask=self.mask,
                         position_ids=pos.unsqueeze(0), past_key_values=self.cache,
                         cache_position=pos, use_cache=True)
        return out.logits[:, -1, :], prompt_len

    def decode(self, token, absolute_pos):
        self.ids.fill_(int(token))
        self.cache_pos.fill_(absolute_pos)
        self.pos_ids.fill_(absolute_pos)
        self.mask[:, absolute_pos] = 1
        if self._g is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(stream)
            self._g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g):
                self.logits = self._step()
        self._g.replay()
        return self.logits


def generate_one(tokenizer, model, prompt, max_new_tokens, device, decoder=None):
    """One request: tokenize -> generate -> decode."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if decoder is None:
        with torch.no_grad():
            out = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 max_new_tokens=max_new_tokens,
                                 min_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 num_beams=1,
                                 use_cache=True,
                                 pad_token_id=pad_id)
        new_ids = out[0, input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True), int(new_ids.numel())

    if int(input_ids.shape[1]) + max_new_tokens + 1 > decoder.window:
        raise ValueError("prompt of %d tokens + %d new exceeds the window %d; "
                         "raise DS_BENCH_OPT_WINDOW or pass --no-opt-1"
                         % (int(input_ids.shape[1]), max_new_tokens, decoder.window))
    new_ids = []
    with torch.no_grad():
        logits, prompt_len = decoder.prefill(input_ids)
        for step in range(max_new_tokens):
            token = int(logits.argmax(-1))
            new_ids.append(token)
            logits = decoder.decode(token, prompt_len + step)
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids)


def parse_args():
    p = argparse.ArgumentParser(description="DeepSpeed single-GPU inference benchmark")
    p.add_argument("--model", default=os.environ.get("DS_BENCH_MODEL", "facebook/opt-1.3b"))
    p.add_argument("--prompts", default=os.environ.get("DS_BENCH_PROMPTS",
                                                       os.path.join(HERE, "prompts.json")))
    p.add_argument("--num-prompts", type=int, default=int(os.environ.get("DS_BENCH_NUM_PROMPTS", 128)))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", default=os.environ.get("DS_BENCH_DTYPE", "fp16"), choices=sorted(DTYPES))
    p.add_argument("--kernel-inject", action="store_true",
                   help="replace attention/MLP with DeepSpeed's fused inference kernels")
    p.add_argument("--no-opt-1", dest="opt_1", action="store_false",
                   help="disable optimization 1")
    p.set_defaults(opt_1=os.environ.get("DS_BENCH_OPT_1", "1") != "0")
    p.add_argument("--warmup-units", type=int, default=int(os.environ.get("DS_BENCH_WARMUP_UNITS", 0)),
                   help="run this many requests first, untimed, to warm the engine")
    p.add_argument("--output-dir", default=os.environ.get("DS_BENCH_OUT", os.path.join(HERE, "outputs")))
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = build_engine(args.model, args.dtype, args.kernel_inject, device)
    decoder = None
    if args.opt_1 and device == "cuda":
        decoder = StepDecoder(model, device)

    records = load_records(args.prompts)
    prompts = [r["prompt"] for r in records][:args.num_prompts]
    if args.warmup_units > 0:
        rng = random.Random(11)
        picks = rng.sample(range(len(prompts)), min(args.warmup_units, len(prompts)))
        prompts = [prompts[i] for i in sorted(picks)] + prompts

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "generations.jsonl")

    new_tokens = 0
    t0 = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as sink:
        for prompt in prompts:
            completion, n_new = generate_one(tokenizer, model, prompt, args.max_new_tokens,
                                             device, decoder)
            sink.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")
            new_tokens += n_new
    if device == "cuda":
        torch.cuda.synchronize()
    loop_s = time.perf_counter() - t0

    n = len(prompts)
    print(json.dumps({"bench": "deepspeed-inference",
                      "model": args.model,
                      "dtype": args.dtype,
                      "opt_1": bool(decoder is not None),
                      "kernel_inject": bool(args.kernel_inject),
                      "requests": n,
                      "warmup_units": args.warmup_units,
                      "new_tokens": new_tokens,
                      "loop_s": round(loop_s, 4),
                      "ms_per_request": round(1e3 * loop_s / max(n, 1), 3),
                      "ms_per_token": round(1e3 * loop_s / max(new_tokens, 1), 4),
                      "output": out_path}))


if __name__ == "__main__":
    main()
