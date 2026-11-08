import csv
import math
import platform
import statistics
import time

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

config = GPT2Config(
    vocab_size=256,
    n_positions=512,
    n_ctx=512,
    n_embd=128,
    n_layer=2,
    n_head=4,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_cache=True,
)
model = GPT2LMHeadModel(config).cpu().eval()

weight_bytes = sum(
    p.numel() * p.element_size() for p in model.parameters()
)


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


@torch.inference_mode()
def request(prompt, output_length):
    start = time.perf_counter()

    # Prefill: consume the entire prompt, select first output token.
    result = model(input_ids=prompt, use_cache=True)
    token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = result.past_key_values
    timestamps = [time.perf_counter()]

    # Decode: consume the previous output token to select the next.
    for _ in range(output_length - 1):
        result = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
        )
        token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = result.past_key_values
        timestamps.append(time.perf_counter())

    ttft = timestamps[0] - start
    e2e = timestamps[-1] - start
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in cache
        for tensor in layer
    )
    return ttft, e2e, intervals, cache_bytes


print({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "threads": torch.get_num_threads(),
    "weight_MiB": round(weight_bytes / 2**20, 3),
})

rows = []
for prompt_length, output_length in [
    (16, 32), (128, 32), (256, 32), (16, 128)
]:
    prompt = torch.randint(0, config.vocab_size, (1, prompt_length))

    for _ in range(3):
        request(prompt, output_length)  # excluded warmups

    samples = []
    wall_start = time.perf_counter()
    for _ in range(20):
        samples.append(request(prompt, output_length))
    wall_seconds = time.perf_counter() - wall_start

    ttfts = [s[0] for s in samples]
    e2es = [s[1] for s in samples]
    intervals = [gap for s in samples for gap in s[2]]

    row = {
        "prompt_tokens": prompt_length,
        "output_tokens": output_length,
        "ttft_p50_ms": statistics.median(ttfts) * 1000,
        "ttft_p95_ms": percentile(ttfts, 95) * 1000,
        "itl_p50_ms": statistics.median(intervals) * 1000,
        "itl_p95_ms": percentile(intervals, 95) * 1000,
        "e2e_p50_ms": statistics.median(e2es) * 1000,
        "e2e_p95_ms": percentile(e2es, 95) * 1000,
        "output_tokens_per_second": 20 * output_length / wall_seconds,
        "requests_per_second": 20 / wall_seconds,
        "final_cache_MiB": samples[-1][3] / 2**20,
    }
    rows.append(row)
    print({k: round(v, 3) if isinstance(v, float) else v
           for k, v in row.items()})

with open("results2.csv", "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)