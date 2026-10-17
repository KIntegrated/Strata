# Gyro-L (TQP5 / TQP6 / TQP7) on Strata

Gyro-L's experts are the **pair-trellis types TQP5/TQP6/TQP7 (ggml 147/148/149)**: 128 values
per block = fp16 scale + `8*K` bytes of a 64-step, K-bit circular stream (K = 5/6/7), two weights
per step, taken from the TQ2_T codebook's first pair (`tq2t_lut_f16[2*hyb(2*s)+j]`).

The llama.cpp clone from 2026-10-07 (`dc255f2`) stops at 146 (`TQ2_T/TQK6/TQK7`), so this port
needs a GGML that has the TQP types.

## GGML

```sh
git clone --depth 1 https://gitlab.com/agentionai-group/llama.cpp ../agention-llama-gitlab
# any revision with GGML_TYPE_TQP5/6/7; release/gyro-l (9aa51ce, 2026-10-12) has them
```

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80;86" -DSTRATA_GGML_DIR="$PWD/../agention-llama-gitlab"
cmake --build build -j
```

## Pack and check

```sh
STRATA_GGUF_PY=../agention-llama-gitlab/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf <Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf> --out packs/gyro-l --compat-bf16
build/native_expert_parity --load-check packs/gyro-l <Qwen3.8-Flash-Next-Gyro-L-TQ3_0.gguf>
build/native_expert_parity --synthetic-hadamard tqp6/tqp6 tqp5/tqp5 tqp7/tqp7
```

The second one must end with `0 failures` (AVX2 bitwise equal to the scalar kernel, GPU
Hadamard bitwise equal to the CPU rotation, GPU dequantizer bitwise equal to ggml's `to_float`).

## Serve

Add **`--mmap-experts`** when the experts do not fit in RAM. Without it the engine fills the
resident expert arena (Gyro-L: 45 GiB) and a 47 GB PC thrashes for 900 s before the server gives
up. With it the file tier keeps the experts in VRAM (24,576 slots here) and system RAM stays page
cache.

## Notes

- The CPU TQP path is **scalar only** (no AVX2/AVX-512 TQP kernel yet). With every expert in VRAM
  the CPU pool is idle, so this costs nothing there; on a card that streams experts it would.
- `hadamard_rows` / `hadamard_rows_f16` split their launch over `grid.y <= 65535`: a 32K prompt's
  `rows = chunk x top-k` exceeds the CUDA grid limit and the engine died with
  `hadamard_rows_f16: invalid argument` before this.
- Measured on an NVIDIA CMP 170HX (sm_80), 262144 context, int8 KV, MTP, fork base 0.1.40.3:
  4K 1123/60, 32K 1880/59, 128K 1857/57 tok/s (prefill/decode).
