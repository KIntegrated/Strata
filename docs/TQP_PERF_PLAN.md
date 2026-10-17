# TQP (Gyro-L) performance plan

Status: Gyro-L runs correctly (TQP5/6/7, load check and parity 0 failures) but is slower than
IQ3_S on the same card. This plan closes that gap.

## Measured gap

NVIDIA CMP 170HX (sm_80), 262144 context, int8 KV, MTP, greedy, 128-token replies, same saved
prompts. Gyro-L on the merged engine (0.1.45 + TQP), IQ3_S on 0.1.44.1.

| prompt | Gyro-L prefill / decode | IQ3_S prefill / decode | gap |
|---|---|---|---|
| 4K | 1130 / 59 | 2064 / 104 | -45% / -43% |
| 32K | 1896 / 61 | 2491 / 94 | -24% / -35% |
| 128K | 1876 / 61 | 2328 / 90 | -19% / -32% |

Both models hold all 24,576 experts in VRAM, route top-10 of 512 experts, and use the same
MTP draft (accepted 57-70% for both). The expert bytes per layer are comparable (TQP6 is
3.125 bits/weight, IQ3_S 3.5). So the gap is in the kernels and the expert I/O mode, not the
model size.

## Root causes

### 1. TQP misses the decode-once (`Split`) path - the big one

`src/kernels/cuda/iq_kernels.cu` has `kSplit<144/145/146> = true` with `SplitTq<...>` calling
`tq_load<TY>` once per weight part and `tq_apply` per activation. There is **no
`kSplit<147..149>`**, so `Fmt<147..149>::dot` is used instead and `row_dot_multi` - the MTP
verify window (up to 6 tokens) and the prompt's NC-token rows - re-decodes the lane's 16
codebook values for **every** activation. TQ2_T decodes once and applies NC times.

### 2. `tqp_states8` is expensive per lane

`tqp_states8<TY>` (`iq_kernels.cu`) computes 8 states with 8 x (3 byte loads + two
`% (8*K)` modulos). `tq_states4` (TQK) does 4 states from one 64-bit window read and shifts.
The same code is used by `dq_tqp` (the f32/f16 dequantizer).

### 3. TQP CPU rows are scalar only

`rows_on()` in `src/kernels/cpu/tq_cpu.cpp` routes types 147-149 to `tq_rows_scalar_tqp`
("no SIMD kernel yet"). The AVX2 / AVX-512 kernels exist for 144-146 only. With every expert
in VRAM the pool is idle, but on a card that streams experts (or with `--peer-device`, the
CPU share, or a smaller GPU) this path is the difference between a few and tens of tok/s.

### 4. The prompt lend region is re-read from the GGUF

Gyro-L needs `--mmap-experts` (see 5), and that mode has no RAM complement, so every prompt
re-reads the 1,799 borrowed cache slots (3.31 GiB) plus the complement from the GGUF. The
engine reported ~130 GB of file reads over the benchmark run; IQ3_S ran `--resident-experts`
with a 0.00 GiB complement and no prompt re-reads. That is most of the 4K prefill gap
(1130 vs 2064 tok/s).

### 5. The resident arena does not fit, and nothing says so up front

Without `--mmap-experts` the engine fills the resident expert arena (observed RSS 40.7 GiB)
and thrashes for 900 s on a 47 GB PC, then the server gives up
("the engine did not report READY within 900 s"). With `--mmap-experts` it starts in ~30 s.

## Work items

Each has its own GitHub issue; this table is the summary.

| # | item | files | expected |
|---|---|---|---|
| 1 | TQP decode-once `Split` (`kSplit<147..149>`, `SplitTq`) | `src/kernels/cuda/iq_kernels.cu` | remove the per-activation re-decode in the verify window and the NC-token rows |
| 2 | Cheaper TQP state extraction (one 64-bit window, no modulo) | `src/kernels/cuda/iq_kernels.cu` (`tqp_states8`, `dq_tqp`) | fewer ALU/load ops per lane |
| 3 | TQP AVX2 / AVX-512 CPU rows | `src/kernels/cpu/tq_avx2.cpp`, `tq_avx512.cpp`, `tq_cpu.cpp` | the pool path at SIMD speed |
| 4 | TQP benchmark + regression coverage | `src/kernels/native_expert_bench.cpp`, `native_expert_parity.cpp`, `bench/` | measure before/after; catch regressions |
| 5 | Keep the prompt lend region in RAM for mmap packs | `src/core/expert_source.cpp`, `src/program/generate.cpp` | 4K prefill without the 3.31 GiB re-read |
| 6 | Refuse or fall back when the resident arena does not fit | `src/program/generate.cpp`, `serve/server.py` | no 900 s thrash on a RAM-short PC |

Order: 1, 2, 4 (measure), 3, 5, 6. 1 is the lowest-risk, highest-leverage change.

## Acceptance

- TQP parity (`--synthetic-hadamard tqp6/tqp5/tqp7`) and the load check stay at 0 failures;
  the decoded rows stay bitwise equal to the ggml reference (the current `to_float` equality).
- 32K decode within 10% of IQ3_S (>= 85 tok/s) and 32K prefill >= 2200 tok/s on the CMP 170HX.
- The 4K/32K/128K matrix re-run recorded in `bench/`.

## Measurement

Reuse the harness at `bench/ab/` (`bench_gyro.py`): the same saved 4K/32K/128K request bodies,
greedy, 128 tokens, MTP, one run per size, `--mmap-experts`, 262144 context. Record
`bench/ab/gyro_l.json` before and after each change. Kernel-level numbers come from
`build/native_expert_bench` (add the TQP types there first, item 4).
