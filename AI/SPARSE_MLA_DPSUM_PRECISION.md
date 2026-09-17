# Sparse-MLA training accuracy: where the error comes from, and the two fixes

Investigation of why the FA4 sparse-MLA (DSA, top-k gathered KV) training gradients
deviate from an fp64 reference, with a stage-by-stage error decomposition, and the two
fixes it motivated. Both apply automatically to sparse-MLA forwards whose inputs require
grad; there are no user-facing knobs:

1. the forward emits the bf16 rounding residual of `out` (`o_lo = fp32(O) - bf16(O)`) and
   the backward preprocess forms `dpsum = rowsum(dO * (out + o_lo))` instead of
   `rowsum(dO * out)`;
2. the forward runs the online softmax with an exact running max instead of FA4's lazy
   rescale (see "Forward" below).

Kernels: `flash_fwd_mla_sm100.py` (residual store in the correction/epilogue warps,
`rescale_threshold` constructor parameter), `flash_bwd_preprocess.py` (residual
consumption), `interface.py` (`o_lo` plumbing, threshold selection).

## Method

`agent_space/dsa-accuracy/decomp.py` runs one sparse-MLA fwd+bwd (B=1, H=128,
hdim 64 rope + 512 latent, bf16, softmax_scale = 1/sqrt(576)), captures every kernel
intermediate by wrapping the interface functions (saved `p`/`row_max`, `dpsum`,
`scale_p`, the bf16 `ds` tensor, fp32 `dk`/`dv` before autograd's cast), and compares
each stage against

1. an exact fp64 reference computed from the same bf16 inputs, and
2. an *emulated ideal bf16 pipeline*: what a dense fused backward of the cuDNN/FA3 kind
   does numerically: fp32 P recompute, P rounded to bf16 only as the `dV` MMA operand,
   dS formed in fp32 then rounded to bf16 once for the dQ/dK MMAs, dpsum from the bf16
   output, fp32 accumulation, bf16 dq/dqv.

Substitution experiments isolate each kernel's own contribution (e.g. `dq` recomputed
in fp64 from the kernel's own bf16 `ds`, compared with the kernel's `dq`).
Relative-L2 error in percent throughout; T = S = 4096 tokens, top-k W = 2048.
One bf16 rounding of a multiplicand contributes ~0.166% RMS (`2^-8 * sqrt(0.541/3)`),
independent roundings add in quadrature.

## Findings

**All kernels compute what they should to fp32 level.** The dq/dqv/dk GEMM kernels
reproduce fp64 GEMMs of the kernel's own bf16 dS to 1e-6..1e-4 relative; the main
kernel's dS matches `P_kernel * (dP - dpsum_kernel) * scale` to 4e-5. There is no
gross bug. The gap to an fp64 reference is made of bf16 roundings:

| term | load-P mode | recompute-P mode | ideal pipeline |
|---|---|---|---|
| dq/dqv output rounding | 0.166 | 0.166 | 0.166 |
| dS rounded to bf16 for the dQ/dK MMAs | 0.166 | 0.166 | 0.166 |
| saved P rounded to bf16 by the forward | 0.146 | – | – |
| dpsum from bf16 `out` (row-coherent) | regime-dependent | regime-dependent | regime-dependent |

iid-normal inputs (every row attends broadly), before this change:

| output | load-P | recompute-P | ideal |
|---|---|---|---|
| dq (bf16) | 0.284 | 0.244 | 0.244 |
| dqv (bf16) | 0.251 | 0.221 | 0.221 |
| dk (fp32) | 0.232 | 0.180 | 0.180 |
| dv (fp32) | 0.221 | 0.170 | 0.170 |

Recompute-P mode is at the ideal pipeline to 1e-4 in every metric; load-P mode pays
one extra rounding (the forward's bf16 `p`) on dS and on the `dV` P-operand. In this
regime the dpsum term is small (0.065% of dq).

**The dpsum term dominates as soon as attention is peaked.** With inputs built so each
token attends mostly to its own key (`q_t += beta * k_t`, `qv_t += beta * v_t`, key t
always selected), `dP ~ dpsum` for the dominant slot and the error of
`dpsum = rowsum(dO * bf16(O))` (~2^-9 relative, coherent across the row) survives
the cancellation in `dS = P * (dP - dpsum)`. FA4 in both modes *and* the ideal pipeline
degrade identically, because all form dpsum from the bf16 output:

| self-attention weight | dq rel-L2 | dq per-row p99 | dq per-row p99.9 |
|---|---|---|---|
| ~0.1 (beta 0.2) | 0.30 | 0.80 | 6.5 |
| ~0.4 (beta 0.3) | 0.38 | 10.7 | 73 |
| ~0.8 (beta 0.4) | 0.89 | 105 | 333 |
| ~0.95 (beta 0.6) | 20.7 | 690 | 2124 |

With an exact dpsum the same runs give dq 0.236-0.237% and p99 < 0.5% everywhere.
Causal DSA training data is in this regime: the indexer selects the current token and
recent context, which usually carry most of the attention mass.

## Fix 1: dpsum from the O residual

The forward has the fp32 `O` in TMEM right before the bf16 downcast. When the input
requires grad, it additionally writes `o_lo = bf16(fp32(O) * scale - bf16(fp32(O) *
scale))` — the rounding residual, exact in fp32 and rounded once to bf16 — to a tensor
of `out`'s shape/dtype, saved for backward. The preprocess reads `out + o_lo`
(recovers the fp32 `O` to ~2^-16 relative) when forming dpsum. Nothing else changes:
`out` and `lse` are bitwise identical, the main backward kernel is untouched, and the
knob composes with `gather_bwd_recompute_p` and `gather_bwd_token_chunk`.

Measured (T=S=4096, W=2048, rel-L2 % vs fp64; "ideal" = emulated dense bf16 pipeline):

| regime | output | before | after | ideal |
|---|---|---|---|---|
| iid, load-P | dq / dqv / dk / dv | 0.284 / 0.251 / 0.232 / 0.221 | 0.278 / 0.247 / 0.226 / 0.217 | 0.244 / 0.221 / 0.180 / 0.170 |
| iid, recompute-P | dq / dqv / dk / dv | 0.244 / 0.221 / 0.180 / 0.170 | 0.237 / 0.216 / 0.168 / 0.166 | 0.244 / 0.221 / 0.180 / 0.170 |
| peaked 0.8, either mode | dq / dqv / dk / dv | 0.889 / 0.884 / 0.857 / 0.199 | 0.237 / 0.236 / 0.167 / 0.134 | 0.889 / 0.884 / 0.856 / 0.199 |
| peaked 0.8, dq per-row p99 / p99.9 | | 105 / 333 | 0.44 / 0.90 | 105 / 333 |
| peaked 0.95, load-P | dq / dk | 20.7 / 20.4 | 0.242 / 0.173 | 20.7 / 20.4 |

Kernel-level dpsum error: 0.166% -> 0.0023%. The remaining dq error is the bf16 output
rounding plus the bf16 dS operand (0.235% floor); dk/dv (fp32) are at the single-rounding
floor of the bf16 dS/P operands.

### Cost (GB200, T=S=16384, W=2048, H=128, `torch.profiler` kernel self time)

| kernel | off | on |
|---|---|---|
| fwd | 7.70 ms | 8.37 ms |
| bwd preprocess | 1.50 ms | 1.01 ms |
| main bwd, dq/dqv gemm, dk gemm | unchanged | unchanged |
| fwd + bwd | 40.73 ms | 40.93 ms |

End to end (CUDA-event timing of `flash_attn_func` + `torch.autograd.grad`, 50 iters):

| config | flag | fwd ms | bwd ms | fwd+bwd ms | peak extra GiB |
|---|---|---|---|---|---|
| load-P, 16K | off / on | 8.87 / 9.60 | 33.03 / 32.54 | 40.73 / 40.93 | 20.55 / 22.55 |
| load-P, 16K, causal | off / on | 8.98 / 9.17 | 33.01 / 32.52 | 41.10 / 40.91 | 20.55 / 22.55 |
| load-P, 32K | off / on | 18.77 / 20.06 | 69.35 / 68.47 | 85.62 / 86.58 | 41.10 / 45.10 |
| recompute-P, 16K | off / on | 8.28 / 8.84 | 36.91 / 36.98 | 44.05 / 44.70 | 12.31 / 14.31 |

Memory: one extra `out`-sized bf16 tensor saved from forward to backward (2 GiB at
16K tokens x 128 heads; 25% of the saved `p` in load-P mode, ~1x the saved activations
in recompute-P mode).

Final defaults for training forwards (o_lo + exact softmax max, see the next section)
versus upstream behavior, same measurement, T=16K (one GPU, one session):

| config | fwd ms | bwd ms | fwd+bwd ms | peak extra GiB |
|---|---|---|---|---|
| load-P, upstream (lazy rescale, no o_lo) | 9.14 | 33.07 | 40.88 | 20.55 |
| load-P, o_lo only | 9.81 | 32.59 | 41.19 | 22.55 |
| load-P, o_lo + exact max (new default) | 10.11 | 32.59 | 41.48 | 22.55 |
| recompute-P, upstream | 8.22 | 36.91 | 44.00 | 12.31 |
| recompute-P, new default | 9.15 | 36.99 | 44.82 | 14.31 |

### Implementation notes (register budget)

Both kernels are at their register cap and the naive versions spilled badly
(fwd local memory 168 -> 928 B/thread, +4.7 ms; preprocess 336 -> 2720 B/thread with
ptxas collapsing to 32 regs, +4.3 ms):

- Forward: the epilogue warps run at 128 regs and already hold the fp32 O tile
  (128 regs) during the downcast. Computing the residual from that tile keeps fp32,
  bf16 and residual tiles live at once. Instead, after the O TMA store of each split
  is issued, the fp32 O is re-read from TMEM 32 columns at a time (the
  `correction_rescale` access pattern), the residual chunk is formed and stored to
  gmem with the existing 128-bit register-to-global copy atom (`STG.E.128`). Source
  level chunking of the register tile does *not* help: ptxas re-hoists it (the
  register-to-global partition's leading mode is `(8, n_atoms)`, i.e. the whole row
  segment; slice `[(None, a), 0, 0]` to address one atom).
- Preprocess: the (O, dO) tile pair already fills 255 regs. The residual variant
  streams one row-slice at a time (load O/dO/o_lo for slice m, reduce over the head
  dimension, keep 8 partial sums) — 74 regs, no spills, and faster than the original
  whole-tile variant. The original code path is kept bit-for-bit when no residual is
  passed. Pitfall: computing the per-slice reduction *inside* the dynamic row-validity
  `if` produced wrong sums for every other slice (verified against torch); the loads
  stay guarded, the reduction is unconditional and the write is masked as before.

## Fix 2 (forward): the lazy softmax rescale puts a coherent gain error on peaked rows

FA4's online softmax only rescales the O/row_sum accumulators when the block max grows by
more than `rescale_threshold = 8` (log2 units); otherwise the stale max is kept and the
block's probabilities are `p = exp2(scale*(S - m_stale))`, up to 2^8. For the row's
dominant key that is `exp2(delta)` with a non-integer `delta`, so its bf16 rounding
(the P@V MMA operand) is a ~2^-9 relative error on the term that *is* the row: the
effective attention weights no longer sum to 1 and the whole output row carries a
coherent gain error. With an exact running max the dominant p is exactly 1.0 and the
remaining roundings are incoherent. This is systematic for exactly the rows that matter
(peaked attention whose dominant key is not processed first), compounds across layers
(a per-token, per-head gain), and enters the backward through dpsum even with `o_lo`.
Measured with the dominant key in slot 0 (top-k order; blocks are processed in
decreasing order, so it comes last), `|<out, o_ref>/<o_ref, o_ref> - 1|` per row:

| config | rescale threshold | out rel-L2 | row gain rms / p99 | dpsum (with o_lo) | dq (with o_lo) |
|---|---|---|---|---|---|
| T=512, W=256, self weight ~0.97 | 8 (upstream) | 0.1706 | 0.064 / 0.24 | 0.041 | 0.637 |
| same | 0 (exact max) | 0.1656 | 0.043 / 0.15 | 0.0012 | 0.239 |
| T=4096, W=2048, self weight ~0.5 | 8 (upstream) | 0.2070 | 0.123 / 0.30 | 0.125 | 0.263 |
| same | 0 (exact max) | 0.1667 | 0.014 / 0.036 | 0.019 | 0.239 |

(ideal-exact-dpsum dq: 0.236 in all four.) Sparse-MLA *training* forwards (any input
requires grad) now use threshold 0; inference forwards keep 8. Cost: fwd 8.37 -> 8.57 ms
at 16K (+2.5%). For A/B experiments construct `FlashAttentionMLAForwardSm100` with
another `rescale_threshold` (the harness in `agent_space/dsa-accuracy/` wraps the class).
An alternative that fixes `out` at zero cost — accumulating `row_sum` over the
bf16-rounded p so the normalization is exact — would not fix recompute-P gradients (the
backward's fp32 P would still disagree with the forward's rounded dominant p), so the
exact max was preferred.

## What was ruled out

- dq vs dqv asymmetry (0.284 vs 0.251 in load-P): identical in the ideal emulation; it
  is a property of the math (dqv has a coherent `scale * dO_i * sum_j P_ij` component
  that dilutes the relative rounding noise), not a kernel defect.
- fp32 accumulation order, `ex2/lg2/rcp.approx`, the `ln2 * log2e` LSE round trip,
  fp32 atomics: <= 1e-5 relative, invisible next to bf16 roundings.
- The preprocess with the original bf16 `out` reproduces `rowsum(dO * bf16(O))` to
  fp32 level; its error was entirely the input's.

## Remaining levers (not done)

1. Prefer `gather_bwd_recompute_p=True` for training: removes the forward's bf16 `p`
   from dS and the double rounding of the `dV` P operand (iid dq 0.278 -> 0.237,
   dk 0.226 -> 0.168 with this change), and saves the `p` tensor.
2. The bf16 rounding of dS as the dQ/dQv/dK MMA operand (0.166%) is the last
   kernel-side term above the output floor. Options: hi+lo bf16 split (two MMAs; the
   dq/dqv + dk gemms are ~30% of the backward, so +10-20% step time) or tf32 (half MMA
   rate, 2x operand traffic). fp16 dS is not available for free: a mixed f16 x bf16
   `tcgen05.mma.kind::f16` instruction descriptor is accepted by ptxas but traps with
   `CUDA_ERROR_ILLEGAL_INSTRUCTION` on GB200 (probe in `agent_space/dsa-accuracy/umma_mixed/`),
   so K/V/Q would have to be converted to fp16 as well.
3. The streamed preprocess variant is faster than the whole-tile one; it could replace
   the original for all backward paths at the cost of a (tolerance-level) change in
   dpsum summation order.
