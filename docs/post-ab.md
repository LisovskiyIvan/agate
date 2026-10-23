# Post-effect A/B harness (q0)

`examples/q0_interior.zig` has a drift-balanced interleaved A/B mode for
attributing the fill-bound post/lighting chain. It is a **screening tool**, not
a CI performance gate: macOS GPU clocks and window focus can make a disabled
effect look slower, so only deltas that clear the block noise floor are
actionable.

## Run

```sh
cd examples

# One pair (600 frames: 2 settle + 8 measured blocks x 60).
AGATE_Q0_AB=ssao AGATE_Q0_STILL=1 AGATE_Q0_TIMINGS=1 \
  zig build run-q0 -Doptimize=ReleaseFast

# Every pair in one process/window; this is the only comparable cross-pair
# ranking (per-pair A/B is still interleaved inside the run).
AGATE_Q0_AB=matrix AGATE_Q0_AB_BLOCK=60 AGATE_Q0_STILL=1 \
  AGATE_Q0_TIMINGS=1 zig build run-q0 -Doptimize=ReleaseFast
```

`AGATE_Q0_AB_BLOCK` changes the block length (clamped to the `AbBench`
scratch). Longer blocks reject more of the effect-switch/TAA-reset transient;
60 is the default and the shortest block that produced stable large-effect
deltas in the 17.10.2026 run. `AGATE_Q0_AB=matrix` reads the same heavy q0
preset for every pair and resets the baseline between pairs. The single-pair
mode obeys the q0 env knobs (`AGATE_Q0_AO_MODE`, `AGATE_Q0_SSGI_PASS`, ...).

Pairs:

| pair | arm A | arm B |
|---|---|---|
| `ssgi` | on (mode from setup) | off |
| `ssao` | on | off |
| `ssr` | inline | half-res pass |
| `ssr_off` | on (mode from setup) | off |
| `bloom` | on | off |
| `fxaa` | on | off |
| `taa` | on | off |
| `fog` | on | off |
| `shadows` | on | off |
| `shadow_locals` | point/spot `cast_shadows` on | off (sun CSM keeps the baseline) |
| `pcss` | local point/spot `source_radius=0.02` | `0.0` |
| `dof` | on | off |
| `shafts` | on | off |
| `local_tonemap` | on | off |

Matrix mode runs 13 cost pairs in this order: `ssao, ssgi, ssr_off, bloom,
fxaa, taa, shadows, shadow_locals, pcss, dof, shafts, local_tonemap, fog`.

## Output

Each pair prints the block trace and three independent summaries:

```text
AB_FRAME_RESULT pair=... delta_median_ms=... noise_floor_ms=... verdict=... ...
AB_POST_RESULT  ...
AB_MAIN_RESULT  ...
```

`FRAME` is the completed command-buffer sample, `POST`/`MAIN` are the existing
GPU phase brackets. The phase spans overlap and their absolute values are not
additive; the A/B delta is the signal. `verdict=SIGNIFICANT` means the pooled
median delta exceeded the range of kept block medians and every adjacent
mixed-block pair agreed in sign.

## 17.10.2026 screening (Apple M4, Metal, ReleaseFast, 728x882)

- Only two pairs cleared the noise floor:
  - `shadows`: **-5.21 ms** frame when off (main phase -2.21 ms); the shadow
    map pass itself is ~0.5 ms, so the rest is PBR shadow sampling.
  - `local_tonemap`: **-1.69 ms** frame when off.
- `ssao`, `ssgi`, `ssr_off`, `bloom`, `fxaa`, `taa`, `fog`, `pcss`, `dof`,
  `shafts` were **inconclusive** at their measured deltas (<1 ms). Disabling a
  post effect can reduce GPU utilization enough to lower clocks and make the
  remaining work look slower; the matrix harness records this instead of
  turning it into a false win.
- Absolute frame times in matrix mode clustered around 13 ms and are not
  comparable across pairs/runs.

This is why the next structural step is temporal upscaling / frame-graph
resource work rather than blind tap cuts: the per-effect post deltas are below
the current measurement floor, while the full-resolution composite/shadow
sampling are the proven large consumers. Raw logs belong outside the repo
(`agate-bench-harness/post_ab_2026-10-17/`).

## 21.10.2026 shadow-budget wave (Apple M4, Metal, ReleaseFast, 728x882)

New pair `shadow_locals` (arm A: point+spot `cast_shadows` on; arm B: off, the
sun CSM keeps the scene baseline) splits local vs sun shadow cost. The `MAIN`
channel is quoted below — it is the phase the effect changes; the `FRAME`
channel sat above its noise floor for the whole session.

- `shadows` (all shadows off): **-1.51 / -1.65 / -1.75 / -1.95 ms** main over
  four runs (three SIGNIFICANT).
- `shadow_locals` (locals off): **-1.73 / -0.89 / -1.74 / -0.53 ms** main over
  four runs — same magnitude as `shadows`, higher scatter.
- Reading: in q0 (1 point + 1 spot, local PCSS on) the **local** point/spot
  shadow sampling is the bulk of shadow cost; the sun CSM sits within noise.
  Absolute medians drifted up to ~0.7 ms between sessions (the `shadows`-off
  arm, which no change here touches, moved 1.62 -> 2.35 ms), so only
  within-run deltas are comparable.

Sun `NdotL == 0` CSM skip — bit-identical (the sun term is multiplied by
`NdotL`; the cascade-debug tint still samples), now in `common/pbr_surface.glsl`
and `pbr.glsl`. Same-session alternating A/B of a gated vs ungated binary over
the `shadows` pair: 1.53 ms vs 1.59 ms shadow cost — **below the noise floor**.
Kept as a strict work reduction on back-facing pixels; no measurable win in q0.

`pcss` re-run (local PCSS vs legacy fixed PCF): **-0.34 / -0.35 ms** main,
consistent sign but INCONCLUSIVE at this session's noise floor. The
importance/distance-scaled PCSS idea is rejected on this evidence: its ceiling
is well under 0.5 ms in the calibration scene.

Shadow-caster hysteresis (audit 1.3) landed in the same wave — a quality fix
(atlas slots stop flapping on near-equal scores), not a perf item, so it has
no A/B row here; see the regression tests in `src/scene/light_rig_tests.zig`.
