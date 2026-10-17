# Q0 feature isolation (lighting bug triage)

`examples/q0_interior.zig` is the fill-bound calibration interior. It now has a
runtime feature panel and hotkeys so a lighting bug can be narrowed without
editing the scene or rebuilding.

## Run

```sh
cd examples
# Interactive: HUD on (because AGATE_Q0_FRAMES is unset).
zig build run-q0 -Doptimize=ReleaseFast
```

- The HUD lists every toggle and its state.
- Every change prints one line to stdout: `q0-feature: <name>=on|off`.
- `O` prints the complete state; `R` restores the post-env baseline.
- `F1` hides/shows the HUD.
- Finite/benchmark runs (`AGATE_Q0_FRAMES>0`) start with the HUD off so the
  frame shape and A/B timing stay unchanged. Force it with
  `AGATE_Q0_PANEL=1`; force it off with `AGATE_Q0_NO_PANEL=1`.

## Keys

| key | feature | what it changes |
|---|---|---|
| `1` | shadows | global `scene.shadows.enabled` (CSM + point/spot maps) |
| `2` | sun | directional light enabled |
| `3` | point | point light enabled (light + its shadow map) |
| `4` | spot | spot light enabled |
| `5` | area | rectangular area light enabled |
| `6` | hemi | hemispheric intensity (restore uses the captured baseline) |
| `7` | ibl | `scene.sky.ibl_intensity` (environment/probe specular) |
| `8` | local pcss | point/spot `source_radius` 0.02 ↔ 0 |
| `U` | sun pcss | `scene.shadows.pcss_enabled` |
| `9` | ssao | `scene.ssao.enabled` |
| `0` | ssgi | `scene.post_process.ssgi.enabled` |
| `S` | ssr | `scene.post_process.ssr.enabled` |
| `T` | taa | `scene.post_process.taa.enabled` |
| `B` | bloom | `scene.post_process.bloom.enabled` |
| `L` | local tone | `scene.post_process.local_tonemapping.enabled` |
| `F` | fog | `scene.post_process.fog.enabled` |
| `D` | dof | `scene.post_process.dof.enabled` |
| `H` | shafts | `scene.post_process.shafts.enabled` |
| `P` | particles | dust emitter start/stop |
| `G` | glass | tinted glass pane visibility |
| `R` | reset | restore every value captured after scene/env setup |
| `O` | state | print one state line |
| `F1` | HUD | show/hide the panel |

## Reproducible isolation (no HUD)

`AGATE_Q0_DISABLE` takes a comma/space list of the same names:

```sh
# Shadows off only.
AGATE_Q0_DISABLE=shadows AGATE_Q0_FRAMES=120 zig build run-q0 -Doptimize=ReleaseFast

# Local shadows off but sun/sun-PCSS still on.
AGATE_Q0_DISABLE=point,spot,pcss AGATE_Q0_FRAMES=120 zig build run-q0 -Doptimize=ReleaseFast

# Lighting-only: kill the post chain features that can hide a shadow bug.
AGATE_Q0_DISABLE=ssao,ssgi,ssr,taa,bloom,local_tonemap,fog,dof,shafts \
  AGATE_Q0_FRAMES=120 zig build run-q0 -Doptimize=ReleaseFast
```

The disable list applies after the normal `AGATE_Q0_NO_*` env setup, so
`R` still restores the exact baseline (including a light that was already
off). Unknown names print a warning and are ignored.

## Suggested triage order for a lighting bug

1. Confirm the bug with everything ON.
2. `1` shadows off. If it disappears, the bug is in the shadow path. Re-enable
   and try `U` (sun PCSS), `8` (local PCSS), then `3`/`4` (light/shadow
   caster) to separate sun CSM from point/spot maps.
3. If shadows are not the cause, turn off the emitters one at a time:
   `2` sun, `3` point, `4` spot, `5` area, `6` hemi, `7` IBL.
4. If the lighting is clean, enable the post features again one at a time:
   `9` AO, `0` SSGI, `S` SSR, `T` TAA, `L` local tone, `F` fog.
5. Press `O` and attach the state line plus the toggle log to the bug report.
   `R` restores the baseline between attempts.

Honest limits: `shadows` is global, so it cannot isolate CSM from local maps by
itself; use `U`/`8` plus the light keys. `5` toggles the area light, not the
emissive ceiling panel mesh. `P` stops new dust but does not clear particles
already alive. The HUD is a debug overlay and is intentionally absent from
finite gate runs unless forced with `AGATE_Q0_PANEL=1`.
