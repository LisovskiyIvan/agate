#!/usr/bin/env python3
"""Patch vendored sokol-zig resource-handle bindings for Zig 0.16.

Why
---
Zig 0.16 lowers a 4-byte `extern struct` (e.g. `gfx.Buffer { id: u32 }`) that is
passed *by value* to an `extern fn` as a widened `i64` LLVM parameter. When the
handle is memory-resident at the call site (any value whose storage escapes SROA,
e.g. a phi between two pipelines), Zig materialises that `i64` with

    %slot = alloca %gfx.Buffer, align 4
    ...
    %v = load i64, ptr %slot, align 8      // 4-byte object, 8-byte load -> UB

LLVM treats the out-of-bounds load as undefined and is free to fold it: in
ReleaseFast/ReleaseSafe it collapsed the whole call argument to `0`, so
`sg_apply_pipeline()` dereferenced a NULL pipeline slot (EXC_BAD_ACCESS at
offset 0x8). Debug never inlines the wrapper, so the bad load never appears.

Declaring the same C entry point with a `u32` parameter is ABI-equivalent on
AArch64 (the C side reads `w0`) and makes Zig load exactly 4 bytes.

What it changes
---------------
For every `extern fn sg_*(..., Handle, ...)` in `gfx.zig` where `Handle` is one of
the 4-byte resource handles, rewrite the parameter type to `u32`, and rewrite the
matching argument in the generated wrapper to `handle.id`.

Usage
-----
    python3 tools/patch_sokol_handle_abi.py            # patch gfx.zig in place
    python3 tools/patch_sokol_handle_abi.py --check    # report only, no writes
    python3 tools/patch_sokol_handle_abi.py FILE...    # other binding files

The script is idempotent: already-patched declarations are skipped, so it can be
re-run after refreshing `vendor/sokol` from upstream
(floooh/sokol-zig @fafdd96d, "machine generated, do not edit").
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

HANDLES = ("Buffer", "Image", "Pipeline", "Sampler", "Shader", "View")
DEFAULT_TARGET = "vendor/sokol/src/sokol/gfx.zig"

EXTERN_RE = re.compile(r"^extern fn (?P<name>sg_[a-z_0-9]+)\((?P<params>.*)\) (?P<ret>.*);$")
WRAP_RE = re.compile(r"^pub fn (?P<name>[a-zA-Z_0-9]+)\(")


def split_top_level(text: str) -> list[str]:
    """Split a parameter/argument list on commas that are not nested."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    tail = "".join(current)
    if tail.strip():
        parts.append(tail)
    return parts


def patch_file(path: pathlib.Path, write: bool) -> int:
    lines = path.read_text().split("\n")
    out = list(lines)
    patched = 0
    problems: list[str] = []

    i = 0
    while i < len(lines):
        m = EXTERN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        extern_params = split_top_level(m.group("params")) if m.group("params").strip() else []
        handle_idx = [n for n, p in enumerate(extern_params) if p.strip() in HANDLES]
        if not handle_idx:
            i += 1
            continue

        # --- locate the generated wrapper that follows this declaration ---
        j = i + 1
        while j < len(lines) and not lines[j].startswith("pub fn ") and not lines[j].startswith("extern fn "):
            j += 1
        if j >= len(lines) or not lines[j].startswith("pub fn "):
            problems.append(f"{path}:{i + 1}: no wrapper found after `extern fn {m.group('name')}`")
            i += 1
            continue

        # --- collect the wrapper signature (it may span several lines) ---
        sig, k = [], j
        depth = 0
        while k < len(lines):
            sig.append(lines[k])
            depth += lines[k].count("(") - lines[k].count(")")
            if depth <= 0 and "{" in lines[k]:
                break
            k += 1
        sig_text = " ".join(s.strip() for s in sig)
        wm = WRAP_RE.match(sig[0])
        if not wm:
            problems.append(f"{path}:{j + 1}: unparsable wrapper signature")
            i = k
            continue
        wrap_params_text = sig_text[sig_text.index("(") + 1 : sig_text.rindex(") {")] if ") {" in sig_text else sig_text[sig_text.index("(") + 1 : sig_text.rindex(")")]
        wrap_params = split_top_level(wrap_params_text) if wrap_params_text.strip() else []
        if len(wrap_params) != len(extern_params):
            problems.append(
                f"{path}:{j + 1}: wrapper `{wm.group('name')}` has {len(wrap_params)} params, "
                f"extern `{m.group('name')}` has {len(extern_params)}; skipping"
            )
            i = k
            continue

        # --- collect the wrapper body ---
        body_start = k
        depth = 0
        while k < len(lines):
            depth += lines[k].count("{") - lines[k].count("}")
            if depth <= 0 and k > body_start:
                break
            if depth <= 0 and k == body_start and lines[k].count("{"):
                pass
            k += 1
        body_end = k
        body = out[body_start : body_end + 1]

        # --- map handle params to wrapper parameter names ---
        arg_names: dict[int, str] = {}
        for n in handle_idx:
            head = wrap_params[n].split(":")[0].strip()
            wtype = wrap_params[n].split(":", 1)[1].strip() if ":" in wrap_params[n] else ""
            if wtype != extern_params[n].strip():
                problems.append(
                    f"{path}:{j + 1}: param {n} is `{wtype}` in wrapper but "
                    f"`{extern_params[n].strip()}` in extern; skipping whole declaration"
                )
                arg_names = {}
                break
            arg_names[n] = head
        if not arg_names:
            i = body_end + 1
            continue

        # --- rewrite the extern declaration ---
        new_params = list(extern_params)
        for n in handle_idx:
            new_params[n] = "u32"
        out[i] = f"extern fn {m.group('name')}({', '.join(p.strip() for p in new_params)}) {m.group('ret')};"

        # --- rewrite the call inside the wrapper body ---
        done = False
        for bi in range(len(body)):
            cm = re.search(rf"\b{re.escape(m.group('name'))}\(", body[bi])
            if not cm:
                continue
            # walk to the matching close paren, spanning lines
            text = "\n".join(body[bi:])
            open_at = cm.end() - 1
            depth = 0
            close_at = None
            for pos in range(open_at, len(text)):
                if text[pos] == "(":
                    depth += 1
                elif text[pos] == ")":
                    depth -= 1
                    if depth == 0:
                        close_at = pos
                        break
            if close_at is None:
                problems.append(f"{path}:{body_start + bi + 1}: unterminated call to `{m.group('name')}`")
                break
            args_text = text[open_at + 1 : close_at]
            args = split_top_level(args_text) if args_text.strip() else []
            if len(args) != len(extern_params):
                problems.append(
                    f"{path}:{body_start + bi + 1}: call passes {len(args)} args, extern has "
                    f"{len(extern_params)}; skipping"
                )
                break
            changed = False
            for n, name in arg_names.items():
                arg = args[n].strip()
                if arg.endswith(".id"):
                    continue
                if arg != name:
                    problems.append(
                        f"{path}:{body_start + bi + 1}: arg {n} is `{arg}`, expected `{name}`; "
                        f"manual fix required"
                    )
                    continue
                args[n] = f"{name}.id"
                changed = True
            if not changed:
                break
            new_args_text = ", ".join(a.strip() for a in args)
            new_text = text[: open_at + 1] + new_args_text + text[close_at:]
            before = body_start + bi
            # splice new_text back into `out` starting at absolute line `body_start + bi`
            tail_lines = new_text.split("\n")
            out[before : before + len(tail_lines)] = tail_lines
            body = out[body_start : body_end + 1]
            done = True
            break
        if not done:
            # declaration was rewritten but the call was not: roll back
            out[i] = lines[i]

        patched += 1
        print(f"  patched {m.group('name')} (+{', '.join(f'arg{n}' for n in handle_idx)})")
        i = body_end + 1

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        return len(problems)

    if write and patched:
        path.write_text("\n".join(out))
        print(f"\nwrote {path} ({patched} declarations patched)")
    elif patched:
        print(f"\n{patched} declarations would be patched (--check, no write)")
    else:
        print("nothing to patch (already applied?)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help=f"binding files to patch (default: {DEFAULT_TARGET})")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    targets = [root / f for f in args.files] if args.files else [root / DEFAULT_TARGET]
    bad = 0
    for t in targets:
        if not t.is_file():
            print(f"missing file: {t}", file=sys.stderr)
            return 2
        print(f"== {t.relative_to(root)}")
        bad += patch_file(t, write=not args.check)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
