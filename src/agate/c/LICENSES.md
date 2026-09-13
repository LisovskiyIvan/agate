# Vendored third-party code in src/agate/c/

Provenance and licensing for the vendored C dependencies. Each file also
embeds its own license text (see the file header/end).

| File | Upstream | Revision | License |
| --- | --- | --- | --- |
| `dr_mp3.h` | [mackron/dr_libs](https://github.com/mackron/dr_libs) (`dr_mp3.h`, v0.7.4) | commit `51e61d308dde6b437fce0c5fabb32cd86b40f4d7` (master, path `dr_mp3.h`) | Public domain or MIT-0 (statements at the end of the file) |
| `stb_vorbis.c` | [nothings/stb](https://github.com/nothings/stb) (`stb_vorbis.c`, v1.22) | commit `1ee679ca2ef753a528db5ba6801e1067b40481b8` (master, path `stb_vorbis.c`) | Public domain or MIT (statements at the end of the file) |
| `stb_image.h` | [nothings/stb](https://github.com/nothings/stb) | vendored before this note existed | Public domain / MIT (embedded in the file header) |
| `cgltf.h` | [jkuhlmann/cgltf](https://github.com/jkuhlmann/cgltf) | vendored before this note existed | MIT (embedded in the file header) |
| `meshopt/` | [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) v1.2 (`meshoptimizer.h`, `indexcodec.cpp`, `vertexcodec.cpp`, `vertexfilter.cpp`) | commit `9d9890c73011d75920af614485296d1e03e95448` (tag `v1.2`) | MIT (see `meshopt/LICENSE.md`) |
| `box3d/` | Box3D v0.1.0 | vendored before this note existed | MIT (see `box3d/LICENSE`) |

## Notes

- `dr_mp3.h` and `stb_vorbis.c` were added for MP3 / Ogg Vorbis decoding in
  the audio module (`src/agate/audio/decode.zig`). Both are compiled once
  through `c_impl.c`; the Zig side sees declarations only
  (`STB_VORBIS_HEADER_ONLY` for stb_vorbis, which ships as a single
  header-plus-implementation file).
- `meshopt/` was added for glTF `EXT_meshopt_compression` support in the
  loader (`src/agate/loader/meshopt.zig`, C glue `agate_cgltf_decode_meshopt`
  in `c_impl.c`). Decoder-only subset: the three .cpp files are runtime-free
  C++ (no exceptions/RTTI/libc++), compiled from `build.zig`. Vendored
  verbatim, no local modifications.
- SHA-256 of the files as vendored:
  - `dr_mp3.h`: `997b7ee18de6e6b81e2a83f1ea9fc62aef25c62b28d48db95635f49e65de0a2f`
  - `stb_vorbis.c`: `4c7cb2ff1f7011e9d67950446b7eb9ca044f2e464d76bfbb0b84dd2e23e65636`
  - `meshopt/meshoptimizer.h`: `21a72040a75bacf6ddefb7e74f1cf566af1e68ea5e4dc0db598278f4681e0b87`
  - `meshopt/indexcodec.cpp`: `8648e71f044e0a23622c199f250818a62d68fd05673e2e692167f257de27565f`
  - `meshopt/vertexcodec.cpp`: `a3c51e91602c607a82ad0fdc32b089e890e3a3caf7934d82b7a993d64cd683ed`
  - `meshopt/vertexfilter.cpp`: `22fe9f4bb0179be212d8d440bd5cb3510a505be8b2a7ba069ae890cdd0010998`
