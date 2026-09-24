# Real KTX2 Basis fixtures (toktx v4.4.2, 16x16 PNG gradients)

Generated from 16x16 RGB/RGBA gradient PNGs (see /tmp, not shipped):

| File | Kind | DFD | Levels | Scheme | Use |
|---|---|---|---|---|---|
| `fx_etc1s_rgb_mip.ktx2` | ETC1S opaque | sRGB | 5 (16..1) | BASIS_LZ | BC7/ASTC/RGBA mip-chain test |
| `fx_etc1s_rgba_mip.ktx2` | ETC1S alpha | sRGB | 5 | BASIS_LZ | alpha + 2-slice test |
| `fx_uastc_rgba_mip.ktx2` | UASTC | sRGB | 5 | NONE | UASTC mip-chain test |
| `fx_uastc_rgba_zstd.ktx2` | UASTC | sRGB | 1 | ZSTD | vendored-zstd proof |
| `fx_uastc_rgb_flat.ktx2` | UASTC opaque | sRGB | 1 | NONE | sRGB variant test |
| `fx_uastc_rgb_flat_linear.ktx2` | UASTC opaque | LINEAR | 1 | NONE | UNORM variant test |
| `fx_etc1s_rgb_flat_linear.ktx2` | ETC1S opaque | LINEAR | 1 | BASIS_LZ | linear ETC1S test |
| `fx_uastc_cube.ktx2` | UASTC cube | sRGB | 1 | NONE | UnsupportedFaceCount test |

All files `ktx validate` clean. Regenerate: toktx commands in the
implementing commit message / task notes (16x16 gradient PNG + flags above).
