"""Downscale / recompress the extracted textures and update mesh_meta.json.

baseColor -> JPEG q92 (RGB) at most --max px, or PNG when the material needs the
alpha channel (hair / eyelashes); normal -> PNG (RGB) at most --max px.
Re-running is safe: a texture that already has the target name/format is skipped.

usage: python tools/prep_textures.py <ModelSlug> [...] [--max 2048]
"""
import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MAX_TEX = 2048
QUALITY = 92


def prep(slug, max_tex=MAX_TEX):
    build = config.build_dir(slug)
    tex = config.tex_dir(slug)
    meta_path = os.path.join(build, "mesh_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit("%s: run tools/extract_fbx.py %s first" % (meta_path, slug))
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    for info in meta["materials"]:
        alpha = bool(info.get("alpha"))
        for key, fmt in (("baseColor", "png" if alpha else "jpg"), ("normal", "png")):
            fn = info.get(key)
            if not fn:
                continue
            src = os.path.join(tex, fn)
            if not os.path.exists(src):
                print("  missing texture", src)
                info[key] = None
                continue

            with Image.open(src) as img:
                has_alpha = (img.mode in ("RGBA", "LA")
                             or (img.mode == "P" and "transparency" in img.info))
                img = img.convert("RGBA" if (alpha and has_alpha) else "RGB")
                if max(img.size) > max_tex:
                    w, h = img.size
                    scale = max_tex / max(w, h)
                    img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                                     Image.LANCZOS)
                out = os.path.splitext(fn)[0] + "." + fmt
                dst = os.path.join(tex, out)
                if fmt == "jpg":
                    img.save(dst, "JPEG", quality=QUALITY, optimize=True)
                else:
                    img.save(dst, "PNG", optimize=True)
                size, dims = os.path.getsize(dst), img.size

            if out != fn and os.path.exists(src):
                os.remove(src)               # drop the original (often 4 MB, PNG/TGA)
            info[key] = out
            print("  %-34s -> %-34s %sx%s %4d KB" % (fn, out, dims[0], dims[1], size // 1024))

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    files = os.listdir(tex) if os.path.isdir(tex) else []
    total = sum(os.path.getsize(os.path.join(tex, f)) for f in files)
    print("  total tex: %d KB in %d files" % (total // 1024, len(files)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompress the extracted textures")
    ap.add_argument("slugs", nargs="*", help="model slugs (default: every .fbx in models/)")
    ap.add_argument("--max", type=int, default=MAX_TEX, help="longest side in pixels")
    a = ap.parse_args(argv)
    for s in (a.slugs or config.all_models()):
        print("== prep_textures", s)
        prep(s, a.max)


if __name__ == "__main__":
    main()
