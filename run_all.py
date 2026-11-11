"""Mixamo FBX -> ARKit / RealityKit BodyTrackedEntity USDZ: whole pipeline in one go.

usage:
    python run_all.py                       # every .fbx in models/
    python run_all.py --only Ch16 Remy      # just these
    python run_all.py --from build_usd      # resume after a step
    python run_all.py --fail-fast
    python run_all.py --models D:/mixamo --dist D:/out

steps: extract_fbx -> prep_textures -> build_usd -> validate_usdz -> render_preview
`build/arkit_skel.json` (Apple's reference skeleton) is produced once, up front, by
tools/dump_robot_skeleton.py and shared by every model.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
sys.path.insert(0, TOOLS)
import config                                            # noqa: E402  (needs TOOLS on the path)

STEPS = ["extract_fbx", "prep_textures", "build_usd", "validate_usdz", "render_preview"]


def run_script(script, args, log_path=None, env=None):
    """run one tool in a child process so a crash cannot take the batch down"""
    cmd = [sys.executable, os.path.join(TOOLS, script + ".py")] + list(args)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n----- %s %s rc=%d %.1fs\n%s"
                      % (script, " ".join(args), proc.returncode, time.time() - t0, out))
    return proc.returncode, out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mixamo FBX -> ARKit USDZ pipeline")
    ap.add_argument("--only", nargs="*", default=None, metavar="SLUG",
                    help="model slugs (default: every .fbx in models/)")
    ap.add_argument("--from", dest="start", default=None, choices=STEPS,
                    help="resume at this step")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop at the first failing model (default: keep going)")
    ap.add_argument("--models", default=None, help="input directory (default models/)")
    ap.add_argument("--build", default=None, help="intermediate directory (default build/)")
    ap.add_argument("--dist", default=None, help="output directory (default dist/)")
    ap.add_argument("--skip-dump", action="store_true",
                    help="reuse the existing build/arkit_skel.json")
    a = ap.parse_args(argv)

    if a.models:
        config.MODELS_DIR = Path(a.models).resolve()
    if a.build:
        config.BUILD_DIR = Path(a.build).resolve()
        config.SKEL_JSON = config.BUILD_DIR / "arkit_skel.json"
    if a.dist:
        config.DIST_DIR = Path(a.dist).resolve()
    config.ensure_dirs()

    # the steps run as child processes, so the directories have to travel as environment
    # variables (this is also how `tools/*.py` find them when run on their own)
    env = dict(os.environ)
    env.update({"MIXAMO_MODELS": str(config.MODELS_DIR),
                "MIXAMO_BUILD": str(config.BUILD_DIR),
                "MIXAMO_DIST": str(config.DIST_DIR),
                "ARKIT_SKEL_JSON": str(config.SKEL_JSON)})

    models = a.only or config.all_models()
    if not models:
        raise SystemExit("no models found in %s - put the Mixamo .fbx files there"
                         % config.MODELS_DIR)
    start = STEPS.index(a.start) if a.start else 0

    # 0. the reference skeleton (shared, produced once from Apple's biped_robot.usdz)
    if not a.skip_dump and (start == 0 or not os.path.exists(config.SKEL_JSON)):
        print("== dump_robot_skeleton (Apple's ARKit reference)")
        rc, out = run_script("dump_robot_skeleton", [], env=env)
        print(out.strip())
        if rc != 0:
            raise SystemExit("cannot read Apple's reference skeleton")

    summary = {}
    for i, slug in enumerate(models, 1):
        bdir = config.build_dir(slug)
        os.makedirs(bdir, exist_ok=True)
        print("\n[%d/%d] %s  (%s)" % (i, len(models), slug, os.path.basename(config.model_path(slug))))
        entry = {"slug": slug, "source": os.path.basename(config.model_path(slug)),
                 "steps": {}, "ok": True}
        log_path = os.path.join(bdir, "pipeline.log")
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("model %s  (%s)\n" % (slug, config.model_path(slug)))

        for k, step in enumerate(STEPS):
            if k < start:
                continue
            args = [config.usdz_path(slug)] if step == "validate_usdz" else [slug]
            rc, out = run_script(step, args, log_path, env)
            tail = [ln for ln in out.strip().splitlines() if ln.strip()]
            entry["steps"][step] = {"rc": rc, "tail": tail[-3:]}
            print("   %-14s rc=%d  %s" % (step, rc, tail[-1] if tail else ""))
            if rc != 0:
                entry["ok"] = False
                entry["failed_step"] = step
                for ln in tail[-25:]:
                    print("      " + ln)
                break
        if entry["ok"]:
            meta_path = os.path.join(bdir, "mesh_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                entry["counts"] = meta.get("counts")
                entry["warnings"] = meta.get("warnings", [])
                entry["materials"] = [m["name"] for m in meta.get("materials", [])]
                entry["fit"] = meta.get("fit", {})
        usdz = config.usdz_path(slug)
        entry["usdz"] = os.path.basename(usdz)
        entry["usdz_mb"] = round(os.path.getsize(usdz) / 1e6, 1) if os.path.exists(usdz) else 0
        summary[slug] = entry
        with open(os.path.join(config.BUILD_DIR, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
        if not entry["ok"] and a.fail_fast:
            break

    print("\n================ summary ================")
    ok = 0
    for slug, e in summary.items():
        ok += bool(e["ok"])
        c = e.get("counts") or {}
        print("%s %-26s tris %-7s verts %-7s %5.1f MB  %s" % (
            "OK  " if e["ok"] else "FAIL", slug, c.get("tris", "-"), c.get("verts", "-"),
            e["usdz_mb"], e.get("failed_step", "")))
    print("%d/%d models completed" % (ok, len(summary)))
    print("output:", config.DIST_DIR)
    return 0 if ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
