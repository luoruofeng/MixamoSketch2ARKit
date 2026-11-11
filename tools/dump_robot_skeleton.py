"""Extract the authoritative ARKit body skeleton from Apple's biped_robot.usdz.

This is the only place that reads Apple's asset: everything downstream (fitting,
validation, preview) works from build/arkit_skel.json, so the reference is parsed
once and every character is measured against exactly the same numbers.

usage: python tools/dump_robot_skeleton.py [--usdz PATH] [--out PATH]
"""
import argparse
import json
import os

os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")   # Apple's file is crate 0.7.0

from pxr import Gf, Usd, UsdSkel

import config

EXPECTED_JOINTS = 91


def dump(usdz=None, out=None):
    src = str(usdz or config.reference_usdz())
    dst = str(out or config.SKEL_JSON)

    stage = Usd.Stage.Open(src)
    skel = None
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            break
    if skel is None:
        raise SystemExit("%s contains no UsdSkel.Skeleton" % src)

    raw = [str(j) for j in skel.GetJointsAttr().Get()]
    joints = [p.split("/")[-1] for p in raw]
    parents = [p.split("/")[-2] if "/" in p else None for p in raw]
    rest = skel.GetRestTransformsAttr().Get()
    bind = skel.GetBindTransformsAttr().Get()
    if len(joints) != EXPECTED_JOINTS:
        print("warning: %d joints, ARKit's defaultBody3D has %d" % (len(joints), EXPECTED_JOINTS))
    dups = len(joints) - len(set(joints))
    if dups:
        raise SystemExit("%s has %d duplicate joint names - cannot map by name" % (src, dups))

    # sanity: bind == product of rest transforms (the whole pipeline relies on it).
    # Gf is row-vector convention like the rest of this project: world = local * parent.
    world = {}
    for i, j in enumerate(joints):
        parent = parents[i]
        world[j] = (rest[i] * world[parent]) if parent else Gf.Matrix4d(rest[i])
    worst = 0.0
    for i, j in enumerate(joints):
        d = world[j] - Gf.Matrix4d(bind[i])
        worst = max(worst, max(abs(d[r][c]) for r in range(4) for c in range(4)))

    data = {
        "source": os.path.basename(src),
        "description": "ARSkeletonDefinition.defaultBody3D as authored in Apple's Biped Robot",
        "upAxis": "Y", "forwardAxis": "+Z", "units": "m",
        "count": len(joints),
        "joints": joints,
        "parents": parents,
        "bind_world": [[[float(b[r][c]) for c in range(4)] for r in range(4)] for b in bind],
        "rest_local": [[[float(l[r][c]) for c in range(4)] for r in range(4)] for l in rest],
    }
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)

    ys = [float(b[3][1]) for b in bind]
    print("source :", src)
    print("joints : %d  (root %r, hips %r, head %.3f m)  height %.3f m"
          % (len(joints), joints[0], joints[1], float(bind[joints.index("head_joint")][3][1]),
             max(ys) - min(ys)))
    print("rest*bind consistency error: %.3e" % worst)
    print("wrote", dst)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--usdz", default=None, help="Apple's biped_robot.usdz (auto-detected)")
    ap.add_argument("--out", default=None, help="destination json (default build/arkit_skel.json)")
    a = ap.parse_args(argv)
    dump(a.usdz, a.out)


if __name__ == "__main__":
    main()
