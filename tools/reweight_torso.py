"""Re-derive the spine / neck weight block from the mesh currently in build/<slug>/.

extract_fbx.py bakes the weights from the FBX geometry, but a character's mesh is
sometimes corrected afterwards (points moved / seams welded), which leaves the torso
weights pointing at the old positions.  This recomputes just the hips -> head block
with arkit_fit.torso_kernel() on the shipped points and keeps every other joint's
influence.

It is not part of run_all.py: use it when you edit build/<slug>/mesh.npz by hand.

usage: python tools/reweight_torso.py <ModelSlug> [...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import arkit_fit as fit


def reweight(slug):
    bdir = config.build_dir(slug)
    npz = os.path.join(bdir, "mesh.npz")
    if not os.path.exists(npz):
        raise SystemExit("%s: run tools/extract_fbx.py %s first" % (npz, slug))
    with open(os.path.join(bdir, "mesh_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    d = dict(np.load(npz))

    JOINTS = meta["joints"]
    JIDX = {j: i for i, j in enumerate(JOINTS)}
    PTS = d["points"].astype(float)
    ji, jw = d["joint_idx"].astype(np.int64), d["joint_w"].astype(float)
    nv = len(PTS)
    pos = {k: np.array(v, float) for k, v in meta["joint_positions"].items()}

    torso = np.array([JIDX[j] for j in fit.TORSO_CHAIN])
    dense = np.zeros((nv, len(JOINTS)), np.float32)
    for k in range(ji.shape[1]):
        dense[np.arange(nv), ji[:, k]] += jw[:, k]

    share = dense[:, torso].sum(1)                     # this vertex' torso share
    dense[:, torso] = fit.torso_kernel(PTS, pos).astype(np.float32) * share[:, None]

    new_ji, new_jw, _ = fit.top_influences(dense, 4)

    moved = int((np.abs(new_jw - jw).max(1) > 1e-3).sum())
    d["joint_idx"], d["joint_w"] = new_ji, new_jw
    bak = npz + ".bak"
    if not os.path.exists(bak):
        os.replace(npz, bak)                           # one-time backup of the original
    np.savez_compressed(npz, **d)
    print("%s: torso weights recomputed (%d of %d verts changed by >1e-3, backup %s)"
          % (slug, moved, nv, os.path.basename(bak)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="recompute the spine / neck weights")
    ap.add_argument("slugs", nargs="*", help="model slugs (default: every .fbx in models/)")
    a = ap.parse_args(argv)
    for s in (a.slugs or config.all_models()):
        reweight(s)


if __name__ == "__main__":
    main()
