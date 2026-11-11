"""
Extract mesh + skin weights + textures from a Mixamo FBX, fit the 91 ARKit joints to
that character and re-shape the mesh into Apple's T-pose.

usage:  python tools/extract_fbx.py <ModelSlug> [...]   (slug = fbx basename, spaces -> _)

Output (build/<slug>/): mesh.npz, mesh_meta.json, tex/*
Everything is written in ARKit / USD space: metres, +Y up, +Z forward, left side on +X.

The joint table is produced by arkit_fit.build_target: every joint takes Apple's bone
DIRECTION and the character's own segment LENGTH, and the mesh is deformed with the same
rotations so the character is straightened into a clean T-pose (bones stay in the flesh).
"""
import argparse
import sys, os, json, hashlib, re, io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import arkit_fit as fit
from fbx_scene import Scene, clean_name

SIDE = fit.SIDE
FINGERS = fit.FINGERS_ALL
BODY = fit.BODY
MAX_INF = 4                 # ARKit/RealityKit limit on influences per vertex
SMOOTH_PASSES = 2           # Laplacian weight blur along the edges (0 = off)
SMOOTH_K = 0.5
SYM_WEIGHTS = True          # mirror the left half's weights onto the right half


# ------------------------------------------------------------------ helpers
def layer_values(geo, layer, comp_type, values_name, idx_name, n_corners, tri_vertex=None):
    """Return per-corner values for a LayerElement* node, whatever its mapping."""
    if layer is None:
        return None
    node = layer.child(values_name)
    if node is None:
        return None
    vals = np.array(node.props[0], float).reshape(-1, comp_type)
    mapping = layer.child("MappingInformationType").props[0] if layer.child("MappingInformationType") else "ByPolygonVertex"
    ref = layer.child("ReferenceInformationType").props[0] if layer.child("ReferenceInformationType") else "Direct"
    idx_node = layer.child(idx_name)
    if mapping == "ByPolygonVertex":
        if ref == "IndexToDirect" and idx_node is not None:
            return vals[np.array(idx_node.props[0], np.int32)]
        return vals
    if mapping in ("ByVertice", "ByControlPoint"):
        base = vals[tri_vertex] if tri_vertex is not None else vals
        if ref == "IndexToDirect" and idx_node is not None:
            return base[np.array(idx_node.props[0], np.int32)]
        return base
    if mapping == "AllSame":
        return np.repeat(vals[:1], n_corners, 0)
    raise ValueError("unsupported mapping %s" % mapping)


def polygons_of(PVI):
    polys, cur = [], []
    for idx in PVI:
        if idx < 0:
            cur.append(~idx)
            polys.append(cur)
            cur = []
        else:
            cur.append(idx)
    return polys


def material_slots(sc, geo_uid, mesh_uid):
    """material uids attached to the geometry, else to its mesh Model (FBX stores both ways)"""
    mats = sc.by.get("Material", [])
    mat_uids = {m.props[0] for m in mats}
    order = [u for u in sc.children_of.get(geo_uid, []) if u in mat_uids]
    if not order:
        order = [u for u in sc.children_of.get(mesh_uid, []) if u in mat_uids]
    return order


# ------------------------------------------------------------------ main
def extract(slug):
    path = config.model_path(slug)
    out = config.build_dir(slug)
    texdir = config.tex_dir(slug)
    os.makedirs(texdir, exist_ok=True)
    for f in os.listdir(texdir):
        os.remove(os.path.join(texdir, f))

    scene = Scene(path)
    S = scene.unit_scale
    AXIS = scene.axis_matrix()          # FBX world -> ARKit world (Y up, +Z forward)
    warnings = []
    print("model:", slug, "| unitScale(cm->m)=%.4f" % S,
          "| up=%d front=%d coord=%d" % (scene.up_axis, scene.front_axis, scene.coord_axis))

    ref = fit.Ref(config.SKEL_JSON)
    ARKIT = ref.joints
    if len(ARKIT) != 91:
        raise SystemExit("reference skeleton has %d joints, expected 91 - rerun "
                         "tools/dump_robot_skeleton.py" % len(ARKIT))

    # -------------------------------------------------------------- materials
    mat_objs = {m.props[0]: m for m in scene.by.get("Material", [])}
    tex_objs = {t.props[0]: t for t in scene.by.get("Texture", [])}
    video_objs = {v.props[0]: v for v in scene.by.get("Video", [])}
    op_links = [(c.props[1], c.props[2], c.props[3]) for c in scene.f.top("Connections").children
                if c.props[0] == "OP"]

    mat_table = []          # [{name, baseColor, normal, alpha, roughness, metallic}]
    mat_index = {}          # material uid -> index in mat_table
    written = {}            # content hash -> file name (dedupe identical textures)

    def source_of(tex_uid):
        """(content, filename) of the media behind a Texture node"""
        tex = tex_objs[tex_uid]
        content, fname = None, None
        for ch in scene.children_of.get(tex_uid, []):
            if ch in video_objs:
                v = video_objs[ch]
                cn = v.child("Content")
                fn = v.child("Filename") or v.child("RelativeFilename")
                fname = clean_name(fn.props[0]) if fn else None
                content = cn.props[0] if (cn is not None and cn.props) else None
        if not fname:
            fn = tex.child("FileName") or tex.child("RelativeFilename")
            fname = clean_name(fn.props[0]) if fn else None
        return content, fname

    # Mixamo sometimes emits a second Texture node pointing at the same file without
    # embedded data; index every embedded media by filename first.
    embedded_by_name = {}
    for t_uid in tex_objs:
        c, fn = source_of(t_uid)
        if c and fn:
            embedded_by_name[os.path.basename(fn).lower()] = (c, fn)

    def resolve(tex_uid):
        content, fname = source_of(tex_uid)
        if not content and fname:
            hit = embedded_by_name.get(os.path.basename(fname).lower())
            if hit:
                content, fname = hit
                print("   tex %-14s reused embedded %s" % ("media", os.path.basename(fname)))
        return content, fname

    def has_alpha(content):
        try:
            im = Image.open(io.BytesIO(content))
            if im.mode in ("RGBA", "LA"):
                return True
            return im.mode == "P" and "transparency" in im.info
        except Exception:
            return False

    def tex_file(content, fname, mat_name, suffix, ext_hint, with_alpha=False):
        if not content:
            warnings.append("%s: texture %s has no embedded data" % (mat_name, suffix))
            return None
        h = hashlib.md5(content).hexdigest()
        key = (h, bool(with_alpha))
        if key in written:
            return written[key]
        ext = os.path.splitext(fname or "")[1].lower()
        ext = ext if ext in (".png", ".jpg", ".jpeg", ".tga", ".bmp") else ext_hint
        base = mat_name + suffix
        name, k = base + ext, 1
        while os.path.exists(os.path.join(texdir, name)):
            k += 1
            name = "%s_%d%s" % (base, k, ext)
        with open(os.path.join(texdir, name), "wb") as fh:
            fh.write(content)
        written[key] = name
        print("   tex %-14s -> %-40s (%s, %d KB)" % (suffix, name, fname, len(content) // 1024))
        return name

    for uid, m in mat_objs.items():
        mat_name = re.sub(r"[^0-9A-Za-z_\-]+", "_", clean_name(m.props[1]))
        links = [(c, pr) for c, p, pr in op_links if p == uid and c in tex_objs]
        diffuse = next((c for c, pr in links if pr == "DiffuseColor"), None)
        normal = next((c for c, pr in links if pr == "NormalMap"), None)
        alpha_tex = next((c for c, pr in links if pr == "TransparentColor"), None)

        # hair / eyelashes: the cutout mask lives in the alpha of the diffuse map
        alpha = False
        if alpha_tex is not None and diffuse is not None:
            if alpha_tex == diffuse:
                alpha = True
            else:
                a_fn = (source_of(alpha_tex)[1] or "").lower()
                d_fn = (source_of(diffuse)[1] or "").lower()
                alpha = bool(a_fn) and os.path.basename(a_fn) == os.path.basename(d_fn)
                if not alpha:
                    warnings.append("%s: separate opacity texture ignored" % mat_name)

        d_content, d_fname = resolve(diffuse) if diffuse is not None else (None, None)
        if alpha and not has_alpha(d_content or b""):
            alpha = False
            warnings.append("%s: alpha requested but the diffuse map has no alpha" % mat_name)

        info = {"name": clean_name(m.props[1]), "baseColor": None, "normal": None,
                "alpha": alpha, "roughness": 0.62, "metallic": 0.0}
        if diffuse is not None:
            info["baseColor"] = tex_file(d_content, d_fname, mat_name, "_baseColor", ".png", alpha)
        if normal is not None:
            n_content, n_fname = resolve(normal)
            info["normal"] = tex_file(n_content, n_fname, mat_name, "_normal", ".png")
        mat_index[uid] = len(mat_table)
        mat_table.append(info)

    # -------------------------------------------------------------- geometry
    pts_all, tris_all, uv_all, nrm_all, mat_all = [], [], [], [], []
    geo_info = []           # (geo uid, mesh name, first vertex, vertex count, bone weights)
    vbase = 0

    for geo, mesh_uid, mesh_name in scene.geometry_of():
        vnode = geo.child("Vertices")
        if vnode is None:
            continue
        V = np.array(vnode.props[0], float).reshape(-1, 3)
        pvi_node = geo.child("PolygonVertexIndex")
        if pvi_node is None or not len(V):
            continue
        polys = polygons_of(pvi_node.props[0])
        n_corners = int(sum(len(p) for p in polys))

        Mmesh = scene.world(mesh_uid) @ AXIS
        Vw = (np.c_[V, np.ones(len(V))] @ Mmesh)[:, :3]
        print("   geo %-22s verts %-6d polys %-6d" % (mesh_name, len(V), len(polys)))

        tri_vertex = np.zeros(n_corners, np.int32)
        k = 0
        for poly in polys:
            tri_vertex[k:k + len(poly)] = poly
            k += len(poly)

        nrm = layer_values(geo, geo.child("LayerElementNormal"), 3, "Normals", "NormalsIndex",
                           n_corners, tri_vertex)
        uvs = layer_values(geo, geo.child("LayerElementUV"), 2, "UV", "UVIndex",
                           n_corners, tri_vertex)
        if nrm is None:
            warnings.append("%s: no normals, recomputed from geometry" % mesh_name)
        if uvs is None:
            warnings.append("%s: no UVs" % mesh_name)

        # material index per polygon
        lmat = geo.child("LayerElementMaterial")
        slots = material_slots(scene, geo.props[0], mesh_uid)
        gmat = [mat_index[u] for u in slots if u in mat_index] or [0]
        if lmat is not None:
            mp = lmat.child("Materials")
            mt = lmat.child("MappingInformationType")
            mtype = mt.props[0] if mt else "ByPolygon"
            raw = mp.props[0] if mp is not None else [0]
            if mtype == "AllSame":
                poly_mat = [raw[0] if raw else 0] * len(polys)
            else:
                poly_mat = list(raw[:len(polys)])
                if len(poly_mat) < len(polys):
                    poly_mat += [0] * (len(polys) - len(poly_mat))
        else:
            poly_mat = [0] * len(polys)
        poly_mat = [gmat[i] if 0 <= i < len(gmat) else gmat[0] for i in poly_mat]

        tris, tuv, tn, tm = [], [], [], []
        corner = 0
        for pi, poly in enumerate(polys):
            n = len(poly)
            for j in range(1, n - 1):
                c = [corner, corner + j, corner + j + 1]
                tris.append([poly[0], poly[j], poly[j + 1]])
                tuv.append(uvs[c] if uvs is not None else np.zeros((3, 2)))
                tn.append(nrm[c] if nrm is not None else np.zeros((3, 3)))
                tm.append(poly_mat[pi])
            corner += n

        tris = np.array(tris, np.int32) + vbase
        tri_uv = np.array(tuv, np.float32)
        tri_n = np.array(tn, np.float32)
        tri_mat = np.array(tm, np.int32)
        if nrm is None:            # flat normals from the geometry itself
            fn = np.cross(Vw[tris[:, 1] - vbase] - Vw[tris[:, 0] - vbase],
                          Vw[tris[:, 2] - vbase] - Vw[tris[:, 0] - vbase])
            tri_n = np.repeat(fn[:, None, :], 3, 1)
        tri_n = tri_n @ AXIS[:3, :3]      # same space as the points

        pts_all.append(Vw)
        tris_all.append(tris)
        uv_all.append(tri_uv)
        nrm_all.append(tri_n)
        mat_all.append(tri_mat)

        weights, _skin_uid = scene.skin_clusters(geo.props[0])
        if not weights:
            warnings.append("%s: no skin clusters, rigidly bound to hips" % mesh_name)
        geo_info.append((geo.props[0], mesh_name, vbase, len(V), weights))
        vbase += len(V)

    assert pts_all, "no geometry found"
    PTS = np.concatenate(pts_all) * S
    TRIS = np.concatenate(tris_all)
    TUV = np.concatenate(uv_all)
    TN = np.concatenate(nrm_all)
    TM = np.concatenate(mat_all)
    nv = len(PTS)
    print("merged: %d verts, %d tris, %d materials" % (nv, len(TRIS), len(mat_table)))

    # -------------------------------------------------------------- bind pose
    B = scene.bind_pose()
    print("bind pose joints:", len(B), "| duplicate-pose max dev: %.4f" % scene.bind_dup_err)
    if scene.bind_dup_err > 1e-3:
        warnings.append("bind poses disagree between Pose nodes (%.4f)" % scene.bind_dup_err)

    CP = {k: (v @ AXIS)[3, :3] * S for k, v in B.items()}   # Mixamo bone position, metres

    # Some Mixamo exports store the bone offsets *only* in the Pose nodes, and a few of
    # them are scrambled (bone names point at another bone's matrix).  There is nothing
    # to fit in that case, so say so instead of writing a broken USDZ.
    _h, _hd, _lf = CP.get("Hips"), CP.get("Head"), CP.get("LeftFoot")
    if _h is not None and _hd is not None:
        _ys = [v[1] for v in CP.values()]
        _tall = max(_ys) - min(_ys)
        weird = (_hd[1] < _h[1] + 0.2 * _tall)
        if _lf is not None:
            weird = weird or _lf[1] > _h[1] - 0.2 * _tall
        if weird:
            raise SystemExit("FATAL: %s: the bind pose does not describe a standing "
                             "skeleton (head at y=%.2f, hips %.2f, left foot %.2f) - the "
                             "FBX Pose nodes are inconsistent, re-export it from Mixamo"
                             % (slug, _hd[1], _h[1], _lf[1] if _lf is not None else -1))

    def mirror_side(name):
        if name.startswith("Right"):
            return "Left" + name[5:]
        if name.startswith("Left"):
            return "Right" + name[4:]
        return None

    def cget(name):
        if name in CP:
            return CP[name].copy()
        alt = mirror_side(name)
        if alt and alt in CP:
            v = CP[alt].copy()
            v[0] = -v[0]
            warnings.append("%s missing in bind pose -> mirrored from %s" % (name, alt))
            return v
        raise SystemExit("FATAL: bones missing from bind pose: %s" % name)

    def tip_from_mesh(group_bone, base, direction, pct=99.5):
        """furthest mesh vertex of a bone group along `direction` (character-specific tip)"""
        d = np.linalg.norm(direction)
        if d < 1e-9:
            return None
        direction = direction / d
        for (_, _, v0, count, wts) in geo_info:
            d_ = wts.get(group_bone)
            if not d_:
                continue
            idx = [vi for vi, w in d_.items() if w > 0.5] or list(d_.keys())
            if len(idx) < 4:
                continue
            P = PTS[v0 + np.array(idx)]
            s = (P - base) @ direction
            return base + direction * float(np.percentile(s, pct))
        return None

    # -------------------------------------------------------------- joint fit
    char = {k: cget(k) for k in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")}
    for side, P in SIDE.items():
        for n in BODY:
            char[P + n] = cget(P + n)
        for an, mix in FINGERS:
            for i in (1, 2, 3):
                char[P + mix + str(i)] = cget(P + mix + str(i))

    # ARKit drives joints by NAME, so a character baked facing -Z (or with its left
    # side on -X) has to be rotated before anything is measured from it.
    char, PTS, align_notes = fit.align_to_arkit(char, PTS)
    for note in align_notes:
        warnings.append(note)
        print("NOTE:", note)

    for side, P in SIDE.items():
        for an, mix in FINGERS:
            base = char[P + mix + "3"]
            prev = char[P + mix + "2"]
            tip = tip_from_mesh(P + mix + "3", base, base - prev)
            if tip is None:
                tip = base + (base - prev) * 0.35
                warnings.append("%s %s tip extrapolated (no mesh)" % (side, an))
            char[side + "_tip_" + an] = tip
        base = char[P + "ToeBase"]
        d = base - char[P + "Foot"]
        d = np.array([d[0], 0.0, d[2]])
        tip = tip_from_mesh(P + "ToeBase", base, d)
        if tip is None:
            tip = base + d * 0.5
            warnings.append("%s toe tip extrapolated (no mesh)" % side)
        char[side + "_tip_Toe"] = tip

    height = float(PTS[:, 1].max() - PTS[:, 1].min())
    pos, D = fit.build_target(char, ref, height_scale=height / ref.height)
    missing = [j for j in ARKIT if j not in pos]
    assert not missing, missing
    print("joints placed: %d (reference directions, character segment lengths)" % len(pos))

    # -------------------------------------------------------------- weight remap
    WMAP = fit.weight_joint_map()

    bone_names = sorted({b for (_, _, _, _, w) in geo_info for b in w})
    BIDX = {b: i for i, b in enumerate(bone_names)}
    WB = np.zeros((nv, len(bone_names)), np.float32)
    for (_, _, v0, count, weights) in geo_info:
        for bone, d in weights.items():
            if not d:
                continue
            vi = np.fromiter(d.keys(), np.int64, len(d))
            WB[v0 + vi, BIDX[bone]] = np.fromiter(d.values(), np.float32, len(d))
    WB /= np.maximum(WB.sum(1, keepdims=True), 1e-8)

    # UV / material seams are duplicated vertices: if their weights differ the mesh
    # tears open the moment the rest pose changes.  Give every coincident vertex group
    # the same (averaged) weights.
    PTS0 = PTS.copy()
    key = np.round(PTS0 * 10000).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    cnt = np.maximum(np.bincount(inv).astype(np.float32), 1.0)
    acc = np.zeros((len(cnt), WB.shape[1]), np.float32)
    for c in range(WB.shape[1]):
        acc[:, c] = np.bincount(inv, weights=WB[:, c], minlength=len(cnt))
    WB = (acc / cnt[:, None])[inv]

    # Mixamo's blends are only a few vertices wide, so re-fitting the mesh into the
    # reference T-pose concentrates the whole segment correction on one edge ring.
    # Blur the weights along the edges first: it spreads the correction over a band
    # (and evens out the left/right differences on the way).
    E = np.concatenate([TRIS[:, [0, 1]], TRIS[:, [1, 2]], TRIS[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]])
    EDEG = np.maximum(np.bincount(E[:, 0], minlength=nv).astype(np.float32), 1.0)
    for _ in range(SMOOTH_PASSES):
        acc = np.zeros_like(WB)
        for c in range(WB.shape[1]):
            acc[:, c] = np.bincount(E[:, 0], weights=WB[E[:, 1], c], minlength=nv)
        WB = (1.0 - SMOOTH_K) * WB + SMOOTH_K * (acc / EDEG[:, None])
    WB[WB < 0.005] = 0.0
    WB /= np.maximum(WB.sum(1, keepdims=True), 1e-8)

    # Mixamo often weights the two halves of a body very differently (here: 1269 vs 306
    # total weight on the arm bone), so the same tracked pose deforms them differently.
    if SYM_WEIGHTS:
        WB, nmoved = fit.mirror_weights(PTS, WB, bone_names)
        print("symmetric weights: %d right-side vertices took the mirrored left weights" % nmoved)

    # ---- straighten the character into the reference T-pose (bones stay in the flesh)
    TF = fit.bone_transforms(char, pos, D, WMAP)
    PTS = fit.deform(PTS, WB, bone_names, TF)

    Rv = np.zeros((nv, 3, 3))
    for bi, name in enumerate(bone_names):
        t = TF.get(name)
        if t is None:
            continue
        idx = np.nonzero(WB[:, bi] > 0)[0]
        Rv[idx] += t[0] * WB[idx, bi][:, None, None]
    TN = np.einsum("nkij,nkj->nki", Rv[TRIS], TN)
    TN /= np.maximum(np.linalg.norm(TN, axis=2, keepdims=True), 1e-8)
    disp = np.linalg.norm(PTS - PTS0, axis=1)
    e0 = np.linalg.norm(PTS0[TRIS[:, 0]] - PTS0[TRIS[:, 1]], axis=1)
    e1 = np.linalg.norm(PTS[TRIS[:, 0]] - PTS[TRIS[:, 1]], axis=1)
    vis = e0 > 0.002                      # sub-2 mm slivers are seam stitches
    ratio = e1 / np.maximum(e0, 1e-6)
    print("T-pose re-fit: max vertex move %.1f mm, mean %.1f mm" % (disp.max() * 1000, disp.mean() * 1000))
    if vis.any():
        k = int(np.argmax(ratio * vis))
        print("visible edge stretch: p99.9 %.2f, max %.2f (worst edge %.1f mm -> %.1f mm)"
              % (float(np.percentile(ratio[vis], 99.9)), float(ratio[vis].max()),
                 e0[k] * 1000, e1[k] * 1000))
        if ratio[vis].max() > 3.5 and (e1[k] - e0[k]) > 0.02:
            warnings.append("T-pose re-fit stretches a visible edge x%.2f (%.0f mm)"
                            % (float(ratio[vis].max()), (e1[k] - e0[k]) * 1000))

    # ---- to ARKit joints (dense, so the weight repair below stays readable)
    JIDX = {j: i for i, j in enumerate(ARKIT)}
    DENSE = np.zeros((nv, len(ARKIT)), np.float32)
    unmapped, dropped = [], 0
    for bi, name in enumerate(bone_names):
        j = WMAP.get(name)
        if j is None:
            unmapped.append(name)
            dropped += int((WB[:, bi] > 0).sum())
            continue
        DENSE[:, JIDX[j]] += WB[:, bi]
    if unmapped:
        warnings.append("bones without ARKit target: %s" % ", ".join(sorted(unmapped)))

    # ---- P0-2: continuous spine / neck weights (was: 11 of 12 torso joints at 0)
    torso = np.array([JIDX[j] for j in fit.TORSO_CHAIN])
    torso_total = DENSE[:, torso].sum(1)
    DENSE[:, torso] = 0.0
    KW = fit.torso_kernel(PTS, pos)
    DENSE[:, torso] = KW * torso_total[:, None]

    # ---- P0-1: hand the distal finger phalanx to _3 (was: _3 and End both at 0).
    # The thumb has no _3 in ARKit's skeleton (Start/_1/_2/End), so it is skipped.
    for side in ("left", "right"):
        for an, _ in FINGERS[:4]:
            j2 = JIDX[side + "_hand%s_2_joint" % an]
            j3 = JIDX[side + "_hand%s_3_joint" % an]
            w2 = DENSE[:, j2].copy()
            move = w2 * fit.tip_ramp(PTS, pos, ARKIT[j2], ARKIT[j3])
            DENSE[:, j2] -= move
            DENSE[:, j3] += move

    jidx, jw, empty = fit.top_influences(DENSE, 4)
    if empty.any():
        warnings.append("%d vertices had no weights -> hips_joint" % int(empty.sum()))
    if dropped:
        warnings.append("%d bone influences dropped (no ARKit joint)" % dropped)
    # ---- QC: where the re-fit moved the mesh most, and which edges suffered
    dom = jidx[np.arange(nv), np.argmax(jw, 1)]
    worst = {}
    for v in np.argsort(-disp)[:3000]:
        n = ARKIT[dom[v]]
        worst[n] = max(worst.get(n, 0.0), float(disp[v]) * 1000)
    print("re-fit moved most: " + ", ".join("%s %.0fmm" % (k, v) for k, v in
                                            sorted(worst.items(), key=lambda kv: -kv[1])[:5]))

    present = np.zeros(len(ARKIT), bool)
    present[jidx[jw > 0]] = True
    wcount = present.astype(np.int32)
    print("weighted joints: %d / %d; max influences: %d" %
          (int(present.sum()), len(ARKIT), int((jw > 0).sum(1).max())))

    # -------------------------------------------------------------- alignment report
    rows = fit.compare(pos, ref, height_scale=height / ref.height)
    dmax = max(rows, key=lambda r: r[1])
    pmax = max(rows, key=lambda r: r[2])
    print("alignment: max bone-direction deviation %.2f deg (%s), max position deviation %.1f mm (%s)"
          % (dmax[1], dmax[0], pmax[2], pmax[0]))
    bad_dir = [r for r in rows if r[1] > 4.0]
    if bad_dir:
        warnings.append("bone directions still off: %s" %
                        ", ".join("%s %.1f deg" % (r[0], r[1]) for r in bad_dir[:6]))

    # -------------------------------------------------------------- write
    used_mats = sorted(set(TM.tolist()))
    TM2 = np.searchsorted(np.array(used_mats), TM).astype(np.int32)
    mats_used = [mat_table[i] for i in used_mats]
    for i, info in enumerate(mats_used):
        info["material_id"] = i

    np.savez_compressed(os.path.join(out, "mesh.npz"),
                        points=PTS.astype(np.float32), tris=TRIS.astype(np.int32),
                        tri_uv=TUV.astype(np.float32), tri_normals=TN.astype(np.float32),
                        tri_mat=TM2, joint_idx=jidx, joint_w=jw)

    meta = {"model": slug, "joints": ARKIT,
            "joint_positions": {j: [float(x) for x in pos[j]] for j in ARKIT},
            "materials": mats_used,
            "counts": {"verts": int(nv), "tris": int(len(TRIS)),
                       "geometries": len(geo_info)},
            "unit_scale": S, "warnings": warnings,
            "fit": {"max_dir_dev_deg": round(float(dmax[1]), 2), "max_dir_dev_joint": dmax[0],
                    "max_pos_dev_mm": round(float(pmax[2]), 1), "max_pos_dev_joint": pmax[0],
                    "ref_height": round(ref.height, 4), "char_height": round(height, 4),
                    "tp_pose_max_move_mm": round(float(disp.max() * 1000), 1),
                    "weighted_joints": int((DENSE > 1e-6).any(0).sum()),
                    "zero_weight_joints": [j for i, j in enumerate(ARKIT) if wcount[i] == 0]}}
    with open(os.path.join(out, "mesh_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print("COUNT", meta["counts"])
    print("bbox", np.round(PTS.min(0), 3), np.round(PTS.max(0), 3))
    print("height %.3f m  arm span %.3f m" % (height, float(PTS[:, 0].max() - PTS[:, 0].min())))
    print("zero-weight joints (%d): %s" % (len(meta["fit"]["zero_weight_joints"]),
                                           ", ".join(meta["fit"]["zero_weight_joints"])))
    for w in warnings:
        print("WARN:", w)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FBX -> mesh + ARKit joint fit")
    ap.add_argument("slugs", nargs="*", help="model slugs (default: every .fbx in models/)")
    a = ap.parse_args(argv)
    slugs = a.slugs or config.all_models()
    if not slugs:
        raise SystemExit("no models: put the Mixamo .fbx files in %s" % config.MODELS_DIR)
    for s in slugs:
        print("== extract_fbx", s)
        extract(s)


if __name__ == "__main__":
    main()
