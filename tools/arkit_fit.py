"""
Fit a Mixamo character's rest pose to Apple's ARKit reference skeleton.

Why this module exists
----------------------
RealityKit does not retarget: ARKit's joint transforms are applied to the joints with
the same name, so the rest pose of the model has to look like Apple's
`biped_robot.usdz` (ARSkeletonDefinition.defaultBody3D).  Two things matter:

  * joint ORIENTATION - already copied 1:1 from the reference (build_usd.py)
  * joint DIRECTION   - the offset of a joint from its parent, expressed in the
                        parent's frame.  If the character's shin points 9 degrees
                        outwards, every tracked leg pose is 9 degrees off.

Bone LENGTH stays the character's own: the reference is a robot with very long legs,
neck and fingers, so copying its segment lengths would stretch any human mesh
(thigh+shin are 878 mm there, i.e. 51 % of its height).

So: keep every Mixamo segment length, rotate each segment so its direction equals the
reference direction, and apply exactly the same rotations to the mesh (linear blend
skinning with the rest-pose *delta* of each bone).  The character is thereby
straightened into a clean T-pose instead of the skeleton sliding out of the flesh.

Also collected here: the skin weight repair that Apple's validation implies
(continuous spine / neck weights, a usable distal finger phalanx).
"""
import json, math, os
import numpy as np

EPS = 1e-9

# Apple's reference has 4 phalanges per finger; a Mixamo hand has 3 bones plus the mesh
# tip.  _3 is placed at the mesh tip, so `End` is the leaf Apple expects to be
# 18..20 mm long (robot with unusually long fingers) - scaled to a human hand that is
# 12..18 mm.  It carries no weight, it only has to exist with a non-zero length.
TIP_LEN = {"Thumb": 0.014, "Index": 0.016, "Mid": 0.018, "Ring": 0.016, "Pinky": 0.012}

FINGERS = [("Index", "HandIndex"), ("Mid", "HandMiddle"), ("Ring", "HandRing"),
           ("Pinky", "HandPinky")]
# Mixamo fingers incl. the thumb (arkit_fit only chains the four fingers above; the
# thumb needs its own map because ARKit's thumb has Start/_1/_2 but no _3)
FINGERS_ALL = FINGERS + [("Thumb", "HandThumb")]

SIDE = {"left": "Left", "right": "Right"}
BODY = ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase")

# A 3-segment Mixamo spine is resampled into Apple's 7 spine joints.  Fractions of the
# way from one Mixamo bone to the next (reference proportions mapped onto the
# character's own segments, so no Mixamo bone changes length).
SPINE_SEQ = [
    ("spine_2_joint", "Hips", "Spine", 1.00),
    ("spine_3_joint", "Spine", "Spine1", 0.50),
    ("spine_4_joint", "Spine", "Spine1", 1.00),
    ("spine_5_joint", "Spine1", "Spine2", 0.50),
    ("spine_6_joint", "Spine1", "Spine2", 0.75),
    ("spine_7_joint", "Spine1", "Spine2", 1.00),
    ("neck_1_joint", "Neck", "Neck", 0.00),
]
# Same idea for the neck: the reference keeps 70 % of neck+head below head_joint.
NECK_SEQ = [("neck_2_joint", 0.19), ("neck_3_joint", 0.45),
            ("neck_4_joint", 0.70), ("head_joint", 1.00)]

# per-chain direction snap strength (0 = keep the character's own direction)
DEFAULT_ALPHA = {"shoulder": 1.0, "spine": 1.0, "neck": 1.0, "leg": 1.0,
                 "foot": 1.0, "toe": 1.0, "arm": 1.0, "finger": 1.0}

TORSO_CHAIN = (["hips_joint"] + ["spine_%d_joint" % k for k in range(1, 8)]
               + ["neck_%d_joint" % k for k in range(1, 5)] + ["head_joint"])


def _seg_tail():
    """Mixamo bone -> the joint at the FAR end of that bone's own segment.

    build_target stores on joint j the rotation of the segment that *ends* at j
    (D[j] = rot_between(tail - head, ref.dir[j])), so a bone's mesh region has to use
    D of the joint at its own far end: the thigh (LeftUpLeg -> left_leg_joint), the
    pelvis (Hips -> spine_2_joint, spine_1 overlaps the hips), and so on.

    Using the near end instead (D of the joint the bone is mapped to) applies the
    PREVIOUS segment's rotation: the pelvis -> hip offset is 19 deg off the reference
    and rotating the whole thigh by it swings both legs 19 deg about +Z, which bows
    them outwards and makes the knee miss pos[left_leg_joint] by 90 mm.
    """
    t = {"Hips": "spine_2_joint", "Spine": "spine_3_joint", "Spine1": "spine_5_joint",
         "Spine2": "neck_1_joint", "Neck": "neck_2_joint", "Head": "head_joint"}
    for side, P in (("left", "Left"), ("right", "Right")):
        t[P + "Shoulder"] = side + "_arm_joint"
        t[P + "Arm"] = side + "_forearm_joint"
        t[P + "ForeArm"] = side + "_hand_joint"
        t[P + "Hand"] = side + "_handIndexStart_joint"   # palm follows the index ray
        t[P + "UpLeg"] = side + "_leg_joint"
        t[P + "Leg"] = side + "_foot_joint"
        t[P + "Foot"] = side + "_toes_joint"
        t[P + "ToeBase"] = side + "_toesEnd_joint"
        t[P + "HandThumb1"] = side + "_handThumb_1_joint"
        t[P + "HandThumb2"] = side + "_handThumb_2_joint"
        t[P + "HandThumb3"] = side + "_handThumb_2_joint"   # no _3 joint for the thumb
        for an, mix in FINGERS:
            t[P + mix + "1"] = side + "_hand%s_1_joint" % an
            t[P + mix + "2"] = side + "_hand%s_2_joint" % an
            t[P + mix + "3"] = side + "_hand%s_3_joint" % an
    return t


SEG_TAIL = _seg_tail()

FACE_ROOT = ("head_joint", "jaw_joint", "left_eye_joint", "right_eye_joint")


# ---------------------------------------------------------------- small maths
def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros(3)


def lerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * t


def slerp(a, b, t):
    a, b = unit(a), unit(b)
    if t >= 1.0:
        return b
    d = float(np.clip(a @ b, -1.0, 1.0))
    if d > 0.9995:
        return unit(lerp(a, b, t))
    th = math.acos(d) * t
    perp = unit(b - a * d)
    return a * math.cos(th) + perp * math.sin(th)


def rot_between(a, b):
    """3x3 rotation taking direction a to direction b (minimal rotation)."""
    a, b = unit(a), unit(b)
    if np.linalg.norm(a) < 0.5 or np.linalg.norm(b) < 0.5:
        return np.eye(3)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = unit(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = unit(np.cross(a, [0.0, 1.0, 0.0]))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def angle_between(a, b):
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------- reference
class Ref:
    """Apple's biped_robot.usdz skeleton (91 joints, metres, +Y up, +Z forward)."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise SystemExit("missing reference skeleton %s\n"
                             "  run: python tools/dump_robot_skeleton.py" % path)
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("units", "m") != "m":
            # a 100x unit slip here silently collapses the face joints, so refuse
            raise SystemExit("%s is in %r, not metres - rerun tools/dump_robot_skeleton.py"
                             % (path, d.get("units")))
        self.joints = d["joints"]
        self.parents = d["parents"]
        self.parent = dict(zip(self.joints, self.parents))
        B = np.array(d["bind_world"], float)
        self.bind_world = B
        self.pos = {n: B[i][3, :3].copy() for i, n in enumerate(self.joints)}
        self.dir = {}
        for n in self.joints:
            p = self.parent[n]
            self.dir[n] = unit(self.pos[n] - self.pos[p]) if p else np.zeros(3)
        ys = [self.pos[n][1] for n in self.joints]
        self.height = float(max(ys) - min(ys))
        self.hips = self.pos["hips_joint"].copy()

    def bone_len(self, j):
        p = self.parent[j]
        return float(np.linalg.norm(self.pos[j] - self.pos[p])) if p else 0.0


# ---------------------------------------------------------------- mixamo -> arkit maps
def weight_joint_map():
    """Mixamo bone -> the ARKit joint whose rotation should drive that bone's region.

    One Mixamo bone per skin cluster, so this is also the joint the weights land on
    before the torso / finger repairs in extract_fbx.py spread them out.
    """
    m = {"Hips": "hips_joint", "Spine": "spine_2_joint", "Spine1": "spine_4_joint",
         "Spine2": "spine_7_joint", "Neck": "neck_1_joint", "Head": "head_joint"}
    for side, P in (("left", "Left"), ("right", "Right")):
        m[P + "Shoulder"] = side + "_shoulder_1_joint"
        m[P + "Arm"] = side + "_arm_joint"
        m[P + "ForeArm"] = side + "_forearm_joint"
        m[P + "Hand"] = side + "_hand_joint"
        m[P + "UpLeg"] = side + "_upLeg_joint"
        m[P + "Leg"] = side + "_leg_joint"
        m[P + "Foot"] = side + "_foot_joint"
        m[P + "ToeBase"] = side + "_toes_joint"
        m[P + "HandThumb1"] = side + "_handThumbStart_joint"
        m[P + "HandThumb2"] = side + "_handThumb_1_joint"
        m[P + "HandThumb3"] = side + "_handThumb_2_joint"   # ARKit's thumb has no _3
        for an, mix in FINGERS:
            m[P + mix + "1"] = side + "_hand%sStart_joint" % an
            m[P + mix + "2"] = side + "_hand%s_1_joint" % an
            m[P + mix + "3"] = side + "_hand%s_2_joint" % an
    return m


# ---------------------------------------------------------------- axis / facing sanity
def _yaw180(points):
    """rotate 180 deg about +Y (a proper rotation: handedness and winding survive)"""
    p = np.array(points, float, copy=True)
    p[..., 0] *= -1.0
    p[..., 2] *= -1.0
    return p


def align_to_arkit(char, points, tol=0.02):
    """Make the character face +Z with its left side on +X, as ARKit expects.

    RealityKit does not retarget: it applies ARKit's joint transforms to the joints
    *by name*, so a character baked facing -Z is driven backwards and one with its
    left hand on -X comes out mirrored.  Both are read from the bind pose itself
    (toes vs feet, left vs right hip) instead of the FBX FrontAxis annotation, which
    exporters routinely get wrong.

    A consistent "facing -Z, left on -X" pose is fixed with a 180 deg yaw of both the
    mesh and the bone positions (exact, no re-weighting needed).  An inconsistent one
    (e.g. facing -Z but left on +X) cannot be fixed by a rotation and is reported.

    returns (char, points, notes)
    """
    notes = []
    if not all(k in char for k in ("LeftUpLeg", "RightUpLeg")):
        return char, points, ["no left/right hips in the bind pose - orientation not checked"]
    side_dx = float(char["LeftUpLeg"][0] - char["RightUpLeg"][0])
    fwd = np.zeros(3)
    if all(k in char for k in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")):
        for P in ("Left", "Right"):
            d = np.asarray(char[P + "ToeBase"], float) - np.asarray(char[P + "Foot"], float)
            fwd += d
        fwd[1] = 0.0
    facing = float(fwd[2]) if np.linalg.norm(fwd) > 1e-6 else float("nan")

    left_ok = side_dx > tol
    face_ok = math.isnan(facing) or facing > tol
    if left_ok and face_ok:
        return char, points, []
    if (not left_ok) and (not face_ok) and not math.isnan(facing):
        pt = np.array(points, float, copy=True)
        pt = _yaw180(pt)
        ch = {k: _yaw180(v).ravel()[:3] for k, v in char.items()}
        notes.append("character was facing -Z with its left side on -X: rotated 180 deg "
                     "about +Y to match ARKit")
        return ch, pt, notes
    if not left_ok:
        notes.append("LEFT/RIGHT MIRRORED: left bones sit on -X (dx=%.3f m) - the named "
                     "ARKit joints will be driven by the wrong side of the body" % side_dx)
    if not face_ok:
        notes.append("character faces -Z (toes point to %.3f) - ARKit drives it backwards" % facing)
    return char, points, notes


# ---------------------------------------------------------------- weights helpers
def top_influences(dense, k=4):
    """(nv, njoints) dense weights -> (nv, k) joint indices / normalised weights.

    Vertices with no weight at all fall back to joint 0 with weight 1 so the USDZ
    never ships an unweighted (exploding) vertex.
    """
    nv, nj = dense.shape
    k = max(1, min(k, nj))
    order = np.argsort(-dense, axis=1)[:, :k].astype(np.int32)
    w = np.take_along_axis(dense, order, 1).astype(np.float32)
    empty = w.sum(1) <= 0
    order[empty] = 0
    w[empty] = 0.0
    w[empty, 0] = 1.0
    s = np.maximum(w.sum(1, keepdims=True), 1e-8)
    return order, (w / s).astype(np.float32), empty


# ---------------------------------------------------------------- fit
def _step(P, D, ref, j, parent, anchor, cprev, alpha):
    """Place joint j: reference direction, |anchor - cprev| length.

    D[j] is the *minimal* rotation that turns the joint's own rest axis onto the new
    one.  Accumulating the parent rotations instead would apply every parent rotation
    about the *child* joint (a 19 deg hip snap becomes a 150 mm swing of the thigh),
    while the joint positions already carry the chain: bone regions translate with
    their joint and only twist by the few degrees their own axis actually changes.
    """
    cdir = np.asarray(anchor, float) - np.asarray(cprev, float)
    length = float(np.linalg.norm(cdir))
    rd = ref.dir[j]
    if length <= EPS:
        P[j] = P[parent].copy()
        D[j] = D[parent].copy()
        return
    u = unit(cdir)
    d = u if alpha <= 0 else (unit(rd) if alpha >= 1.0 else unit(slerp(u, rd, alpha)))
    P[j] = P[parent] + d * length
    D[j] = rot_between(u, d)


def build_target(char, ref, alpha=None, height_scale=1.0):
    """
    char: Mixamo bind pose in metres / ARKit axes.  Keys:
          Hips, Spine, Spine1, Spine2, Neck, Head,
          <Left|Right>_Shoulder / _Arm / _ForeArm / _Hand / _UpLeg / _Leg / _Foot /
          _ToeBase / _HandIndex1..3 (Middle, Ring, Pinky) / _HandThumb1..3,
          <left|right>_tip_<Index|Mid|Ring|Pinky|Toe>  (mesh-detected tip)
    returns (pos: 91 joint positions, D: accumulated rotation per joint)
    """
    a = dict(DEFAULT_ALPHA)
    a.update(alpha or {})
    P, D = {}, {}
    for j in ("root", "hips_joint"):
        P[j] = np.array(char["Hips"], float)
        D[j] = np.eye(3)
    P["spine_1_joint"] = P["hips_joint"].copy()      # Apple: spine_1 overlaps hips
    D["spine_1_joint"] = D["hips_joint"].copy()

    # ---- spine (spine_7 sits on the Mixamo chest bone, which is where the clavicles
    # attach, so the shoulders stay in the flesh instead of floating 6 cm up)
    prev = np.asarray(char["Hips"], float)
    for j, k0, k1, f in SPINE_SEQ:
        anchor = lerp(char[k0], char[k1], f)
        _step(P, D, ref, j, ref.parent[j], anchor, prev, a["spine"])
        prev = anchor

    # ---- neck + head (one Mixamo segment, the reference keeps 4 joints inside it)
    cprev = char["Neck"]
    for j, f in NECK_SEQ:
        anchor = lerp(char["Neck"], char["Head"], f)
        _step(P, D, ref, j, ref.parent[j], anchor, cprev, a["neck"])
        cprev = anchor

    # ---- face joints: reference offsets from head_joint, scaled with the character
    hd = P["head_joint"]
    for j in ref.joints:
        if j in P:
            continue
        if ref.parent.get(j) in FACE_ROOT:
            P[j] = hd + (ref.pos[j] - ref.pos["head_joint"]) * height_scale
            D[j] = D["head_joint"].copy()

    # ---- limbs
    for side, S in (("left", "Left"), ("right", "Right")):
        _step(P, D, ref, side + "_shoulder_1_joint", "spine_7_joint",
              char[S + "Shoulder"], char["Spine2"], a["shoulder"])
        _step(P, D, ref, side + "_arm_joint", side + "_shoulder_1_joint",
              char[S + "Arm"], char[S + "Shoulder"], a["arm"])
        _step(P, D, ref, side + "_forearm_joint", side + "_arm_joint",
              char[S + "ForeArm"], char[S + "Arm"], a["arm"])
        _step(P, D, ref, side + "_hand_joint", side + "_forearm_joint",
              char[S + "Hand"], char[S + "ForeArm"], a["arm"])

        _step(P, D, ref, side + "_upLeg_joint", "hips_joint",
              char[S + "UpLeg"], char["Hips"], a["leg"])
        _step(P, D, ref, side + "_leg_joint", side + "_upLeg_joint",
              char[S + "Leg"], char[S + "UpLeg"], a["leg"])
        _step(P, D, ref, side + "_foot_joint", side + "_leg_joint",
              char[S + "Foot"], char[S + "Leg"], a["foot"])
        _step(P, D, ref, side + "_toes_joint", side + "_foot_joint",
              char[S + "ToeBase"], char[S + "Foot"], a["toe"])
        _step(P, D, ref, side + "_toesEnd_joint", side + "_toes_joint",
              char[side + "_tip_Toe"], char[S + "ToeBase"], a["toe"])

        # fingers: Start/_1/_2 follow the Mixamo bones, _3 sits on the mesh tip, End is
        # the leaf phalanx the reference has and Mixamo does not.
        for an, mix in FINGERS:
            chain = [(side + "_hand%sStart_joint" % an, S + mix + "1", S + "Hand"),
                     (side + "_hand%s_1_joint" % an, S + mix + "2", S + mix + "1"),
                     (side + "_hand%s_2_joint" % an, S + mix + "3", S + mix + "2")]
            for j, key, pkey in chain:
                _step(P, D, ref, j, ref.parent[j], char[key], char[pkey], a["finger"])
            _step(P, D, ref, side + "_hand%s_3_joint" % an, side + "_hand%s_2_joint" % an,
                  char[side + "_tip_" + an], char[S + mix + "3"], a["finger"])
            je = side + "_hand%sEnd_joint" % an
            P[je] = P[side + "_hand%s_3_joint" % an] + ref.dir[je] * TIP_LEN[an]
            D[je] = D[side + "_hand%s_3_joint" % an].copy()

        chain = [(side + "_handThumbStart_joint", S + "HandThumb1", S + "Hand"),
                 (side + "_handThumb_1_joint", S + "HandThumb2", S + "HandThumb1"),
                 (side + "_handThumb_2_joint", S + "HandThumb3", S + "HandThumb2")]
        for j, key, pkey in chain:
            _step(P, D, ref, j, ref.parent[j], char[key], char[pkey], a["finger"])
        je = side + "_handThumbEnd_joint"
        P[je] = P[side + "_handThumb_2_joint"] + ref.dir[je] * TIP_LEN["Thumb"]
        D[je] = D[side + "_handThumb_2_joint"].copy()

    return P, D


# ---------------------------------------------------------------- mesh re-fit
def bone_transforms(char, pos, D, wmap, seg_tail=None):
    """Rigid rest-pose delta per Mixamo bone: p' = pos[j] + R @ (p - char[bone]).

    R is the rotation of the bone's OWN segment (see SEG_TAIL), so the bone's head lands
    on pos[j] and its tail on the fitted child joint: the region keeps its length and two
    neighbours agree on the joint between them instead of shearing it open.
    """
    seg = SEG_TAIL if seg_tail is None else seg_tail
    out = {}
    for bone, j in wmap.items():
        if bone in char and j in pos:
            R = D.get(seg.get(bone, j))
            if R is None:
                R = D[j]
            out[bone] = (np.asarray(R, float), np.asarray(char[bone], float),
                         np.asarray(pos[j], float))
    return out


def deform(points, W, bone_names, transforms):
    """LBS with the rest-pose delta of every bone: the mesh follows the straightened skeleton.

    W is a dense (nverts, nbones) weight matrix; bones without a transform keep their
    vertices in place (and are renormalised out).
    """
    pts = np.asarray(points, float)
    out = np.zeros_like(pts)
    tot = np.zeros(len(pts))
    for bi, name in enumerate(bone_names):
        t = transforms.get(name)
        if t is None:
            continue
        idx = np.nonzero(W[:, bi] > 0)[0]
        if not len(idx):
            continue
        R, p0, p1 = t
        w = W[idx, bi][:, None]
        out[idx] += ((pts[idx] - p0) @ R.T + p1) * w
        tot[idx] += w[:, 0]
    ok = tot > EPS
    out[ok] /= tot[ok, None]
    out[~ok] = pts[~ok]
    return out


# ---------------------------------------------------------------- weights
def chain_arc(pos):
    """torso chain polyline and the arc-length coordinate of every joint on it"""
    pts = np.array([pos[j] for j in TORSO_CHAIN], float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, np.concatenate([[0.0], np.cumsum(seg)])


def project_arc(points, pts, arc):
    """arc coordinate of every vertex along the chain.

    Nearest-point-on-polyline (the old version) is not continuous: front-of-chest
    vertices sit ~100 mm away from a nearly vertical spine, so two segments are equally
    close and a 6 mm vertex move flips the nearest segment -- the arc then jumps 40 mm,
    neighbours land on different joints and the skin tears when that joint rotates.
    Projecting along the chain axis is monotone and 1-Lipschitz in the position (a 6 mm
    move changes the arc by at most 6 mm), which is what the weight ramp needs.
    """
    P = np.asarray(points, float)
    pts = np.asarray(pts, float)
    axis = pts[-1] - pts[0]
    axis = axis / max(np.linalg.norm(axis), EPS)
    s = np.maximum.accumulate((pts - pts[0]) @ axis)      # hips/spine_1 coincide -> equal
    return np.interp((P - pts[0]) @ axis, s, arc)


def torso_kernel(points, pos, blur=2):
    """
    Continuous spine / neck weights.  Every torso joint gets a share of the vertex'
    torso weight according to where the vertex sits along the hips -> head chain, which
    replaces the old "one Mixamo bone per ARKit joint" mapping that left
    spine_1/3/5/7 and neck_1/3/4 at exactly zero weight.
    Returns (W[nv, len(TORSO_CHAIN)], joint names).
    """
    pts, arc = chain_arc(pos)
    t = project_arc(points, pts, arc)
    n = len(TORSO_CHAIN)
    W = np.zeros((len(points), n))
    k = np.clip(np.searchsorted(arc, t, side="left") - 1, 0, n - 2)
    for seg in np.unique(k):
        m = k == seg
        lo, hi = arc[seg], arc[seg + 1]
        if hi - lo < EPS:
            W[m, seg] += 1.0
        else:
            f = (t[m] - lo) / (hi - lo)
            W[m, seg] += 1.0 - f
            W[m, seg + 1] += f
    for _ in range(blur):                     # spread over +-2 joints, no hard steps
        # edge padding, NOT np.roll: rolling would wrap head_joint's weight into hips_joint
        Wp = np.pad(W, ((0, 0), (1, 1)), mode="edge")
        W = 0.25 * Wp[:, :-2] + 0.5 * Wp[:, 1:-1] + 0.25 * Wp[:, 2:]
    W /= np.maximum(W.sum(1, keepdims=True), EPS)
    return W


def tip_ramp(points, pos, start, end, lo=0.45):
    """0..1 ramp along the distal phalanx (start -> end), used to hand the fingertip to _3"""
    a = np.asarray(pos[start], float)
    b = np.asarray(pos[end], float)
    ab = b - a
    L2 = max(float(ab @ ab), EPS)
    t = ((np.asarray(points, float) - a) @ ab) / L2
    return smoothstep((t - lo) / (1.0 - lo))


def mirror_name(bone):
    if bone.startswith("Left"):
        return "Right" + bone[4:]
    if bone.startswith("Right"):
        return "Left" + bone[5:]
    return bone


def mirror_weights(points, W, bone_names, tol=0.03, blend=0.5):
    """
    Copy the left half's weights onto the right half (mirrored, with the Left/Right bone
    names swapped and mixed in with `blend`).  Mixamo characters are often weighted very
    unevenly from side to side (this one: 1269 vs 306 total on the arm bone), so the same
    tracked pose deforms the two arms by different amounts.
    Only vertices whose mirror partner is within `tol` are touched.
    """
    P = np.asarray(points, float)
    bi = {b: i for i, b in enumerate(bone_names)}
    left = np.nonzero(P[:, 0] > 0.001)[0]
    right = np.nonzero(P[:, 0] < -0.001)[0]
    if not len(left) or not len(right):
        return W, 0
    cell = max(tol, 0.005)
    key = lambda v: np.floor(v / cell).astype(np.int64)
    grid = {}
    for i in left:
        k = tuple(key(np.array([-P[i, 0], P[i, 1], P[i, 2]])))
        grid.setdefault(k, []).append(i)

    pairs = []
    for j in right:
        q = np.array([-P[j, 0], P[j, 1], P[j, 2]])
        k0 = key(q)
        best, bd = -1, tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in grid.get((k0[0] + dx, k0[1] + dy, k0[2] + dz), ()):
                        d = float(np.linalg.norm(np.array([-P[i, 0], P[i, 1], P[i, 2]]) - q))
                        if d < bd:
                            best, bd = i, d
        if best >= 0:
            pairs.append((j, best))
    if not pairs:
        return W, 0

    jj = np.array([p[0] for p in pairs])
    ii = np.array([p[1] for p in pairs])
    src = W[ii].copy()
    # swap the side in the bone names (columns): Left* -> Right*
    for b, col in bi.items():
        r = mirror_name(b)
        if r != b and r in bi and b.startswith("Left"):
            src[:, col], src[:, bi[r]] = W[ii, bi[r]].copy(), W[ii, col].copy()
    W[jj] = (1.0 - blend) * W[jj] + blend * src
    W /= np.maximum(W.sum(1, keepdims=True), EPS)
    return W, len(jj)


# ---------------------------------------------------------------- reporting
def compare(pos, ref, height_scale=1.0):
    """
    Direction deviation (deg) and position deviation (mm) of the produced skeleton
    against the reference scaled to the character.  Directions must be ~0; position
    residuals are length differences (the reference is a robot with much longer legs,
    neck and fingers) and are reported, not enforced.
    """
    rows = []
    hips = np.asarray(pos.get("hips_joint", ref.hips), float)
    for j in ref.joints:
        if j not in pos:
            continue
        p = ref.parent[j]
        if p is None:
            continue
        cd = pos[j] - pos[p]
        dd = 0.0
        if ref.bone_len(j) > 1e-6 and np.linalg.norm(cd) > 1e-6:
            dd = angle_between(cd, ref.pos[j] - ref.pos[p])
        target = hips + (ref.pos[j] - ref.hips) * height_scale
        rows.append((j, dd, float(np.linalg.norm(pos[j] - target)) * 1000.0))
    return rows
