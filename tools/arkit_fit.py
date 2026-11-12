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

Two chains are exempt, because Apple's reference is a robot whose *shape* is not a
human's and copying it reshapes the mesh rather than the tracking:

  * the hand (alpha 0): its fingers are a paddle, 10-40 degrees away from a hand's.
  * the leg chain (see DEFAULT_SAGITTAL / `in_plane`): only the sideways splay is
    straightened.  Its pitch is kept, because the robot stands with a 15 degree knee
    bend and an ankle -> ball-of-foot line 12 degrees shallower than a human foot's;
    snapping a human leg onto that lifts the whole foot off the ground (toes up) and
    adds a permanent bent knee to every tracked frame.

Also collected here: the skin weight repairs that Apple's validation implies
(continuous spine / neck weights, the metacarpal share of the palm, and the finger
joint mapping that used to be one joint too proximal).
"""
import json, math, os
import numpy as np

EPS = 1e-9

# ---- finger topology (read off Apple's biped_robot.usdz, not guessed)
# Apple's finger chain has FIVE joints and four bones:
#
#     Start  ->  _1   ->  _2   ->  _3   ->  End
#     metacarpal MCP     PIP     DIP      tip
#
# `Start` sits INSIDE the palm (index: 30 % of the way from the wrist to the knuckle)
# and Start->_1 - the metacarpal - is by far the longest bone of the finger
# (80 mm vs 47 / 26 / 26 mm on Apple's 1.71 m robot).  The tracked `left_handIndexStart`
# therefore lives near the wrist, not at the knuckle.
#
# A Mixamo hand has no metacarpal: HandIndex1..3 are the MCP / PIP / DIP phalanges and
# HandIndex4 is the (essentially unweighted) tip cap.  So 1..3 have to land on _1.._3 and
# the mesh tip on End.  Mapping them onto Start.._2 (one joint too proximal, which is what
# this file used to do) hands the knuckle flesh the tracked *metacarpal* transform and
# folds every finger into the palm - the "severely deformed hand" this was reported as.
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

# per-chain direction snap strength
#   1 = the segment is aimed at Apple's reference direction, 0 = the character's own
#   local direction is kept, so the segment is carried rigidly by its parent (no flesh
#   is sheared at all).
# Fingers stay at 0: Apple's reference hand is a robot *paddle* - its finger directions
# point 10-40 deg away from a human hand's - so snapping them fanned the fingers out and
# curled them into a claw.  A human hand is already the shape ARKit tracks, so there is
# nothing to straighten.
DEFAULT_ALPHA = {"shoulder": 1.0, "spine": 1.0, "neck": 1.0, "leg": 1.0,
                 "foot": 1.0, "toe": 1.0, "arm": 1.0, "finger": 0.0}

# Chains that keep the character's own STANCE instead of Apple's: only the sideways
# component of the segment is straightened and the pitch / forward lean stays exactly as
# the character has it (see `in_plane`).
#
# The leg chain belongs here.  Apple's reference is a robot that stands with a 15 deg
# knee bend (its shin leans 9 deg backwards) and with an ankle -> ball-of-foot line that
# is 12 deg shallower than a human foot's (the two feet are equally flat on the ground;
# Mixamo simply puts ToeBase deeper, close to the sole).  Aiming a human's shin / foot at
# those directions therefore does two things the mesh cannot afford:
#
#   * it pitches the whole foot up by 3-15 deg (heel down, toes up - the sole leaves the
#     ground), because the ankle->ball line is a bad proxy for what "the foot is flat"
#     means, and
#   * it bends the knee by 8-15 deg more than the character has it.
#
# Both are baked into the mesh by the T-pose re-fit, so they are not just a preview
# artefact: every tracked frame inherits them.  The character's own pitch is already
# what ARKit tracks, so only the splay (a shin pointing 9 deg outwards, a toe-out foot)
# needs the reference.
DEFAULT_SAGITTAL = {"leg": True, "foot": True, "toe": True}

# Where Apple's metacarpal (`hand*Start`) sits inside the palm, as a fraction of
# wrist -> knuckle, per finger; measured on the reference (index 0.295, middle 0.295,
# ring 0.294, pinky 0.296) so one number covers all four.
START_FRAC = 0.295

TORSO_CHAIN = (["hips_joint"] + ["spine_%d_joint" % k for k in range(1, 8)]
               + ["neck_%d_joint" % k for k in range(1, 5)] + ["head_joint"])


def _seg_tail():
    """Mixamo bone -> the joint at the FAR end of that bone's own segment.

    build_target stores on joint j the accumulated rotation of the segment that *ends*
    at j (see `_step`), so a bone's mesh region has to use the rotation of the joint at
    its own far end: the thigh (LeftUpLeg -> left_leg_joint), the pelvis (Hips ->
    spine_2_joint, spine_1 overlaps the hips), and so on.

    Using the near end instead applies the PREVIOUS segment's rotation: the pelvis -> hip
    offset is 19 deg off the reference and rotating the whole thigh by it swings both legs
    19 deg about +Z, which bows them outwards and makes the knee miss pos[left_leg_joint]
    by 90 mm.
    """
    t = {"Hips": "spine_2_joint", "Spine": "spine_3_joint", "Spine1": "spine_5_joint",
         "Spine2": "neck_1_joint", "Neck": "neck_2_joint", "Head": "head_joint"}
    for side, P in (("left", "Left"), ("right", "Right")):
        t[P + "Shoulder"] = side + "_arm_joint"
        t[P + "Arm"] = side + "_forearm_joint"
        t[P + "ForeArm"] = side + "_hand_joint"
        t[P + "Hand"] = side + "_handIndex_1_joint"   # palm follows the index metacarpal
        t[P + "UpLeg"] = side + "_leg_joint"
        t[P + "Leg"] = side + "_foot_joint"
        t[P + "Foot"] = side + "_toes_joint"
        t[P + "ToeBase"] = side + "_toesEnd_joint"
        t[P + "Toe_End"] = side + "_toesEnd_joint"
        t[P + "HandThumb1"] = side + "_handThumb_1_joint"
        t[P + "HandThumb2"] = side + "_handThumb_2_joint"
        t[P + "HandThumb3"] = side + "_handThumb_2_joint"   # no _3 joint for the thumb
        t[P + "HandThumb4"] = side + "_handThumb_2_joint"
        for an, mix in FINGERS:
            t[P + mix + "1"] = side + "_hand%s_2_joint" % an
            t[P + mix + "2"] = side + "_hand%s_3_joint" % an
            t[P + mix + "3"] = side + "_hand%sEnd_joint" % an
            t[P + mix + "4"] = side + "_hand%sEnd_joint" % an
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


def in_plane(u):
    """`u` turned about +Y until it lies in the character's sagittal (YZ) plane.

    This is the half of a bone direction that costs retargeting accuracy: a shin pointing
    9 deg outwards, a foot toed out.  What is left (pitch and forward lean) is the
    character's stance - "how much the knee is bent", "how the sole lies on the ground" -
    and it is what the mesh is already shaped like.  Keeping it is what `DEFAULT_SAGITTAL`
    asks for; the reference only decides the sideways part (0 for every leg bone).

    A YAW rotation, not a projection: dropping the X component would shorten the
    horizontal run and change the elevation with it (a foot toed out 20 deg came out 2.7
    deg more plantar-flexed, i.e. the sole tilted again).  A rotation about Y keeps the
    vertical component and the length of the horizontal one, so the elevation angle - the
    only thing that decides whether the sole is flat and how the knee is bent - survives
    untouched.
    """
    v = unit(u)
    h = math.hypot(float(v[0]), float(v[2]))
    if h < 1e-6:                      # the bone is vertical: nothing to straighten
        return v
    z = h if v[2] >= 0.0 else -h       # keep pointing forward / backward as it did
    return unit(np.array([0.0, v[1], z]))


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
    before the torso / palm repairs in extract_fbx.py spread them out.

    The target is the joint at the bone's own HEAD (its parent joint in Apple's
    hierarchy): HandIndex1 is the MCP->PIP phalanx, whose head is `handIndex_1_joint`
    (the MCP) - NOT `handIndexStart_joint`, which is the metacarpal inside the palm.
    Apple's `*Start` joints have no Mixamo counterpart; they are fed from the palm by
    `palm_starts()` below.
    """
    m = {"Hips": "hips_joint", "Spine": "spine_2_joint", "Spine1": "spine_4_joint",
         "Spine2": "spine_7_joint", "Neck": "neck_1_joint", "Head": "head_joint",
         # Mixamo's eyeballs are the only weight the face has; without a target their
         # vertices were left unweighted and ended up rigidly bound to the hips
         "LeftEye": "left_eyeball_joint", "RightEye": "right_eyeball_joint"}
    for side, P in (("left", "Left"), ("right", "Right")):
        m[P + "Shoulder"] = side + "_shoulder_1_joint"
        m[P + "Arm"] = side + "_arm_joint"
        m[P + "ForeArm"] = side + "_forearm_joint"
        m[P + "Hand"] = side + "_hand_joint"
        m[P + "UpLeg"] = side + "_upLeg_joint"
        m[P + "Leg"] = side + "_leg_joint"
        m[P + "Foot"] = side + "_foot_joint"
        m[P + "ToeBase"] = side + "_toes_joint"
        m[P + "Toe_End"] = side + "_toesEnd_joint"
        m[P + "HandThumb1"] = side + "_handThumbStart_joint"
        m[P + "HandThumb2"] = side + "_handThumb_1_joint"
        m[P + "HandThumb3"] = side + "_handThumb_2_joint"   # ARKit's thumb has no _3
        m[P + "HandThumb4"] = side + "_handThumbEnd_joint"
        for an, mix in FINGERS:
            m[P + mix + "1"] = side + "_hand%s_1_joint" % an      # MCP
            m[P + mix + "2"] = side + "_hand%s_2_joint" % an      # PIP
            m[P + mix + "3"] = side + "_hand%s_3_joint" % an      # DIP
            m[P + mix + "4"] = side + "_hand%sEnd_joint" % an     # tip cap
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
def _step(P, D, ref, j, parent, anchor, cprev, alpha, sagittal=False):
    """Place joint j on the fitted chain and accumulate its rotation into D[j].

    `anchor`/`cprev` are the character's own bone head/tail, so the length |anchor-cprev|
    is preserved and only the direction changes.  `alpha` decides how much:

      alpha = 1  the segment is aimed at Apple's reference direction in world space
      alpha = 0  the local rotation is identity: the segment keeps the character's own
                 shape and is carried rigidly by its parent (a hand stays a hand)

    `sagittal=True` swaps the target for the character's own direction with the sideways
    component removed (`in_plane`): the segment is straightened out of any splay / toe-out
    but keeps its own pitch, i.e. its own knee bend and its own sole angle.

    D[j] is the ACCUMULATED rotation of the segment that ends at j,
    D[j] = D[parent] @ rot_between(u, D[parent]^T * target), so it satisfies
    D[j] @ u = target and the mesh region of a bone rotates by exactly the rotation the
    chain performs.  (Per-joint minimal rotations, the old version, do not compose: two
    neighbouring bones then get unrelated rotations and the flesh between them shears -
    in the hand that turned straight fingers into a torn fan.)
    """
    cdir = np.asarray(anchor, float) - np.asarray(cprev, float)
    length = float(np.linalg.norm(cdir))
    Ap = D[parent]
    if length <= EPS:
        P[j] = P[parent].copy()
        D[j] = Ap.copy()
        return
    u = unit(cdir)
    if alpha <= 0.0:
        tgt = u                       # local rotation stays identity
    else:
        rd = in_plane(u) if sagittal else unit(ref.dir[j])
        tgt = Ap.T @ rd if alpha >= 1.0 else unit(slerp(u, Ap.T @ rd, alpha))
    R = rot_between(u, tgt)
    D[j] = Ap @ R
    P[j] = np.asarray(P[parent], float) + Ap @ (tgt * length)


def sagittal_flags(over=None):
    """the effective per-chain "keep the character's own stance" flags (DEFAULT_SAGITTAL
    merged with `over`), so a caller can record what the fit actually did"""
    f = dict(DEFAULT_SAGITTAL)
    f.update(over or {})
    return f


def build_target(char, ref, alpha=None, height_scale=1.0, sagittal=None):
    """
    char: Mixamo bind pose in metres / ARKit axes.  Keys:
          Hips, Spine, Spine1, Spine2, Neck, Head,
          <Left|Right>_Shoulder / _Arm / _ForeArm / _Hand / _UpLeg / _Leg / _Foot /
          _ToeBase / _HandIndex1..3 (Middle, Ring, Pinky) / _HandThumb1..3,
          <left|right>_tip_<Index|Mid|Ring|Pinky|Thumb|Toe>  (mesh-detected tip)
    returns (pos: 91 joint positions, D: accumulated rotation per joint, see `_step`)

    `sagittal` overrides DEFAULT_SAGITTAL (per chain: keep the character's own stance,
    straighten only the sideways splay) - see the comment on that table.
    """
    a = dict(DEFAULT_ALPHA)
    a.update(alpha or {})
    sg = sagittal_flags(sagittal)
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

    # ---- face joints: reference offsets from head_joint, scaled with the character.
    # The eyeballs are the only face geometry a Mixamo rig actually skins (to
    # LeftEye/RightEye), so when the character's own eye position is known it is used
    # instead of the reference offset: the eyeball vertices then do not move at all.
    hd = P["head_joint"]
    hrot, hbase = D["head_joint"], np.asarray(char["Head"], float)
    eye_at = {}
    for s, key in (("left", "LeftEye"), ("right", "RightEye")):
        if key in char:
            eye_at[s] = hd + hrot @ (np.asarray(char[key], float) - hbase)
    for j in ref.joints:
        if j in P:
            continue
        if ref.parent.get(j) in FACE_ROOT:
            P[j] = hd + (ref.pos[j] - ref.pos["head_joint"]) * height_scale
            D[j] = hrot.copy()
            side = j.split("_")[0]
            if "eye" in j and side in eye_at:
                P[j] = eye_at[side].copy()

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

        # the hips -> hip offset is a sideways bone: it still has to follow the reference
        # exactly, or the legs end up too far apart / too close together.  Everything
        # below it (thigh, shin, foot, toe) keeps the character's own stance.
        _step(P, D, ref, side + "_upLeg_joint", "hips_joint",
              char[S + "UpLeg"], char["Hips"], a["leg"])
        _step(P, D, ref, side + "_leg_joint", side + "_upLeg_joint",
              char[S + "Leg"], char[S + "UpLeg"], a["leg"], sagittal=sg["leg"])
        _step(P, D, ref, side + "_foot_joint", side + "_leg_joint",
              char[S + "Foot"], char[S + "Leg"], a["foot"], sagittal=sg["foot"])
        _step(P, D, ref, side + "_toes_joint", side + "_foot_joint",
              char[S + "ToeBase"], char[S + "Foot"], a["toe"], sagittal=sg["toe"])
        _step(P, D, ref, side + "_toesEnd_joint", side + "_toes_joint",
              char[side + "_tip_Toe"], char[S + "ToeBase"], a["toe"], sagittal=sg["toe"])

        # fingers.  The metacarpal (Start) has no Mixamo bone: it is synthesised inside the
        # palm at the reference's fraction of wrist -> knuckle.  HandIndex1..3 are the
        # MCP/PIP/DIP phalanges (-> _1/_2/_3) and the mesh tip is End.  With alpha 0 every
        # segment keeps the character's own local direction, so the hand is carried
        # rigidly by the wrist and its shape is untouched.
        for an, mix in FINGERS:
            meta = lerp(char[S + "Hand"], char[S + mix + "1"], START_FRAC)
            tip = char[side + "_tip_" + an]
            chain = [(side + "_hand%sStart_joint" % an, meta, char[S + "Hand"]),
                     (side + "_hand%s_1_joint" % an, char[S + mix + "1"], meta),
                     (side + "_hand%s_2_joint" % an, char[S + mix + "2"], char[S + mix + "1"]),
                     (side + "_hand%s_3_joint" % an, char[S + mix + "3"], char[S + mix + "2"]),
                     (side + "_hand%sEnd_joint" % an, tip, char[S + mix + "3"])]
            for j, anchor, cprev in chain:
                _step(P, D, ref, j, ref.parent[j], anchor, cprev, a["finger"])

        # thumb: Mixamo Thumb1..3 are the metacarpal / MCP / IP, which is exactly Apple's
        # Start / _1 / _2 (Apple's thumb has no _3), and the tip is End.
        chain = [(side + "_handThumbStart_joint", char[S + "HandThumb1"], char[S + "Hand"]),
                 (side + "_handThumb_1_joint", char[S + "HandThumb2"], char[S + "HandThumb1"]),
                 (side + "_handThumb_2_joint", char[S + "HandThumb3"], char[S + "HandThumb2"]),
                 (side + "_handThumbEnd_joint", char[side + "_tip_Thumb"],
                  char[S + "HandThumb3"])]
        for j, anchor, cprev in chain:
            _step(P, D, ref, j, ref.parent[j], anchor, cprev, a["finger"])

    return P, D


# ---------------------------------------------------------------- mesh re-fit
def bone_transforms(char, pos, D, wmap, seg_tail=None):
    """Rigid rest-pose delta per Mixamo bone: p' = pos[j] + R @ (p - char[bone]).

    R is the ACCUMULATED rotation of the bone's OWN segment (see SEG_TAIL and `_step`), so
    the bone's head lands on pos[j] and its tail on the fitted child joint: the region
    keeps its length and two neighbours agree on the joint between them.  For a chain that
    is not snapped (the fingers, alpha 0) R is the same for every bone in the chain, i.e.
    the hand is transformed rigidly and cannot be deformed.
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


def palm_starts(points, pos, side, lo=0.35, sigma_frac=0.35):
    """How the palm's `hand_joint` weight splits over the four metacarpals (Start).

    Apple's finger chain begins with a metacarpal joint inside the palm, Mixamo's does not,
    so without this repair handIndexStart / handMidStart / ... would carry exactly zero
    weight (usdchecker and RealityKit both expect every deformable joint to be used) and
    the distal palm would never follow the metacarpals.

    Every palm vertex is handed to the knuckle it points at (soft, distance weighted, so
    neighbouring metacarpal regions blend instead of stepping) and the share ramps along
    wrist -> knuckle, so the wrist itself keeps following hand_joint.
    Returns (nv, 4) fractions of the hand_joint weight, summing to <= 1.
    """
    P = np.asarray(points, float)
    hand = np.asarray(pos[side + "_hand_joint"], float)
    knuckles = np.stack([np.asarray(pos[side + "_hand%s_1_joint" % an], float)
                         for an, _ in FINGERS])
    v = P - hand
    palm = float(np.mean(np.linalg.norm(knuckles - hand, axis=1)))
    t = np.zeros((len(P), len(knuckles)))
    d = np.zeros_like(t)
    for k, knuckle in enumerate(knuckles):
        ab = knuckle - hand
        L2 = max(float(ab @ ab), EPS)
        t[:, k] = (v @ ab) / L2
        d[:, k] = np.linalg.norm(v - t[:, k:k + 1] * ab, axis=1)
    w = np.exp(-((d / max(sigma_frac * palm, EPS)) ** 2)) * (t > 0.0)
    w /= np.maximum(w.sum(1, keepdims=True), EPS)
    ramp = smoothstep((np.clip(t.max(1), 0.0, 1.0) - lo) / (1.0 - lo))
    return w * ramp[:, None]


def mirror_name(bone):
    if bone.startswith("Left"):
        return "Right" + bone[4:]
    if bone.startswith("Right"):
        return "Left" + bone[5:]
    return bone


def mirror_fill(points, W, bone_names, missing, tol=0.05):
    """Give a bone that only exists on ONE side the weight of its counterpart.

    `mirror_weights` mirrors the whole left half onto the right, but it needs the two
    halves to be within 3 cm of each other - which is exactly what fails when an export
    left a finger without any skin cluster (its vertices were weighted to the neighbouring
    fingers, so nothing lines up).  Here only the missing bones' columns are written, from
    the nearest mirrored vertex that carries the counterpart, so the finger at least
    follows its own joints instead of hanging off a neighbour permanently.
    Returns (W, number of vertices touched).
    """
    P = np.asarray(points, float)
    bi = {b: i for i, b in enumerate(bone_names)}
    cell = max(tol, 0.01)
    touched = 0
    for is_left_src in (True, False):
        pairs = [(bi[b], bi[mirror_name(b)]) for b in missing
                 if b in bi and mirror_name(b) in bi
                 and mirror_name(b).startswith("Left") == is_left_src]
        if not pairs:
            continue
        src_cols = sorted({s for _d, s in pairs})
        cand = np.nonzero((W[:, src_cols] > 0).any(1))[0]
        cand = cand[P[cand, 0] > 0] if is_left_src else cand[P[cand, 0] < 0]
        dst = np.nonzero(P[:, 0] < 0)[0] if is_left_src else np.nonzero(P[:, 0] > 0)[0]
        if not len(cand) or not len(dst):
            continue
        q = np.c_[-P[cand, 0], P[cand, 1], P[cand, 2]]     # the source side, mirrored over
        grid = {}
        for k, i in enumerate(cand):
            grid.setdefault(tuple(np.floor(q[k] / cell).astype(np.int64)), []).append(k)
        for j in dst:
            k0 = np.floor(P[j] / cell).astype(np.int64)
            best, bd = -1, tol
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for k in grid.get((k0[0] + dx, k0[1] + dy, k0[2] + dz), ()):
                            d = float(np.linalg.norm(q[k] - P[j]))
                            if d < bd:
                                best, bd = k, d
            if best < 0:
                continue
            i = cand[best]
            for dcol, scol in pairs:                       # one mirrored vertex fills them all
                if W[i, scol] > 0:
                    W[j, dcol] = W[i, scol]
            touched += 1
    return W, touched


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


def _pitch(v):
    """elevation of a direction in degrees: -90 straight down, 0 horizontal"""
    v = unit(v)
    return math.degrees(math.asin(float(np.clip(v[1], -1.0, 1.0))))


def stance_report(char, pos, D=None):
    """Per side: the character's own knee bend / foot pitch and the fitted ones (deg).

    The foot pitch is the ankle -> ball-of-foot line, so a big change in it is exactly
    "the sole no longer lies on the ground": the mesh region of a foot is rotated by the
    rotation of that segment (SEG_TAIL), which means this number is the amount the whole
    foot was pitched while being fitted.  With DEFAULT_SAGITTAL both rows are identical.
    Returns {side: {"knee_bend_own", "knee_bend", "foot_pitch_own", "foot_pitch"}}.
    """
    out = {}
    for side, S in SIDE.items():
        own = [char.get(S + n) for n in ("UpLeg", "Leg", "Foot", "ToeBase")]
        got = [pos.get(side + n) for n in ("_upLeg_joint", "_leg_joint",
                                           "_foot_joint", "_toes_joint")]
        if any(v is None for v in own + got):
            continue
        hip, knee, ankle, ball = [np.asarray(v, float) for v in own]
        fhip, fknee, fankle, fball = [np.asarray(v, float) for v in got]
        out[side] = {
            "knee_bend_own": 180.0 - angle_between(knee - hip, ankle - knee),
            "knee_bend": 180.0 - angle_between(fknee - fhip, fankle - fknee),
            "foot_pitch_own": _pitch(ball - ankle),
            "foot_pitch": _pitch(fball - fankle),
        }
    return out
