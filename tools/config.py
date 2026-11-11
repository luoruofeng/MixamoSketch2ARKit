"""Single source of truth for every path used by the pipeline.

Nothing else in the project may hard-code a directory: the layout is

    <project>/models/            input Mixamo FBX files (one or many)
    <project>/build/<slug>/      intermediate files for one character
    <project>/build/arkit_skel.json   Apple's reference skeleton (shared)
    <project>/dist/<slug>_ARKit.usdz  finished, ARKit-ready assets
    <project>/Biped-Robot/...    Apple's reference biped_robot.usdz (input asset)

Any of them can be moved with an environment variable, which is what the
`--models / --build / --dist` switches of run_all.py set:

    MIXAMO_MODELS          input directory
    MIXAMO_BUILD           intermediate directory
    MIXAMO_DIST            output directory
    ARKIT_REFERENCE_USDZ   Apple's biped_robot.usdz
    ARKIT_SKEL_JSON        extracted reference skeleton (build/arkit_skel.json)
    MIXAMO_SKIP            comma separated slugs to leave alone
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_NAME = "biped_robot.usdz"


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


MODELS_DIR = _env_path("MIXAMO_MODELS", PROJECT_ROOT / "models")
BUILD_DIR = _env_path("MIXAMO_BUILD", PROJECT_ROOT / "build")
DIST_DIR = _env_path("MIXAMO_DIST", PROJECT_ROOT / "dist")
SKEL_JSON = _env_path("ARKIT_SKEL_JSON", BUILD_DIR / "arkit_skel.json")

SKIP = frozenset(s.strip() for s in os.environ.get("MIXAMO_SKIP", "").split(",") if s.strip())


def ensure_dirs():
    for d in (MODELS_DIR, BUILD_DIR, DIST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def slug_of(path_or_name):
    """'models/Whiteclown N Hallin.fbx' -> 'Whiteclown_N_Hallin'"""
    base = os.path.basename(str(path_or_name))
    if base.lower().endswith(".fbx"):
        base = base[:-4]
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", base).strip("_")


def model_files():
    """slug -> absolute fbx path (a slug is lossy, so always resolve through the folder)"""
    if not MODELS_DIR.is_dir():
        return {}
    out = {}
    for fn in sorted(os.listdir(MODELS_DIR)):
        if fn.lower().endswith(".fbx"):
            out[slug_of(fn)] = str(MODELS_DIR / fn)
    return out


def all_models(include_skipped=False):
    """every model slug in the input directory, in a stable order"""
    return [s for s in sorted(model_files()) if include_skipped or s not in SKIP]


def model_path(slug):
    files = model_files()
    if slug not in files:
        raise SystemExit("unknown model %r.\n  put the Mixamo .fbx files in %s\n  found: %s"
                         % (slug, MODELS_DIR, ", ".join(sorted(files)) or "(nothing)"))
    return files[slug]


def build_dir(slug):
    return str(BUILD_DIR / slug)


def tex_dir(slug):
    return os.path.join(build_dir(slug), "tex")


def usdc_path(slug):
    return os.path.join(build_dir(slug), slug + "_ARKit.usdc")


def usdz_path(slug):
    return str(DIST_DIR / (slug + "_ARKit.usdz"))


def reference_usdz():
    """Apple's biped_robot.usdz - the authoritative ARKit skeleton lives in it."""
    env = os.environ.get("ARKIT_REFERENCE_USDZ")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise SystemExit("ARKIT_REFERENCE_USDZ does not exist: %s" % p)
        return p
    hits = sorted(p for p in PROJECT_ROOT.rglob(REFERENCE_NAME) if "__MACOSX" not in p.parts)
    if not hits:
        raise SystemExit("cannot find Apple's %s under %s.\n"
                         "  download \"Biped Robot\" from Apple's sample assets and unzip it,\n"
                         "  or set ARKIT_REFERENCE_USDZ to the .usdz file"
                         % (REFERENCE_NAME, PROJECT_ROOT))
    return hits[0]
