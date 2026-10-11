"""Zip the trained weights the default config needs, so others can run the demo without training.

    python scripts/tools/package_weights.py            # default models only (~140 MB)
    python scripts/tools/package_weights.py --all      # every trained model

The archive keeps repository-relative paths: unzip it in the repository root. It contains a
manifest with SHA-256 checksums.

Sharing: the segmentation models and detector v2 were trained on RailSem19, whose licence restricts
redistribution of its frames and asks every user to register. Weights are not frames, but share the
package privately (team, guide, examiners), not publicly.
"""

import argparse
import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

from railsight.config import ROOT, load_config

NOTE = """RailSight trained weights
=========================
Unzip in the RailSight repository root; paths match configs/default.yaml.

Segmentation models and detector v2 were trained on RailSem19 (wilddash.cc/railsem19), whose
licence restricts redistribution. Share this package privately only.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include every trained model, not just the defaults")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()

    cfg = load_config()
    if args.all:
        files = [*sorted((ROOT / "runs/seg").glob("*/best/*")),
                 *sorted((ROOT / "runs/det").glob("*/weights/best.pt"))]
    else:
        seg_dir = ROOT / cfg["models"]["seg_weights"]
        files = [*sorted(seg_dir.glob("*")), ROOT / cfg["models"]["det_weights"]]
    files = [f for f in files if f.is_file()]
    if not files:
        raise SystemExit("no trained weights found under runs/")

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    name = f"railsight-weights-{'all' if args.all else 'default'}-{date.today():%Y%m%d}.zip"
    manifest = {"created": f"{date.today()}", "files": {}}
    with zipfile.ZipFile(out / name, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = f.relative_to(ROOT).as_posix()
            z.write(f, rel)
            manifest["files"][rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
        z.writestr("WEIGHTS_README.txt", NOTE)
        z.writestr("weights_manifest.json", json.dumps(manifest, indent=2))
    size = (out / name).stat().st_size / 2**20
    print(f"{len(files)} files -> {out / name} ({size:.0f} MB)")
    print("Share privately: models trained on RailSem19 (licence restricts redistribution).")


if __name__ == "__main__":
    main()
