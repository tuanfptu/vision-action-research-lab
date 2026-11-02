"""Fresh project-managed official assets, pinned to public upstream commits."""

import subprocess
from pathlib import Path

from va_lab.utils.runtime import write_json

UPSTREAMS = {
    "openarm_mujoco": (
        "https://github.com/enactic/openarm_mujoco.git",
        "a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c",
    ),
    "openarm_dataset": (
        "https://github.com/enactic/openarm_dataset.git",
        "58af735f600bd4f907fb12ac6ca30c76c8524329",
    ),
}


def acquire_assets(root: str = "assets/upstream") -> dict:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (url, sha) in UPSTREAMS.items():
        path = root / name
        if not path.exists():
            subprocess.run(["git", "clone", "--no-checkout", url, str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", sha], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != sha:
            raise ValueError(f"Unexpected asset revision for {name}: {actual}")
        if subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError(f"Modified upstream checkout: {name}")
        records[name] = {"url": url, "revision": sha, "license": "Apache-2.0"}
    write_json(root.parent / "upstream_manifest.json", records)
    return records
