"""Download only G1 assets from a pinned Menagerie revision (including license)."""
import argparse
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import time

REVISION = "370c43965323c0944b433b202aac5058b5f907a9"
ROOT = Path(__file__).resolve().parents[1]


def download(destination: Path):
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/unitree_g1/"
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = destination / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(base + name, headers={"User-Agent": "g1-ppo"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        data = response.read()
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(attempt + 1)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        return path

    for name in ["g1.xml", "scene.xml", "LICENSE", "README.md"]:
        fetch(name)
    tree = ET.parse(destination / "g1.xml")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, ["assets/" + mesh.attrib["file"] for mesh in tree.findall(".//asset/mesh")]))
    (destination / "source.json").write_text(json.dumps({"revision": REVISION, "url": base}, indent=2))
    print(f"G1 model ready: {destination / 'scene.xml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "assets/unitree_g1")
    download(parser.parse_args().output)
