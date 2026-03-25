from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "ml-100k.zip"
    print(f"Downloading: {DATA_URL}")
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    zip_path.write_bytes(r.content)
    print(f"Saved: {zip_path}")

    extract_dir = data_dir / "ml-100k"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        zf.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}")


if __name__ == "__main__":
    main()
