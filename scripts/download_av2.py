"""Download a subset of the Argoverse 2 Motion Forecasting dataset.

The dataset sits in a public S3 bucket; this script lists it over HTTPS and
fetches the first N scenarios of a split (about 250 KB per scenario).

    python scripts/download_av2.py --split train --num 20000
    python scripts/download_av2.py --split val --num 3000
"""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

BUCKET = "https://s3.amazonaws.com/argoverse"
PREFIX = "datasets/av2/motion-forecasting"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(split: str, num_scenarios: int) -> list[tuple[str, int]]:
    keys, token = [], None
    want = num_scenarios * 2  # one parquet + one map per scenario
    while len(keys) < want:
        query = {"list-type": "2", "prefix": f"{PREFIX}/{split}/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(query)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for item in root.findall("s3:Contents", NS):
            keys.append((item.find("s3:Key", NS).text, int(item.find("s3:Size", NS).text)))
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token:
            break
    return keys[:want]


def fetch(key: str, size: int, dest_root: Path) -> None:
    dest = dest_root / Path(key).relative_to(PREFIX)
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", tmp)
            tmp.rename(dest)
            return
        except Exception:
            if attempt == 2:
                raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--num", type=int, required=True, help="number of scenarios")
    ap.add_argument("--dest", type=Path, default=Path("data/raw/av2"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    keys = list_keys(args.split, args.num)
    total_gb = sum(s for _, s in keys) / 1e9
    print(f"{len(keys) // 2} scenarios, {total_gb:.2f} GB -> {args.dest / args.split}")
    with ThreadPoolExecutor(args.threads) as pool:
        list(tqdm(pool.map(lambda k: fetch(*k, args.dest), keys), total=len(keys)))


if __name__ == "__main__":
    main()
