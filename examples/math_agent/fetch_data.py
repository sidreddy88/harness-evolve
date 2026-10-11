"""Download a sample of GSM-Hard (Gao et al., "PAL: Program-aided Language
Models", 2022): GSM8K questions with large, awkward numbers, so the arithmetic
itself is hard and how the agent uses its calculator matters.

    python fetch_data.py                                   # 80 problems -> data/problems.jsonl
    python fetch_data.py --n 260 --out data/problems-bench.jsonl   # the benchmark set

The data is downloaded, not shipped with this repo.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

URL = ("https://datasets-server.huggingface.co/rows?dataset=reasoning-machines%2Fgsm-hard"
       "&config=default&split=train&offset={offset}&length=100")
OUT = Path(__file__).parent / "data" / "problems.jsonl"


def main(n: int = 80, seed: int = 7, out: Path = OUT) -> None:
    rows = []
    for offset in range(0, 1300, 100):
        with urllib.request.urlopen(URL.format(offset=offset), timeout=60) as resp:
            rows += [r["row"] for r in json.load(resp)["rows"]]
    sample = random.Random(seed).sample(rows, n)
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({"id": f"gsmhard-{i:03d}", "question": r["input"], "answer": r["target"]}) + "\n")
    print(f"{n} problems -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    main(n=args.n, out=args.out)
