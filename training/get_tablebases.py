"""Download the 3-4-man Syzygy WDL tables into weights/syzygy/ (~30 MB).

    python -m training.get_tablebases

3-4-man is 35 .rtbw files. 5-man WDL is another ~380 MB, over the 200 MB
submission budget, so we stop at 4.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import chess.syzygy

_MIRRORS = (
    "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl",
    "https://tablebase.sesse.net/syzygy/3-4-5",
)


def _table_files(max_pieces: int) -> list[str]:
    names = chess.syzygy.tablenames(piece_count=max_pieces)
    return [f"{name}.rtbw" for name in names if len(name.replace("v", "")) <= max_pieces]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("weights/syzygy"))
    parser.add_argument("--max-pieces", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = _table_files(args.max_pieces)
    print(f"{len(files)} tables -> {args.out}")
    for name in files:
        target = args.out / name
        if target.exists() and target.stat().st_size > 0:
            continue
        for mirror in _MIRRORS:
            try:
                urllib.request.urlretrieve(f"{mirror}/{name}", target)
                print(f"  {name}  ({target.stat().st_size / 1e6:.2f} MB)")
                break
            except Exception as error:  # try the next mirror
                print(f"  {name} from {mirror}: {error}")
        else:
            raise SystemExit(f"could not fetch {name} from any mirror")

    total = sum(f.stat().st_size for f in args.out.glob("*.rtbw"))
    print(f"done: {total / 1e6:.1f} MB in {args.out}")


if __name__ == "__main__":
    main()
