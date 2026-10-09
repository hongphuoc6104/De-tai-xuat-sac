#!/usr/bin/env python3
"""Run a bounded real-data CLI E2E smoke with shard interruption and resume proof."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

CODE_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    """Read one small published JSON descriptor."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Exercise public CLI using at most a few fields per requested lens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lenses", type=int, nargs="+", default=[4, 10, 40])
    parser.add_argument("--per-lens", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.per_lens <= 5:
        parser.error("Local E2E smoke allows only 1..5 images per lens; use Colab for full data.")
    data = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw, processed, work = output / "raw", output / "processed", output / "work"
    sources = []
    if 4 in args.lenses:
        sources.append(data / "train_images")
    for lens in (10, 40):
        if lens in args.lenses:
            sources.append(data / f"data_{lens}x.zip")
    if not sources or any(not source.exists() for source in sources):
        parser.error("Missing local source for a requested lens.")
    started = time.monotonic()
    actions = []

    def invoke(*arguments: str | Path) -> None:
        command = [sys.executable, "-m", "histology_data", *map(str, arguments)]
        result = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True, check=False)
        actions.append(dict(command=command, exit_code=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        (output / "execution_actions.json").write_text(json.dumps(actions, indent=2, ensure_ascii=False), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout or "CLI smoke failed.")

    prepare = ["prepare", "--metadata", data / "Metadata.xlsx", "--release", raw,
               "--config", CODE_ROOT / "configs/data_shards_smoke.json",
               "--per-lens", str(args.per_lens), "--lenses", *map(str, args.lenses),
               "--max-shards", "1"]
    for source in sources:
        prepare.extend(["--source", source])
    initial_counts = []
    for _ in range(args.per_lens * len(args.lenses) + 1):
        invoke(*prepare)
        release = read_json(raw / "release.json")
        initial_counts.append(len(release["shards"]))
        if release["complete"]:
            break
    else:
        raise RuntimeError("Bounded packaging invocations failed to reach completion.")
    assert len(release["catalog"]["images"]) <= args.per_lens * len(args.lenses)
    assert release["catalog"]["mode"] == "smoke" and not release["catalog"]["training_ready"]
    assert all(part["size_bytes"] <= 32 * 1024 * 1024 for part in release["shards"])
    invoke("verify", "--release", raw)
    process = ["process", "--release", raw, "--work-root", work, "--output", processed,
               "--config", CODE_ROOT / "configs/data_shards_smoke.json", "--max-shards", "1"]
    for _ in release["shards"]:
        invoke(*process)
    commits = sorted(processed.glob("shard-*/*/commit.json"))
    assert len(commits) == len(release["shards"])
    before = {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in commits}
    invoke(*process)
    assert before == {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in commits}
    records = [read_json(path) for path in commits]
    assert all(record["complete"] and not record["training_ready"] for record in records)
    staged_images = list((work / "staged").rglob("*.tiff")) if (work / "staged").exists() else []
    assert not staged_images, "Default processing must bound working raw storage after commits."
    summary = dict(timestamp=dt.datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
                   duration_seconds=round(time.monotonic() - started, 3),
                   lenses=args.lenses, per_lens=args.per_lens,
                   images_selected=len(release["catalog"]["images"]),
                   shards=len(release["shards"]), packaging_progress=initial_counts,
                   images_processed=sum(record["images_seen"] for record in records),
                   tiles_written=sum(record["tiles_written"] for record in records),
                   images_review=sum(record["images_review"] for record in records),
                   raw_complete=release["complete"], training_ready=False,
                   commit_unchanged_on_resume=True, staged_raw_cleaned=True,
                   catalog_id=release["catalog_id"], processing_version=records[0]["processing_version"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
