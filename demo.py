#!/usr/bin/env python3
"""
Generate zip lists (and optional archives) for val/test take files using EgoExo4D metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def parse_capture_ids(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    capture_ids: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                capture_ids.add(item)
    return capture_ids


def take_matches_capture_ids(take: dict, capture_ids: set[str]) -> bool:
    if not capture_ids:
        return True
    candidates: set[str] = set()
    for key in ("take_name", "root_dir", "capture_uid"):
        value = take.get(key)
        if value:
            candidates.add(value)
    root_dir = take.get("root_dir")
    if root_dir:
        candidates.add(Path(root_dir).name)
    capture = take.get("capture")
    if isinstance(capture, dict):
        for key in ("capture_name", "origin_capture_id", "capture_uid"):
            value = capture.get(key)
            if value:
                candidates.add(value)
    return bool(candidates & capture_ids)


def build_uid_to_split(splits: dict) -> dict:
    uid_to_split = splits.get("take_uid_to_split")
    if uid_to_split:
        return uid_to_split
    split_to_uids = splits.get("split_to_take_uids")
    if not split_to_uids:
        raise ValueError("splits.json missing take_uid_to_split or split_to_take_uids")
    uid_to_split = {}
    for split, uids in split_to_uids.items():
        for uid in uids:
            uid_to_split[uid] = split
    return uid_to_split


def write_list_file(path: Path, entries: list[str]) -> None:
    path.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")


def run_zip(
    data_dir: Path,
    output_dir: Path,
    split: str,
    list_path: Path,
) -> Path | None:
    if not list_path.is_file() or list_path.stat().st_size == 0:
        print(f"No files listed for {split}; skipping zip.")
        return None
    zip_path = output_dir / f"{split}_videos.zip"
    print(f"Zipping {split} files to: {zip_path}")
    with list_path.open("r", encoding="utf-8") as f:
        subprocess.run(
            ["zip", "-0", "-@", str(zip_path)],
            cwd=str(data_dir),
            stdin=f,
            check=True,
        )
    return zip_path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_meta_dir = repo_root / "data" / "ee4d_motion_uniegomotion"
    default_data_dir = os.environ.get("EGOEXO4D_DATA_DIR")

    parser = argparse.ArgumentParser(
        description="Generate zip file lists (and optional archives) for val/test EgoExo4D takes.")
    parser.add_argument("--data-dir", default=default_data_dir, help="EgoExo4D dataset root.")
    parser.add_argument("--metadata-dir", default=str(default_meta_dir), help="Metadata root.")
    parser.add_argument("--splits-json", default=None, help="Path to splits.json.")
    parser.add_argument("--takes-json", default=None, help="Path to takes.json.")
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated splits to include (default: val,test).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
        help="Where to write the zip outputs and path lists.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="If set, randomly sample this many takes and zip all items under their directories.",
    )
    parser.add_argument(
        "--capture-id",
        action="append",
        default=None,
        help=("Capture id(s) to include (repeat flag or pass a comma-separated list). "
              "Matches take_name/root_dir/capture fields."),
    )
    parser.add_argument(
        "--run-zip",
        action="store_true",
        help="If set, run zip for each split using the generated file lists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()

    if not args.data_dir:
        raise SystemExit("Missing --data-dir (or set EGOEXO4D_DATA_DIR).")

    data_dir = Path(args.data_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"DATA_DIR does not exist: {data_dir}")
    splits_path = (Path(args.splits_json).expanduser().resolve()
                   if args.splits_json else metadata_dir / "annotations" / "splits.json")
    takes_path = (Path(args.takes_json).expanduser().resolve()
                  if args.takes_json else metadata_dir / "takes.json")
    if not splits_path.is_file():
        raise SystemExit(f"splits.json not found: {splits_path}")
    if not takes_path.is_file():
        raise SystemExit(f"takes.json not found: {takes_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_json(splits_path)
    uid_to_split = build_uid_to_split(splits)
    target_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    capture_ids = parse_capture_ids(args.capture_id)

    takes = load_json(takes_path)
    split_paths = {split: [] for split in target_splits}
    split_seen = {split: set() for split in target_splits}
    split_sizes = {split: 0 for split in target_splits}
    split_missing = {split: [] for split in target_splits}

    # Filter takes to those in the target splits
    target_takes = [t for t in takes if uid_to_split.get(t.get("take_uid")) in target_splits]
    if capture_ids:
        target_takes = [t for t in target_takes if take_matches_capture_ids(t, capture_ids)]
        if not target_takes:
            raise SystemExit("No takes matched the provided --capture-id values.")

    if args.sample_n is not None and len(target_takes) > args.sample_n:
        print(f"Sampling {args.sample_n} takes from {len(target_takes)} candidates.")
        random.seed(args.seed)
        target_takes = random.sample(target_takes, args.sample_n)
    
    processed_takes = 0
    for take in target_takes:
        take_uid = take.get("take_uid")
        split = uid_to_split.get(take_uid)
        root_dir = take.get("root_dir")
        if not root_dir:
            continue
        processed_takes += 1

        # Collect all files under the take directory
        take_dir = data_dir / root_dir
        print("Including take dir:", take_dir)
        if take_dir.is_dir():
            for root, _, files in os.walk(take_dir):
                for file in files:
                    abs_path = Path(root) / file
                    rel_path = abs_path.relative_to(data_dir)
                    rel_str = rel_path.as_posix()
                    if rel_str not in split_seen[split]:
                        split_paths[split].append(rel_str)
                        split_sizes[split] += abs_path.stat().st_size
                        split_seen[split].add(rel_str)
        else:
            split_missing[split].append(str(take_dir))

    if args.sample_n is not None:
        assert processed_takes == len(target_takes), \
            f"Expected to process {len(target_takes)} takes, but processed {processed_takes}."

    for split in target_splits:
        split_paths[split].sort()
        list_path = output_dir / f"{split}_video_paths.txt"
        write_list_file(list_path, split_paths[split])
        if split_missing[split]:
            missing_path = output_dir / f"{split}_missing_video_paths.txt"
            write_list_file(missing_path, split_missing[split])

    zip_outputs = []
    if args.run_zip:
        for split in target_splits:
            list_path = output_dir / f"{split}_video_paths.txt"
            zip_path = run_zip(data_dir, output_dir, split, list_path)
            if zip_path:
                zip_outputs.append(zip_path)

    total_bytes = sum(split_sizes.values())
    print("File size summary:")
    for split in target_splits:
        print(f"- {split}: {len(split_paths[split])} files, {format_bytes(split_sizes[split])}")
        print("Included", split_paths[split])
        if split_missing[split]:
            print(f"  missing: {len(split_missing[split])} files")
            print(split_missing[split])
    print(f"- total: {format_bytes(total_bytes)}")
    print(f"Wrote list files to: {output_dir}")
    if zip_outputs:
        print("Zip outputs:")
        for zip_path in zip_outputs:
            print(f"- {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
Suggested next steps:

Run 
CAP_ID=cmu_bike10_2
python scripts/extract_val_test_videos.py \
    --data-dir /home/share/datasets/egoexo4d \
    --capture-id $CAP_ID \
    --output-dir examples/$CAP_ID
to generate the lists and size estimate.

"""
