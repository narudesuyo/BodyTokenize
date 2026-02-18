#!/usr/bin/env python3
"""
Generate zip lists (and optional archives) for val/test take files using EgoExo4D metadata.
Filters to takes that have both hand_pose and body_pose (or bodypose) in their available annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
import os
import time

# Annotation names that count as "bodypose"
BODY_KEYS = {"body_pose", "bodypose", "ego_body_pose", "ego_bodypose", "egobodypose"}
# Annotation names that count as "hand pose"
HAND_KEYS = {"hand_pose", "handpose", "ego_hand_pose", "ego_handpose", "egohandpose"}


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


def take_has_hand_and_body(annotations: list[str] | None) -> bool:
    """True if the take has both bodypose and hand pose in its annotation list."""
    if not annotations:
        return False
    ann_set = set(annotations)
    return bool(ann_set & BODY_KEYS) and bool(ann_set & HAND_KEYS)


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
    expected_bytes: int | None = None,   # 追加
    poll_sec: float = 2.0,               # 追加
) -> Path | None:
    if not list_path.is_file() or list_path.stat().st_size == 0:
        print(f"No files listed for {split}; skipping zip.")
        return None

    zip_path = output_dir / f"{split}_videos.zip"
    if zip_path.exists():
        zip_path.unlink()  # 途中再開はしない想定（必要なら消さない運用に変える）

    print(f"Zipping {split} files to: {zip_path}")
    if expected_bytes:
        print(f"Expected input size: {format_bytes(expected_bytes)}")

    start = time.time()

    # zip を非同期起動
    with list_path.open("r", encoding="utf-8") as f:
        proc = subprocess.Popen(
            ["zip", "-0", "-@", str(zip_path)],
            cwd=str(data_dir),
            stdin=f,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )

        last_print = 0.0
        while True:
            ret = proc.poll()
            now = time.time()

            # 定期的に進捗表示
            if now - last_print >= poll_sec:
                last_print = now
                zipped = zip_path.stat().st_size if zip_path.exists() else 0
                elapsed = now - start

                if expected_bytes and expected_bytes > 0:
                    frac = min(zipped / expected_bytes, 0.999999)
                    speed = zipped / max(elapsed, 1e-6)  # bytes/sec
                    remaining = (expected_bytes - zipped) / max(speed, 1e-6) if speed > 0 else float("inf")
                    print(
                        f"[{split}] {format_bytes(zipped)} / {format_bytes(expected_bytes)} "
                        f"({frac*100:.1f}%) | {format_bytes(int(speed))}/s | ETA {int(remaining)}s",
                        flush=True,
                    )
                else:
                    speed = zipped / max(elapsed, 1e-6)
                    print(
                        f"[{split}] zipped={format_bytes(zipped)} | {format_bytes(int(speed))}/s | elapsed {int(elapsed)}s",
                        flush=True,
                    )

            if ret is not None:
                break

        if ret != 0:
            raise subprocess.CalledProcessError(ret, proc.args)

    total = zip_path.stat().st_size if zip_path.exists() else 0
    print(f"Done: {zip_path} ({format_bytes(total)}), elapsed {int(time.time() - start)}s")
    return zip_path 


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_meta_dir = repo_root / "data" / "ee4d_motion_uniegomotion"
    DATA_ROOT = os.environ.get("DATA_ROOT")

    parser = argparse.ArgumentParser(
        description="Generate zip file lists for val/test EgoExo4D takes (hand+body pose only).")
    parser.add_argument("--data-dir", default="/home/share/datasets/egoexo4d", help="EgoExo4D dataset root.")
    parser.add_argument("--metadata-dir", default=str(default_meta_dir), help="Metadata root.")
    parser.add_argument("--splits-json", default=os.path.join(DATA_ROOT, "ee4d/ee4d_motion_uniegomotion/annotations/splits.json"), help="Path to splits.json.")
    parser.add_argument("--takes-json", default=os.path.join(DATA_ROOT, "ee4d/ee4d_motion_uniegomotion/takes.json"), help="Path to takes.json.")
    parser.add_argument(
        "--annotations-key",
        default="take_uid_to_benchmark",
        help="Key in splits.json for take_uid -> list of annotation names (default: take_uid_to_available_annotations).",
    )
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
    uid_to_annotations = splits.get(args.annotations_key)
    if uid_to_annotations is None:
        uid_to_annotations = splits.get("take_uid_to_annotations") or {}
    if not uid_to_annotations:
        print(f"Warning: no key '{args.annotations_key}' (or take_uid_to_annotations) in splits.json; including all takes.")
    target_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    capture_ids = parse_capture_ids(args.capture_id)

    takes = load_json(takes_path)
    split_paths = {split: [] for split in target_splits}
    split_seen = {split: set() for split in target_splits}
    split_sizes = {split: 0 for split in target_splits}
    split_missing = {split: [] for split in target_splits}

    # Filter takes to those in the target splits
    target_takes = [t for t in takes if uid_to_split.get(t.get("take_uid")) in target_splits]
    # Keep only takes that have both hand_pose and bodypose in available annotations
    if uid_to_annotations:
        before = len(target_takes)
        target_takes = [t for t in target_takes if take_has_hand_and_body(uid_to_annotations.get(t.get("take_uid")))]
        print(f"Filtered to takes with hand_pose+bodypose: {len(target_takes)} / {before}")
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
                    if file.endswith(".gz") or file.endswith(".vrs"):
                        continue
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
        assert processed_takes == len(target_takes), (
            f"Expected to process {len(target_takes)} takes, but processed {processed_takes}."
        )

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
            zip_path = run_zip(data_dir, output_dir, split, list_path, expected_bytes=split_sizes.get(split))
            if zip_path:
                zip_outputs.append(zip_path)

    total_bytes = sum(split_sizes.values())
    print("File size summary:")
    for split in target_splits:
        print(f"- {split}: {len(split_paths[split])} files, {format_bytes(split_sizes[split])}")
        if split_missing[split]:
            print(f"  missing: {len(split_missing[split])} files")
    print(f"- total: {format_bytes(total_bytes)}")
    print(f"Wrote list files to: {output_dir}")
    if zip_outputs:
        print("Zip outputs:")
        for zip_path in zip_outputs:
            print(f"- {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
