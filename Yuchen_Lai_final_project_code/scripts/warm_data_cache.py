from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import AudioFeatureTransform, DEAMDataset, VEATICDataset, VideoFrameTransform


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm on-disk preprocessing caches for DEAM and VEATIC.")
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--audio-frames", type=int, default=256)
    parser.add_argument("--veatic-split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--max-deam", type=int, default=None)
    parser.add_argument("--max-veatic", type=int, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    audio_transform = AudioFeatureTransform(target_num_frames=args.audio_frames)
    video_transform = VideoFrameTransform(size=args.frame_size)

    deam = DEAMDataset(
        audio_zip_path=PROJECT_ROOT / "data/deam/DEAM_audio.zip",
        annotations_dir=PROJECT_ROOT / "data/deam/annotations/annotations",
        cache_dir=PROJECT_ROOT / "data/cache/deam",
        transform=audio_transform,
    )
    warm_dataset("deam", deam, max_items=args.max_deam)

    splits = ["train", "test"] if args.veatic_split == "all" else [args.veatic_split]
    for split in splits:
        veatic = VEATICDataset(
            zip_path=PROJECT_ROOT / "data/veatic/VEATIC.zip",
            cache_dir=PROJECT_ROOT / "data/cache/veatic",
            clip_length=args.clip_length,
            stride=args.clip_stride,
            split=split,
            transform=video_transform,
            audio_transform=audio_transform,
        )
        warm_dataset(f"veatic_{split}", veatic, max_items=args.max_veatic)


def warm_dataset(name: str, dataset, max_items: int | None = None) -> None:
    total = len(dataset) if max_items is None else min(len(dataset), max_items)
    print(f"Warming {name}: {total} items")
    for index in range(total):
        dataset[index]
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == total:
            print(f"  {name} {index + 1}/{total}", flush=True)


if __name__ == "__main__":
    main()
