"""Sample DVQA val_easy + val_hard down to a manageable size, stratified by
question_type, with at most K questions drawn from any one image.

This script reads DVQA directly from its tar.gz archives — it never fully
unpacks them. Two streaming passes:

    1. Open qa.tar.gz, parse `{split}_qa.json` for each requested split
       straight into memory.
    2. Sample → produces a list of needed image filenames.
    3. Open images.tar.gz, stream-extract just those images to
       `<dvqa-data>/extracted_images/` (or wherever --images-out points).
       Bails out as soon as the full needed set has been found.

Each output .jsonl line is one DVQAExample.to_dict() — the same format
`run_dvqa_baseline.py` / `run_dvqa_agents.py` consume.

Usage (default layout: data/dvqa/{qa,images,metadata}.tar.gz):
    python scripts/sample_dvqa.py \\
        --dvqa-data data/dvqa \\
        --n 500 --seed 252 --max-per-image 3 \\
        --splits val_easy val_hard \\
        --out samples/

Reasoning-only sample:
    python scripts/sample_dvqa.py \\
        --dvqa-data data/dvqa \\
        --n 500 --seed 252 --max-per-image 3 \\
        --splits val_easy val_hard \\
        --reasoning-only \\
        --out samples/

Or, with pre-extracted DVQA on disk:
    python scripts/sample_dvqa.py \\
        --qa-root /path/to/DVQA_dataset \\
        --image-root /path/to/DVQA_dataset/images \\
        --n 500 --seed 252 --out samples/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Make the package importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvqa_eval.chart_agents.datasets.dvqa import (  # noqa: E402
    DVQAExample,
    Split,
    load_split,
    records_to_examples,
)
from dvqa_eval.chart_agents.datasets.dvqa_archives import (  # noqa: E402
    extract_images_from_tar,
    peek_tar_summary,
    read_qa_split_from_tar,
)


# Population proportions per DVQA paper Table 1 (whole dataset; very similar
# in each split). We stratify the sample to match.
TARGET_PROPORTIONS: Dict[str, float] = {
    "structure": 0.135,  # 471,108 / 3,487,194
    "data": 0.319,       # 1,113,704 / 3,487,194
    "reasoning": 0.463,  # 1,613,974 / 3,487,194
    # remainder (~0.08) absorbed into reasoning when rounding
}


def stratified_sample(
    examples: List[DVQAExample],
    n: int,
    rng: random.Random,
    max_per_image: int = 3,
    reasoning_only: bool = False,
) -> List[DVQAExample]:
    """Sample DVQA examples, with a per-image cap.

    By default, this performs stratified sampling by question_type using
    TARGET_PROPORTIONS. If reasoning_only=True, it ignores the stratified
    proportions and draws up to n examples only from question_type='reasoning'.

    The algorithm:
      1. Bucket by question_type.
      2. Shuffle each bucket.
      3. Pull questions one at a time per bucket, rejecting any whose image
         has already contributed `max_per_image` questions to the sample.
      4. Stop when we hit the quota for that bucket.
    """
    buckets: Dict[str, List[DVQAExample]] = defaultdict(list)
    for ex in examples:
        buckets[ex.question_type].append(ex)
    for qtype in buckets:
        rng.shuffle(buckets[qtype])

    if reasoning_only:
        image_counts: Counter = Counter()
        chosen: List[DVQAExample] = []
        pool = buckets.get("reasoning", [])

        for ex in pool:
            if len(chosen) >= n:
                break
            if image_counts[ex.image_filename] >= max_per_image:
                continue
            chosen.append(ex)
            image_counts[ex.image_filename] += 1

        if len(chosen) < n:
            print(
                f"WARNING: only got {len(chosen)}/{n} for question_type=reasoning "
                f"(pool size {len(pool)}, max-per-image={max_per_image})",
                file=sys.stderr,
            )

        rng.shuffle(chosen)
        return chosen

    # Target counts per question_type. Round so the total equals n.
    targets = {q: int(round(n * p)) for q, p in TARGET_PROPORTIONS.items()}
    drift = n - sum(targets.values())
    if drift != 0:
        targets["reasoning"] = max(0, targets["reasoning"] + drift)

    image_counts: Counter = Counter()
    chosen: List[DVQAExample] = []

    for qtype, target in targets.items():
        pool = buckets.get(qtype, [])
        taken = 0
        for ex in pool:
            if taken >= target:
                break
            if image_counts[ex.image_filename] >= max_per_image:
                continue
            chosen.append(ex)
            image_counts[ex.image_filename] += 1
            taken += 1
        if taken < target:
            print(
                f"WARNING: only got {taken}/{target} for question_type={qtype} "
                f"(pool size {len(pool)}, max-per-image={max_per_image})",
                file=sys.stderr,
            )

    rng.shuffle(chosen)
    return chosen


def write_jsonl(records: List[DVQAExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in records:
            f.write(json.dumps(ex.to_dict()) + "\n")


# ---------------------------------------------------------------------------
# Two source modes: tar.gz archives vs pre-extracted DVQA_dataset/
# ---------------------------------------------------------------------------


def _load_from_tars(
    qa_tar: Path,
    splits: List[Split],
    image_root_for_paths: Path,
) -> Dict[Split, List[DVQAExample]]:
    """Stream qa.tar.gz once, building DVQAExample lists for all splits."""
    print(f"\nStreaming QA records from {qa_tar} ...", flush=True)
    by_split_records = read_qa_split_from_tar(qa_tar, splits)
    result: Dict[Split, List[DVQAExample]] = {}
    for split in splits:
        records = by_split_records[split]
        exs = records_to_examples(records, split, image_root_for_paths)
        print(f"  {split}: {len(exs):,} QA records", flush=True)
        result[split] = exs
    return result


def _load_from_dirs(
    qa_root: Path,
    image_root: Path,
    splits: List[Split],
) -> Dict[Split, List[DVQAExample]]:
    """Pre-extracted DVQA on disk."""
    result: Dict[Split, List[DVQAExample]] = {}
    for split in splits:
        print(f"\nLoading {split}_qa.json from {qa_root} ...", flush=True)
        exs = load_split(qa_root, image_root, split)
        print(f"  {split}: {len(exs):,} QA records", flush=True)
        result[split] = exs
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_paths(args: argparse.Namespace) -> dict:
    """Figure out where the QA/images live based on the CLI flags.

    Modes:
      A) --dvqa-data <dir>          → looks for {dir}/qa.tar.gz, images.tar.gz
      B) --qa-tar / --images-tar    → individual paths, override mode A
      C) --qa-root / --image-root   → pre-extracted directories
    """
    if args.qa_root is not None or args.image_root is not None:
        if args.qa_root is None or args.image_root is None:
            sys.exit("ERROR: --qa-root and --image-root must be given together.")
        return {
            "mode": "dirs",
            "qa_root": Path(args.qa_root),
            "image_root": Path(args.image_root),
            "images_out": Path(args.image_root),
        }

    qa_tar = Path(args.qa_tar) if args.qa_tar else None
    images_tar = Path(args.images_tar) if args.images_tar else None

    if args.dvqa_data is not None:
        base = Path(args.dvqa_data)
        qa_tar = qa_tar or (base / "qa.tar.gz")
        images_tar = images_tar or (base / "images.tar.gz")

    if qa_tar is None or images_tar is None:
        sys.exit(
            "ERROR: must give either --dvqa-data, or both --qa-tar and --images-tar, "
            "or --qa-root and --image-root."
        )

    for p in (qa_tar, images_tar):
        if not p.exists():
            sys.exit(f"ERROR: file not found: {p}")

    if args.images_out:
        images_out = Path(args.images_out)
    elif args.dvqa_data:
        images_out = Path(args.dvqa_data) / "extracted_images"
    else:
        images_out = images_tar.parent / "extracted_images"

    return {
        "mode": "tar",
        "qa_tar": qa_tar,
        "images_tar": images_tar,
        "images_out": images_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stratified DVQA sampler with streaming tar.gz access.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Source: tar archives (preferred)
    parser.add_argument(
        "--dvqa-data",
        default=None,
        help="Folder holding qa.tar.gz / images.tar.gz / metadata.tar.gz.",
    )
    parser.add_argument("--qa-tar", default=None, help="Path to qa.tar.gz (overrides --dvqa-data).")
    parser.add_argument(
        "--images-tar", default=None, help="Path to images.tar.gz (overrides --dvqa-data)."
    )
    parser.add_argument(
        "--images-out",
        default=None,
        help="Directory to extract sampled images into (default: <dvqa-data>/extracted_images).",
    )
    # Source: pre-extracted dirs (backwards-compatible)
    parser.add_argument(
        "--qa-root", default=None, help="Folder with already-extracted {split}_qa.json files."
    )
    parser.add_argument(
        "--image-root", default=None, help="Folder with already-extracted bar_*.png files."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val_easy", "val_hard"],
        choices=["train", "val_easy", "val_hard"],
    )
    parser.add_argument("--n", type=int, default=500, help="Questions per split.")
    parser.add_argument("--seed", type=int, default=252)
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument(
        "--reasoning-only",
        action="store_true",
        help="Sample only reasoning questions instead of stratifying across question types.",
    )
    parser.add_argument("--out", type=Path, default=Path("samples"))
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Sample only; do not extract images. (Useful if you want to inspect "
        "the JSONL first and extract later.)",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Before sampling, print a short summary of qa.tar.gz (slow: streams "
        "the whole archive once just to count members). Skip for routine runs.",
    )
    args = parser.parse_args()

    paths = _resolve_paths(args)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # ----- 1. Load QA records (from tar or from disk) -----
    if paths["mode"] == "tar" and args.peek:
        print("\nPeek qa.tar.gz:", flush=True)
        print(json.dumps(peek_tar_summary(paths["qa_tar"]), indent=2))
        print("\nPeek images.tar.gz: (skipping — would stream the whole archive)")

    if paths["mode"] == "tar":
        by_split = _load_from_tars(paths["qa_tar"], args.splits, paths["images_out"])
    else:
        by_split = _load_from_dirs(paths["qa_root"], paths["image_root"], args.splits)

    # ----- 2. Sample each split -----
    summary: Dict[str, dict] = {}
    needed_image_filenames: set = set()
    written_paths: List[Path] = []

    for split in args.splits:
        examples = by_split[split]
        print(f"\n=== {split} ===")
        sample = stratified_sample(
            examples,
            n=args.n,
            rng=rng,
            max_per_image=args.max_per_image,
            reasoning_only=args.reasoning_only,
        )

        type_counts = Counter(ex.question_type for ex in sample)
        bbox_count = sum(1 for ex in sample if ex.bbox_answer is not None)
        print(f"  sampled {len(sample)}: {dict(type_counts)}")
        print(f"  with non-empty bbox_answer: {bbox_count}")

        suffix = "_reasoning" if args.reasoning_only else ""
        out_path = args.out / f"{split}_n{args.n}{suffix}.jsonl"
        write_jsonl(sample, out_path)
        written_paths.append(out_path)
        print(f"  wrote {out_path}")

        needed_image_filenames.update(ex.image_filename for ex in sample)
        summary[split] = {
            "n_sampled": len(sample),
            "n_population": len(examples),
            "question_type_counts": dict(type_counts),
            "n_with_bbox_answer": bbox_count,
            "out_path": str(out_path),
        }

    # ----- 3. Stream-extract images from images.tar.gz -----
    extraction_report: Optional[dict] = None
    if paths["mode"] == "tar" and not args.no_extract:
        print(
            f"\n=== Extracting {len(needed_image_filenames)} images "
            f"from {paths['images_tar']} into {paths['images_out']} ==="
        )
        extraction_report = extract_images_from_tar(
            paths["images_tar"],
            needed_image_filenames,
            paths["images_out"],
        )
        if extraction_report["missing"]:
            print(
                f"WARNING: {len(extraction_report['missing'])} image(s) not found "
                f"in images.tar.gz. First few: {extraction_report['missing'][:5]}",
                file=sys.stderr,
            )
    elif paths["mode"] == "tar" and args.no_extract:
        print(
            f"\n(--no-extract set; skipping image extraction. "
            f"Sample JSONLs reference paths under {paths['images_out']}.)"
        )

    # ----- 4. Persist a summary so the run is auditable -----
    summary_path = args.out / "sampling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n": args.n,
                "max_per_image": args.max_per_image,
                "reasoning_only": args.reasoning_only,
                "splits": summary,
                "source_mode": paths["mode"],
                "qa_source": str(paths.get("qa_tar") or paths.get("qa_root")),
                "images_source": str(paths.get("images_tar") or paths.get("image_root")),
                "images_out": str(paths["images_out"]),
                "extraction": extraction_report,
            },
            f,
            indent=2,
        )
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
