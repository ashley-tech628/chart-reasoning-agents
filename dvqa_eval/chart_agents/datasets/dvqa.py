"""DVQA loader.

DVQA (Kafle et al., CVPR 2018) ships as three JSON files of QA records and
one image folder. This loader just streams records and resolves their
absolute image path; sampling lives in `scripts/sample_dvqa.py`.

Expected layout (after downloading from
https://github.com/kushalkafle/DVQA_dataset):

    <qa_root>/
        train_qa.json
        val_easy_qa.json
        val_hard_qa.json
    <image_root>/
        bar_train_xxxxxxxx.png
        bar_val_easy_xxxxxxxx.png
        bar_val_hard_xxxxxxxx.png

`qa_root` and `image_root` are often the same directory.

Raw QA record schema (as shipped):
    {
        "question_id": int,
        "image": "bar_val_easy_00000001.png",
        "question": "...",
        "answer": "...",
        "template_id": "structure" | "data" | "reasoning",
        "answer_bbox": [x, y, w, h] | []
    }

We normalize `template_id` → `question_type` and `answer_bbox` → `bbox_answer`
on the way into `DVQAExample` so downstream code can use the friendlier names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal, Optional

Split = Literal["train", "val_easy", "val_hard"]
QuestionType = Literal["structure", "data", "reasoning"]


@dataclass
class DVQAExample:
    """One question-answer record with the image resolved to an absolute path."""

    question_id: int
    image_filename: str
    image_path: str
    question: str
    answer: str
    question_type: QuestionType
    bbox_answer: Optional[List[float]]  # [x, y, w, h] or None
    split: Split

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "image": self.image_filename,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "bbox_answer": self.bbox_answer,
            "split": self.split,
        }


def _qa_json_path(qa_root: Path, split: Split) -> Path:
    return qa_root / f"{split}_qa.json"


def load_split(
    qa_root: str | Path,
    image_root: str | Path,
    split: Split,
) -> List[DVQAExample]:
    """Load all QA records for one split. Returns a list (DVQA val splits fit
    easily in memory: ~580K records is ~150 MB)."""
    qa_root = Path(qa_root)
    image_root = Path(image_root)
    path = _qa_json_path(qa_root, split)
    if not path.exists():
        raise FileNotFoundError(f"DVQA QA file not found: {path}")

    with open(path) as f:
        records = json.load(f)

    examples: List[DVQAExample] = []
    for r in records:
        img = r["image"]
        bbox = r.get("answer_bbox") or r.get("bbox_answer") or None
        if isinstance(bbox, list) and len(bbox) == 0:
            bbox = None
        examples.append(
            DVQAExample(
                question_id=int(r["question_id"]),
                image_filename=img,
                image_path=str(image_root / img),
                question=r["question"],
                answer=str(r["answer"]),
                question_type=r.get("template_id") or r["question_type"],
                bbox_answer=bbox,
                split=split,
            )
        )
    return examples


def iter_split(
    qa_root: str | Path,
    image_root: str | Path,
    split: Split,
) -> Iterator[DVQAExample]:
    """Streaming variant. Reuses `load_split` since DVQA val files fit in RAM
    but exposes an iterator for code that prefers it."""
    for ex in load_split(qa_root, image_root, split):
        yield ex


def records_to_examples(
    records: list,
    split: Split,
    image_root: str | Path,
) -> List[DVQAExample]:
    """Promote a list of raw QA dicts (as returned by `read_qa_split_from_tar`)
    into `DVQAExample` objects, resolving each image's path against
    `image_root`. The image directory does not need to exist yet — paths are
    resolved lazily; downstream code extracts the actual PNGs there."""
    image_root = Path(image_root)
    examples: List[DVQAExample] = []
    for r in records:
        img = r["image"]
        bbox = r.get("answer_bbox") or r.get("bbox_answer") or None
        if isinstance(bbox, list) and len(bbox) == 0:
            bbox = None
        examples.append(
            DVQAExample(
                question_id=int(r["question_id"]),
                image_filename=img,
                image_path=str(image_root / img),
                question=r["question"],
                answer=str(r["answer"]),
                question_type=r.get("template_id") or r["question_type"],
                bbox_answer=bbox,
                split=split,
            )
        )
    return examples


def load_split_from_tar(
    qa_tar_path: str | Path,
    image_root: str | Path,
    split: Split,
) -> List[DVQAExample]:
    """Stream `qa.tar.gz`, parse just `{split}_qa.json` from it in memory,
    and return `DVQAExample` records. `image_root` is where the images
    *will* live after extraction; it does not need to exist yet."""
    # Local import to keep `dvqa.py` importable even without the archives
    # module available (e.g. in environments using pre-extracted DVQA).
    from chart_agents.datasets.dvqa_archives import read_qa_split_from_tar
    by_split = read_qa_split_from_tar(qa_tar_path, [split])
    return records_to_examples(by_split[split], split, image_root)
