"""Audit saved paired predictions, without datasets, credentials or model calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.aggregate_dvqa import summarize_split


def load_rows(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = int(row["question_id"])
        if qid in rows:
            raise ValueError(f"Duplicate question ID {qid} in {path.name}")
        if row.get("error"):
            raise ValueError(f"Failed prediction {qid}; cannot silently exclude it")
        if not isinstance(row.get("gold"), str) or not isinstance(row.get("pred"), str):
            raise ValueError(f"Expected string gold/prediction for {qid}")
        rows[qid] = row
    return rows


def audit(root: Path = ROOT) -> list[dict]:
    historical = root / "results" / "historical"
    reported = {r["split"]: r for r in json.loads((historical / "reported_results.json").read_text(encoding="utf-8"))}
    results = []
    for split in ("val_easy", "val_hard"):
        baseline = load_rows(historical / f"baseline_{split}.jsonl")
        agents = load_rows(historical / f"agents_{split}.jsonl")
        if baseline.keys() != agents.keys() or len(baseline) != 30:
            raise ValueError(f"{split}: expected exactly 30 matching IDs")
        pairs = []
        for qid in sorted(baseline):
            b, a = baseline[qid], agents[qid]
            if b["gold"] != a["gold"] or b["question_type"] != a["question_type"]:
                raise ValueError(f"{split}/{qid}: mismatched gold or question type")
            pairs.append((b, a))
        computed = summarize_split(pairs, split)
        for metric in ("baseline_strict_em", "agents_strict_em", "baseline_relaxed", "agents_relaxed"):
            if abs(computed[metric] - reported[split][metric]) > 1e-12:
                raise ValueError(f"{split}: saved report disagrees on {metric}")
        results.append(computed)
    return results


if __name__ == "__main__":
    for r in audit():
        print(f"PASS {r['split']}: n={r['n_paired']}, strict EM "
              f"{r['baseline_strict_em']:.1%} -> {r['agents_strict_em']:.1%}, "
              f"relaxed {r['baseline_relaxed']:.1%} -> {r['agents_relaxed']:.1%}")
    print("Historical score audit only; no inference or benchmark rerun performed.")
