import argparse
import json
import math
import re
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick(d, keys):
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, dict) and "answer" in v:
                return v["answer"]
            return v
    return None


def get_answer(row):
    return pick(row, ["final", "answer", "prediction", "pred", "final_answer"])


def get_gold(row):
    return pick(row, ["gold", "gold_answer", "label", "target", "gt", "ground_truth"])


def get_id(row, idx):
    v = pick(row, ["id", "uid", "question_id", "image_id", "image", "image_path", "filename"])
    return str(v) if v is not None else str(idx)


def strict_norm(x):
    return str(x).strip().lower()


def relaxed_norm(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9.\-]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    return word_to_num.get(x, x)


def is_correct(pred, gold, relaxed=False):
    if pred is None or gold is None:
        return False

    if relaxed:
        return relaxed_norm(pred) == relaxed_norm(gold)

    return strict_norm(pred) == strict_norm(gold)


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    prob = 0.0
    for i in range(k + 1):
        prob += math.comb(n, i) * (0.5 ** n)

    return min(1.0, 2 * prob)


def get_question_type(row):
    return str(pick(row, ["question_type", "type", "category"]) or "reasoning")


def get_consistent(row):
    critic = row.get("critic") or row.get("critic_analysis") or row.get("reasoning_critic")

    if isinstance(critic, dict) and "consistent" in critic:
        return bool(critic["consistent"])

    if "consistent" in row:
        return bool(row["consistent"])

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--agents", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    agents = load_jsonl(args.agents)

    base_map = {get_id(r, i): r for i, r in enumerate(baseline)}
    agent_map = {get_id(r, i): r for i, r in enumerate(agents)}

    common_ids = [k for k in base_map.keys() if k in agent_map]

    if not common_ids:
        n = min(len(baseline), len(agents))
        pairs = [(str(i), baseline[i], agents[i]) for i in range(n)]
    else:
        pairs = [(k, base_map[k], agent_map[k]) for k in common_ids]

    n = len(pairs)

    base_strict = []
    agent_strict = []
    base_relaxed = []
    agent_relaxed = []

    type_stats = defaultdict(lambda: [0, 0, 0])

    critic_total = 0
    critic_consistent = 0
    consistent_correct = 0
    consistent_wrong = 0
    inconsistent_correct = 0
    inconsistent_wrong = 0

    disagreements = []

    for qid, b, a in pairs:
        gold = get_gold(a)
        if gold is None:
            gold = get_gold(b)

        bp = get_answer(b)
        ap = get_answer(a)

        bs = is_correct(bp, gold, relaxed=False)
        ars = is_correct(ap, gold, relaxed=False)
        br = is_correct(bp, gold, relaxed=True)
        ar = is_correct(ap, gold, relaxed=True)

        base_strict.append(bs)
        agent_strict.append(ars)
        base_relaxed.append(br)
        agent_relaxed.append(ar)

        qt = get_question_type(a)
        type_stats[qt][0] += 1
        type_stats[qt][1] += int(br)
        type_stats[qt][2] += int(ar)

        cons = get_consistent(a)
        if cons is not None:
            critic_total += 1
            if cons:
                critic_consistent += 1
                if ar:
                    consistent_correct += 1
                else:
                    consistent_wrong += 1
            else:
                if ar:
                    inconsistent_correct += 1
                else:
                    inconsistent_wrong += 1

        if br != ar:
            disagreements.append({
                "id": qid,
                "question": pick(a, ["question"]),
                "gold": gold,
                "baseline": bp,
                "agents": ap,
                "baseline_relaxed_correct": br,
                "agents_relaxed_correct": ar,
            })

    base_strict_acc = sum(base_strict) / n
    agent_strict_acc = sum(agent_strict) / n
    base_relaxed_acc = sum(base_relaxed) / n
    agent_relaxed_acc = sum(agent_relaxed) / n

    b_only_agent = sum((not b) and a for b, a in zip(base_relaxed, agent_relaxed))
    c_only_base = sum(b and (not a) for b, a in zip(base_relaxed, agent_relaxed))
    both = sum(b and a for b, a in zip(base_relaxed, agent_relaxed))
    neither = sum((not b) and (not a) for b, a in zip(base_relaxed, agent_relaxed))
    p_val = mcnemar_exact_p(b_only_agent, c_only_base)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# val_easy (n={n} paired)")
    lines.append("")
    lines.append("| Metric | Baseline | Agents | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Strict EM | {base_strict_acc:.3f} | {agent_strict_acc:.3f} | {agent_strict_acc - base_strict_acc:+.3f} |")
    lines.append(f"| Relaxed | {base_relaxed_acc:.3f} | {agent_relaxed_acc:.3f} | {agent_relaxed_acc - base_relaxed_acc:+.3f} |")
    lines.append("")
    lines.append("## Per question_type (relaxed)")
    lines.append("")
    lines.append("| Type | n | Baseline | Agents | Delta |")
    lines.append("|---|---:|---:|---:|---:|")

    for qt, (cnt, bc, ac) in type_stats.items():
        ba = bc / cnt
        aa = ac / cnt
        lines.append(f"| {qt} | {cnt} | {ba:.3f} | {aa:.3f} | {aa - ba:+.3f} |")

    lines.append("")
    lines.append(
        f"**McNemar (relaxed):** b={b_only_agent} "
        f"(only agents right), c={c_only_base} "
        f"(only baseline right), both={both}, neither={neither}, p={p_val:.4f}."
    )

    if critic_total > 0:
        lines.append("")
        lines.append("## Critic analysis")
        lines.append("")
        lines.append(f"- P(consistent) = {critic_consistent / critic_total:.3f}")
        lines.append(f"- consistent ∧ correct : {consistent_correct}")
        lines.append(f"- consistent ∧ wrong : {consistent_wrong}")
        lines.append(f"- inconsistent ∧ correct : {inconsistent_correct}")
        lines.append(f"- inconsistent ∧ wrong : {inconsistent_wrong}")

    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")

    with open(out / "disagreements_val_easy.jsonl", "w", encoding="utf-8") as f:
        for r in disagreements:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n".join(lines))
    print()
    print(f"Wrote {out / 'results.md'}")
    print(f"Wrote {out / 'disagreements_val_easy.jsonl'}")


if __name__ == "__main__":
    main()