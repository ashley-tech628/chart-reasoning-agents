from pathlib import Path
import argparse
from scripts.aggregate_dvqa import load_predictions, pair_records, summarize_split

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--agents", required=True)
args = parser.parse_args()

base = load_predictions(Path(args.baseline))
agents = load_predictions(Path(args.agents))
pairs = pair_records(base, agents)
s = summarize_split(pairs, "val_easy")

print("\n=== EASY ONLY RESULT ===")
print(f"paired n = {s['n_paired']}")
print(f"Baseline Strict EM : {s['baseline_strict_em']:.3f}")
print(f"Agents Strict EM   : {s['agents_strict_em']:.3f}")
print(f"Strict delta       : {s['agents_strict_em'] - s['baseline_strict_em']:+.3f}")
print()
print(f"Baseline Relaxed   : {s['baseline_relaxed']:.3f}")
print(f"Agents Relaxed     : {s['agents_relaxed']:.3f}")
print(f"Relaxed delta      : {s['agents_relaxed'] - s['baseline_relaxed']:+.3f}")

m = s["mcnemar_relaxed"]
print()
print("McNemar relaxed:")
print(f"only agents right   = {m['b_only_a_right']}")
print(f"only baseline right = {m['c_only_b_right']}")
print(f"both right          = {m['both_right']}")
print(f"neither right       = {m['neither_right']}")
print(f"p-value             = {m['p_value']:.4g}")
