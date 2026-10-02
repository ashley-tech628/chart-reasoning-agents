# Reproducing the demo and auditing results

## Offline path (locally checked)

From the repository root:

```bash
python -m pip install -r requirements-demo.txt
python scripts/run_blackboard_demo.py --question "Which quarter has the largest gap between Product A and Product B?" --no-save
python scripts/verify_results.py
python -m unittest discover -s tests -v
```

Expect Q3 and intermediate gaps 7, 11, 13, 8. The tests change Q1's input value and expect Q1, so the demo is not simply returning a constant answer. The offline path imports no provider configuration and makes no model calls. A reference image supplied through `--image` is not parsed.

`examples/sample_chart.json` follows the current schema: `groups` are Q1–Q4 and `categories` are Product A/B. Each row has `group`, `category`, `value`, and an optional `visual_mark_index`.

To rebuild the human-readable historical summary:

```bash
python scripts/aggregate_dvqa.py --baseline-easy results/historical/baseline_val_easy.jsonl --baseline-hard results/historical/baseline_val_hard.jsonl --agents-easy results/historical/agents_val_easy.jsonl --agents-hard results/historical/agents_val_hard.jsonl --out results/audit
```

Run the stricter verifier first. The general aggregator inherits inner-join/error-filtering behavior from the research project; it is not a completeness validator.

## New image/LLM experiments (not run in this edition)

Create an isolated Python environment, install `requirements.txt`, and configure `.env` from `.env.example`. On PowerShell use `Copy-Item .env.example .env`; on POSIX use `cp .env.example .env`. Set a key locally, never in source or a notebook. The plain single-shot baseline script does not load `.env` itself; for that script, set `OPENAI_API_KEY` in the shell environment or use an environment loader. The CrewAI path loads project configuration through python-dotenv.

For `PERCEPTION_BACKEND=offline_cv`, install the native Tesseract OCR program and make it available on PATH. The Python pytesseract package alone is insufficient. Generate the synthetic PNG using `scripts/make_sample_chart.py`, then run the image pipeline as shown in the README. The source parser is tuned to relatively clean bar charts.

For new DVQA experiments, obtain the dataset separately under its terms. Place `qa.tar.gz` and `images.tar.gz` in `data/dvqa/`. The sampler can also read pre-extracted directories. Do not commit raw archives or extracted images.

```bash
python scripts/sample_dvqa.py --dvqa-data data/dvqa --n 30 --seed 252 --max-per-image 1 --reasoning-only --out samples
```

Inspect the emitted manifest and JSONL names, then pass those exact paths to the runners. Example, if the sampler emits `samples/val_easy_n30_reasoning.jsonl`:

```bash
python scripts/run_dvqa_baseline.py --samples samples/val_easy_n30_reasoning.jsonl --out eval_runs/new-baseline-easy --model gpt-4o-mini
python scripts/run_dvqa_agents.py --samples samples/val_easy_n30_reasoning.jsonl --out eval_runs/new-agents-easy
```

Repeat for hard, then aggregate. This creates a **new experiment**, not the original historical sample. The `--model` argument above is the historical identifier, not a promise of current provider availability. Choose and record a supported model for a new run.

Record the Git revision, clean/dirty state, package versions, complete model settings, seed, dataset version, question IDs, image grouping, failures, runtime and usage. Use new output directories when changing models or settings: inherited resume behavior skips existing question IDs and can otherwise mix experiments.

The inherited full dependency file contains broad lower bounds. There is no tested full-environment lockfile. The local packaging environment had only the dependencies needed for offline checks, so the provider/OCR path remains unverified.
