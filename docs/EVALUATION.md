# Evaluation record

## What was checked

Four saved prediction files were minimized to question ID, gold/prediction, question type, critic consistency, latency, and error status. The included files retain the fields necessary for score auditing; image paths, questions, prompts and explanations were not copied. Source file hashes are in `source-manifest.json`.

There are 30 unique matching IDs per system per split, with matching gold labels and no recorded error rows. `scripts/verify_results.py` checks these conditions before scoring. The original `aggregate_dvqa.py` is also retained, but its generic inner join can exclude failed or unmatched rows; use the strict verifier for the bundled evidence.

| Split | Strict baseline | Strict agents | Relaxed baseline | Relaxed agents |
|---|---:|---:|---:|---:|
| easy, n=30 | 50.0% | 66.7% | 53.3% | 70.0% |
| hard, n=30 | 36.7% | 70.0% | 36.7% | 73.3% |

These reproduce the source result table. They do not demonstrate that the revised demo or current full pipeline would regenerate the same predictions.

## Scorer semantics and limitations

The inherited `strict_exact_match` lowercases, removes punctuation except hyphens, collapses whitespace, and compares strings. It is not literal string equality. In particular, punctuation removal also removes decimal points; it must not be treated as a robust general numeric evaluator. The function remains unchanged here to preserve historical scoring provenance.

The relaxed metric additionally folds number words, removes selected articles/units, and permits 5% relative numeric tolerance. It is a project-specific supplemental metric. Neither the name nor the source comments establish equivalence to the current official DVQA evaluation protocol.

Paired McNemar p-values for relaxed matching are approximately 0.2266 on easy and 0.01921 on hard, as recomputed by the inherited code. The samples are small; selection history, repeated tuning, and image-level dependence have not been reconstructed. Do not interpret these exploratory p-values as proof of generalization.

## Critic behavior

On hard, two wrong answers were marked consistent and fourteen correct answers were marked inconsistent under relaxed scoring. Therefore a critic-consistent flag is not an answer-correctness guarantee. In the offline blackboard implementation, the evidence critic merely checks that a nonempty answer has nonempty support details; it does not independently recompute the evidence.

## Reproducibility boundary

The report names GPT-4o-mini. Original baseline metadata records `gpt-4o-mini`, 30 predictions, and zero errors. Agent metadata records `multi_agent_pipeline` and 30 predictions, but omits its complete runtime model settings and package versions. The local source HEAD and file hashes are available; the exact experiment commit is not verified.

The source configuration defaulted the vision model to a different identifier from the report's backbone. This is further reason to make model settings explicit for new runs rather than infer historical settings from current defaults. No claim about current model price, availability, or live benchmark reproduction is made.

Future experiments should freeze a held-out question list, record image-level splits and seeds, store all model/settings identifiers, report failures in the denominator, and measure latency and calls alongside accuracy. Any revised scorer should get a new result table rather than silently overwrite this historical record.
