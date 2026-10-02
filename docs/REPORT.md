# Grounded Chart Reasoning: Portfolio Technical Report

Team project: Chen Si, Xinzi Huang, Quanquan Peng, and Xinying Liu. Adapted from the local CSE 252D Spring 2026 team report and inspected source code; student IDs are omitted. This document is a concise technical synthesis, not a reproduction of the full original PDF.

## Problem

Chart question answering requires more than detecting text. A system must associate an axis label with a legend entry, recover a numerical value, choose an operation, and return an answer of the requested type. A correct numeric intermediate can still yield an incorrect answer if the user asks for a label. OCR failures can also remove the target entirely from the structured representation.

## Design

The system separates visual interpretation from reasoning over typed chart data. A ChartDSL cell contains an axis-side group, a legend-side category, a value, and an optional visual-mark index. This convention makes group-versus-category confusion explicit.

The LLM-assisted implementation uses sequential CrewAI stages for perception, representation, question parsing, evidence selection, reasoning, short-answer extraction, and critique. Its source includes deterministic answer overrides and a bounded reasoning-only repair loop. Repair does not recover missing bars or rerun all upstream stages automatically.

The second implementation uses deterministic Python components that communicate through a shared blackboard. The router selects relevant solver branches, solvers post candidate answers and computed details, critics post feedback, and a judge ranks candidates using confidence heuristics, support, objections, and answer-type compatibility. The judge is not simply a majority vote. Its scores are heuristics, not calibrated probabilities.

## Demonstration

The synthetic quarterly-sales fixture contains Product A values 42, 58, 71, 88 and Product B values 35, 47, 58, 80. For a largest-gap question, the four differences are 7, 11, 13, 8, making Q3 the answer. The saved trace exposes route selection, candidate evidence, feedback, and final selection. Changing the fixture changes the answer; the demo does not use an image parser or a hosted model.

## Experimental evidence

The original report evaluates paired samples of 30 DVQA reasoning questions from each of val_easy and val_hard. The source result table reports strict EM increases from 50.0% to 66.7% on easy and 36.7% to 70.0% on hard. Recomputing the table from the saved per-question predictions reproduces these figures.

This is an audit of historical predictions, not a fresh model evaluation. The source record does not fully bind the predictions to a pinned dependency environment, complete agent model configuration, and exact source revision. The offline demo and packaging fixes postdate those results. Metric normalization and statistical limitations are described in `EVALUATION.md`.

## Failure modes and tradeoffs

1. **Perception bottleneck:** missing labels or incomplete bar extraction can make downstream reasoning wrong even when the arithmetic is correct.
2. **Role confusion:** confusing legend categories and axis groups can return the wrong label; typed schemas and answer-type checks make this easier to diagnose.
3. **Verification limits:** the blackboard evidence check accepts the existence of support details, not their independent truth. The historical LLM critic has both false accepts and false rejects.
4. **Duplicate cells:** the current ChartDSL validator keeps the largest value for duplicate group/category pairs. That is a heuristic that can discard uncertainty; it is not a general resolution strategy for conflicting observations.
5. **Rule coverage:** deterministic routing and solvers support a subset of English question patterns. They are useful for inspectable demonstrations but do not establish open-ended language understanding.
6. **Cost and latency:** multi-stage calls introduce overhead. This edition makes no current cost-saving claim because exact historical token usage and complete configuration are not available.

## Engineering validation in this edition

Offline tests cover the synthetic answer, its intermediate values, all four message rounds, a changed-data case, empty-chart abstention, label routing, schema roundtripping, saved-score agreement, duplicate IDs, missing predictions, and recorded failures. These checks require only Pydantic plus the standard library. CI is configured to run them, but hosted CI and full LLM/OCR execution were not run during local packaging.

## Next experiments

Use a larger held-out set with explicit image-level independence, freeze prompts/settings, record exact revisions, compare deterministic and LLM-assisted routes under the same input representation, and independently verify arithmetic and cited evidence. Keep quality, latency, calls, and failures visible together.
