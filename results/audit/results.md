# DVQA Preliminary Results

## val_easy (n=30 paired)

| Metric | Baseline | Agents | Delta |
|---|---:|---:|---:|
| Strict EM | 0.500 | 0.667 | +0.167 |
| Relaxed   | 0.533 | 0.700 | +0.167 |

### Per question_type (relaxed)

| Type | n | Baseline | Agents | Delta |
|---|---:|---:|---:|---:|
| reasoning | 30 | 0.533 | 0.700 | +0.167 |

**McNemar (relaxed)**: b=8 (only agents right), c=3 (only baseline right), both=13, neither=6, **p = 0.2266**.

### Critic analysis

- P(consistent) = 0.467
- consistent ∧ correct   : 9
- consistent ∧ wrong     : 5
- inconsistent ∧ correct : 12
- inconsistent ∧ wrong   : 4

## val_hard (n=30 paired)

| Metric | Baseline | Agents | Delta |
|---|---:|---:|---:|
| Strict EM | 0.367 | 0.700 | +0.333 |
| Relaxed   | 0.367 | 0.733 | +0.367 |

### Per question_type (relaxed)

| Type | n | Baseline | Agents | Delta |
|---|---:|---:|---:|---:|
| reasoning | 30 | 0.367 | 0.733 | +0.367 |

**McNemar (relaxed)**: b=15 (only agents right), c=4 (only baseline right), both=7, neither=4, **p = 0.01921**.

### Critic analysis

- P(consistent) = 0.333
- consistent ∧ correct   : 8
- consistent ∧ wrong     : 2
- inconsistent ∧ correct : 14
- inconsistent ∧ wrong   : 6
