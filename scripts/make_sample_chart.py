"""Generate a simple grouped bar chart for end-to-end smoke testing.

Mirrors the example used throughout the project report (Product A vs B
across Q1..Q4) so the question "Which quarter has the largest gap?" has a
known ground-truth answer of Q3 (gap = 13).
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    import matplotlib.pyplot as plt  # imported lazily

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    product_a = [42, 58, 71, 88]
    product_b = [35, 47, 58, 80]

    x = list(range(len(quarters)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], product_a, width, label="Product A")
    ax.bar([i + width / 2 for i in x], product_b, width, label="Product B")

    ax.set_xlabel("Quarter")
    ax.set_ylabel("Sales")
    ax.set_title("Quarterly Sales by Product")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.legend()

    for i, (a, b) in enumerate(zip(product_a, product_b)):
        ax.text(i - width / 2, a + 1, str(a), ha="center", fontsize=9)
        ax.text(i + width / 2, b + 1, str(b), ha="center", fontsize=9)

    out = Path("examples/sample_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved sample chart to {out}")


if __name__ == "__main__":
    main()
