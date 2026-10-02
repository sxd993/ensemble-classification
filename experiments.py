"""Эксперимент: влияние числа участников ансамбля на качество."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ensemble import run

ROOT = Path(__file__).parent
TREE_METRICS = ROOT.parent / "classication-by-desicion-tree" / "results" / "metrics.json"


def baseline() -> dict:
    rows = json.loads(TREE_METRICS.read_text(encoding="utf-8"))
    return next(row for row in rows if row["train_fraction"] == .8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперименты с ансамблевой классификацией")
    parser.add_argument("--data", required=True)
    parser.add_argument("--technique", choices=["bagging", "random_forest", "boosting"], default="random_forest")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-samples", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    output = Path(args.results_dir); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in range(50, 101, 10):
        result = run(args.data, args.technique, count, .8, args.max_depth, args.max_samples, args.learning_rate, args.random_state)
        rows.append(result)
        print(f"n={count}: " + ", ".join(f"{key}={result[key]:.4f}" for key in ("accuracy", "precision", "recall", "f1")))
    tree = baseline()
    (output / "metrics.json").write_text(json.dumps({"ensemble": rows, "decision_tree_baseline": tree}, ensure_ascii=False, indent=2), encoding="utf-8")

    xs = [row["n_estimators"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F-мера"}
    for metric, label in names.items():
        line, = ax.plot(xs, [row[metric] for row in rows], marker="o", linewidth=2, label=f"Случайный лес — {label}" if args.technique == "random_forest" else f"{args.technique} — {label}")
        ax.axhline(tree[metric], linestyle="--", alpha=.75, color=line.get_color(), label=f"Дерево решений — {label}")
    ax.set_xlabel("Количество участников ансамбля")
    ax.set_ylabel("Значение показателя")
    ax.set_title("Качество ансамбля и базового дерева решений")
    ax.set_ylim(0, 1); ax.set_xticks(xs); ax.grid(True, linestyle="--", alpha=.45); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(output / "quality_vs_ensemble_size.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
