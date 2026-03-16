#!/usr/bin/env python3
"""Generate README charts from eval metrics.

Outputs:
- docs/charts/ — SVG charts (if matplotlib available)
- docs/RESULTS.md — Mermaid charts + tables (always works on GitHub)
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
CHARTS_DIR = PROJECT_ROOT / "docs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_pp_metrics() -> dict | None:
    path = RESULTS_DIR / "pride_prejudice_latest.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_eval_metrics() -> dict | None:
    path = RESULTS_DIR / "latest.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def generate_mermaid_results(pp_data: dict, eval_data: dict | None) -> str:
    """Generate Mermaid charts + markdown tables."""
    lines = ["# Woven Imprint — Evaluation Results\n"]

    # Eval suite results
    if eval_data:
        lines.append("## Benchmark Suite\n")
        lines.append(
            f"**{eval_data['total_passed']}/{eval_data['total_tests']} passed** "
            f"| Average score: {eval_data['avg_score']:.1%} "
            f"| Duration: {eval_data['duration_ms']:.0f}ms\n"
        )

        for suite in eval_data.get("suites", []):
            lines.append(f"### {suite['suite_name']}\n")
            lines.append("| Benchmark | Score | Status |")
            lines.append("|-----------|-------|--------|")
            for r in suite["results"]:
                status = "PASS" if r["passed"] else "FAIL"
                icon = "&#x2705;" if r["passed"] else "&#x274C;"
                lines.append(f"| {r['name']} | {r['score']:.2f} | {icon} {status} |")
            lines.append("")

    # Pride and Prejudice results
    lines.append("## Pride and Prejudice — Relationship Evolution\n")
    lines.append(
        "16 scenes from the public domain novel (Project Gutenberg). "
        "Characters and interactions extracted from the text.\n"
    )

    # Extract Elizabeth → Darcy arc
    elizabeth_darcy = []
    darcy_elizabeth = []
    bingley_jane = []
    jane_bingley = []

    for m in pp_data.get("metrics", []):
        pair = m.get("pair", [])
        if "Elizabeth Bennet" in pair and "Mr. Darcy" in pair:
            if pair[0] == "Elizabeth Bennet":
                if m.get("a_to_b"):
                    elizabeth_darcy.append((m["scene_number"], m["scene_title"], m["a_to_b"]))
                if m.get("b_to_a"):
                    darcy_elizabeth.append((m["scene_number"], m["scene_title"], m["b_to_a"]))
            else:
                if m.get("b_to_a"):
                    elizabeth_darcy.append((m["scene_number"], m["scene_title"], m["b_to_a"]))
                if m.get("a_to_b"):
                    darcy_elizabeth.append((m["scene_number"], m["scene_title"], m["a_to_b"]))
        elif "Mr. Bingley" in pair and "Jane Bennet" in pair:
            if pair[0] == "Mr. Bingley":
                if m.get("a_to_b"):
                    bingley_jane.append((m["scene_number"], m["scene_title"], m["a_to_b"]))
                if m.get("b_to_a"):
                    jane_bingley.append((m["scene_number"], m["scene_title"], m["b_to_a"]))
            else:
                if m.get("b_to_a"):
                    bingley_jane.append((m["scene_number"], m["scene_title"], m["b_to_a"]))
                if m.get("a_to_b"):
                    jane_bingley.append((m["scene_number"], m["scene_title"], m["a_to_b"]))

    # Elizabeth → Darcy table
    if elizabeth_darcy:
        lines.append("### Elizabeth Bennet → Mr. Darcy\n")
        lines.append("*From hostility to love — tracked across the novel's key scenes.*\n")
        lines.append("| Scene | Trust | Affection | Respect | Familiarity | Tension |")
        lines.append("|-------|-------|-----------|---------|-------------|---------|")
        for num, title, dims in elizabeth_darcy:
            lines.append(
                f"| {num}. {title} | {dims['trust']:.2f} | {dims['affection']:.2f} | "
                f"{dims['respect']:.2f} | {dims['familiarity']:.2f} | {dims['tension']:.2f} |"
            )
        lines.append("")

        # Mermaid XY chart for Elizabeth→Darcy
        lines.append("```mermaid")
        lines.append("xychart-beta")
        lines.append('    title "Elizabeth → Darcy: Relationship Arc"')
        scene_labels = [f'"{num}"' for num, _, _ in elizabeth_darcy]
        lines.append(f'    x-axis "Scene" [{", ".join(scene_labels)}]')
        lines.append('    y-axis "Score" -0.3 --> 0.3')
        trust_vals = [f"{d['trust']:.2f}" for _, _, d in elizabeth_darcy]
        affection_vals = [f"{d['affection']:.2f}" for _, _, d in elizabeth_darcy]
        lines.append(f"    line [{', '.join(trust_vals)}]")
        lines.append(f"    line [{', '.join(affection_vals)}]")
        lines.append("```\n")

    # Darcy → Elizabeth table
    if darcy_elizabeth:
        lines.append("### Mr. Darcy → Elizabeth Bennet\n")
        lines.append("*Respect and affection climb steadily while she still scorns him.*\n")
        lines.append("| Scene | Trust | Affection | Respect | Familiarity | Tension |")
        lines.append("|-------|-------|-----------|---------|-------------|---------|")
        for num, title, dims in darcy_elizabeth:
            lines.append(
                f"| {num}. {title} | {dims['trust']:.2f} | {dims['affection']:.2f} | "
                f"{dims['respect']:.2f} | {dims['familiarity']:.2f} | {dims['tension']:.2f} |"
            )
        lines.append("")

    # Bingley ↔ Jane
    if bingley_jane:
        lines.append("### Mr. Bingley → Jane Bennet\n")
        lines.append("*Pure warmth. Zero tension. As Austen intended.*\n")
        lines.append("| Scene | Trust | Affection | Respect | Familiarity | Tension |")
        lines.append("|-------|-------|-----------|---------|-------------|---------|")
        for num, title, dims in bingley_jane:
            lines.append(
                f"| {num}. {title} | {dims['trust']:.2f} | {dims['affection']:.2f} | "
                f"{dims['respect']:.2f} | {dims['familiarity']:.2f} | {dims['tension']:.2f} |"
            )
        lines.append("")

    # Key findings
    if elizabeth_darcy:
        first = elizabeth_darcy[0][2]
        worst = max(elizabeth_darcy, key=lambda x: abs(x[2].get("trust", 0)))
        last = elizabeth_darcy[-1][2]

        lines.append("### Key Findings\n")
        lines.append(
            f"- **Peak hostility** at scene {worst[0]} ({worst[1]}): "
            f"trust={worst[2]['trust']:.2f}, tension={worst[2]['tension']:.2f}"
        )
        lines.append(
            f"- **Resolution** at scene {elizabeth_darcy[-1][0]}: "
            f"trust={last['trust']:.2f}, affection={last['affection']:.2f}"
        )
        lines.append(
            f"- **Familiarity** climbed from {first['familiarity']:.2f} to "
            f"{last['familiarity']:.2f} — they know each other intimately by the end"
        )
        lines.append(
            "- All relationship changes are **LLM-assessed** from conversation "
            "content, not scripted"
        )
        lines.append("")

    return "\n".join(lines)


def generate_svg_charts(pp_data: dict) -> bool:
    """Generate SVG charts using matplotlib. Returns False if not available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping SVG generation")
        return False

    # Extract Elizabeth → Darcy arc
    scenes = []
    trust = []
    affection = []
    respect = []
    tension = []

    for m in pp_data.get("metrics", []):
        pair = m.get("pair", [])
        dims = None
        if pair == ["Elizabeth Bennet", "Mr. Darcy"] and m.get("a_to_b"):
            dims = m["a_to_b"]
        elif pair == ["Mr. Darcy", "Elizabeth Bennet"] and m.get("b_to_a"):
            dims = m["b_to_a"]

        if dims:
            scenes.append(m["scene_number"])
            trust.append(dims["trust"])
            affection.append(dims["affection"])
            respect.append(dims["respect"])
            tension.append(dims["tension"])

    if not scenes:
        return False

    # Key plot events for annotations (scene_number → label)
    annotations = {
        1: '"Tolerable"\ninsult',
        4: "Accomplished\nwomen debate",
        6: "Netherfield\nBall",
        9: "First proposal\n(refused)",
        10: "The Letter",
        11: "Visits\nPemberley",
        12: "Lydia\nelopes",
        13: "Darcy rescues\nBennets",
        16: "Second proposal\n(accepted)",
    }

    # Elizabeth → Darcy arc chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(scenes, trust, "o-", label="Trust", color="#2563eb", linewidth=2.5, markersize=6)
    ax.plot(
        scenes, affection, "s-", label="Affection", color="#dc2626", linewidth=2.5, markersize=6
    )
    ax.plot(
        scenes,
        respect,
        "^-",
        label="Respect",
        color="#16a34a",
        linewidth=2,
        markersize=5,
        alpha=0.8,
    )
    ax.plot(
        scenes,
        tension,
        "D-",
        label="Tension",
        color="#ea580c",
        linewidth=1.5,
        markersize=4,
        alpha=0.6,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    # Add novel timeline annotations
    for scene_num, label in annotations.items():
        if scene_num in scenes:
            idx = scenes.index(scene_num)
            # Place annotation above or below depending on values
            y_val = max(trust[idx], affection[idx], tension[idx])
            y_offset = 0.12 if y_val < 0.3 else -0.18
            ax.annotate(
                label,
                xy=(scene_num, y_val),
                xytext=(scene_num, y_val + y_offset),
                fontsize=7,
                ha="center",
                va="bottom" if y_offset > 0 else "top",
                color="#555555",
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5),
            )

    ax.set_xlabel("Scene", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Elizabeth Bennet → Mr. Darcy: Relationship Arc\n"
        "Pride and Prejudice — tracked by Woven Imprint",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_ylim(-0.5, 1.15)
    ax.set_xticks(scenes)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "elizabeth_darcy_arc.svg", format="svg", dpi=150)
    plt.close(fig)
    print("  Generated: docs/charts/elizabeth_darcy_arc.svg")

    return True


def main():
    print("Generating charts and results...\n")

    pp_data = load_pp_metrics()
    eval_data = load_eval_metrics()

    if not pp_data:
        print("No Pride & Prejudice metrics found. Run eval/pride_and_prejudice.py first.")
        return

    # Generate Mermaid + tables (always works)
    results_md = generate_mermaid_results(pp_data, eval_data)
    results_path = PROJECT_ROOT / "docs" / "RESULTS.md"
    results_path.write_text(results_md)
    print("  Generated: docs/RESULTS.md")

    # Try SVG charts
    generate_svg_charts(pp_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
