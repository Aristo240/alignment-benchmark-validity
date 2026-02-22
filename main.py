"""
main.py
Entry point for the alignment benchmark construct validity experiment.

Usage:
  python main.py --pilot          # Run on 50 items per source (cheap test)
  python main.py --full           # Run on 200 items per source
  python main.py --analyze-only   # Skip API calls, analyze existing data

Requirements:
  pip install anthropic pandas scipy pingouin matplotlib seaborn
  export ANTHROPIC_API_KEY=your_key_here

Estimated cost (pilot, 50 items × 3 sources × 4 conditions = 600 calls):
  ~$0.50–$1.00 using claude-haiku-4-5-20251001
"""

import argparse
import json
import os
import sys

# Make sure submodules are importable
sys.path.insert(0, os.path.dirname(__file__))

from data.loader import (
    load_raw_items,
    generate_condition_items,
    validate_condition_balance,
    validate_question_consistency,
)
from inference.model_runner import run_experiment, load_responses
from coding.rater import code_all, compute_cohen_kappa, compute_item_variance
from analysis.stats import (
    compute_sycophancy_rate,
    compute_effect_size,
    run_repeated_measures_anova,
    run_posthoc_tests,
    print_results_summary,
)
from analysis.plots import generate_all_plots

import pandas as pd


def save_coded(coded: list, path: str = "data/coded_responses.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in coded:
            f.write(json.dumps(item) + "\n")
    print(f"Coded responses saved to {path}")


def load_coded(path: str = "data/coded_responses.jsonl") -> list:
    coded = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                coded.append(json.loads(line))
    return coded


def main():
    parser = argparse.ArgumentParser(
        description="Alignment Benchmark Construct Validity Experiment"
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Run on 50 items per source (fast, cheap)")
    parser.add_argument("--full", action="store_true",
                        help="Run on 200 items per source")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip API calls; analyze data/responses.jsonl only")
    parser.add_argument("--extended-conditions", action="store_true",
                        help="Include 5th condition (researcher vocab without eval framing)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model string")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ALIGNMENT BENCHMARK CONSTRUCT VALIDITY EXPERIMENT")
    print("="*60 + "\n")

    # ── Step 1: Load and prepare data ────────────────────────────────────────
    if not args.analyze_only:
        max_per_source = 50 if args.pilot else 200
        print(f"Step 1: Loading dataset ({max_per_source} items per source)...")
        raw_items = load_raw_items(max_per_source=max_per_source)

        print("\nStep 2: Generating condition copies...")
        items = generate_condition_items(
            raw_items,
            use_extended_conditions=args.extended_conditions
        )

        print("\nStep 3: Validating data integrity...")
        validate_condition_balance(items)
        validate_question_consistency(items)
        print("✓ Data validation passed")

        # ── Step 2: Run experiment ────────────────────────────────────────────
        print(f"\nStep 4: Running experiment via Anthropic API (model: {args.model})...")
        print("(This may take a while — progress is saved and resumable)\n")
        responses = run_experiment(
            items=items,
            model=args.model,
            output_path="data/responses.jsonl",
        )
    else:
        print("Analyze-only mode: loading existing responses...")
        if not os.path.exists("data/responses.jsonl"):
            print("ERROR: data/responses.jsonl not found. Run without --analyze-only first.")
            sys.exit(1)
        responses = load_responses("data/responses.jsonl")

    # ── Step 3: Code responses ────────────────────────────────────────────────
    print("\nStep 5: Coding responses...")
    coded = code_all(responses)
    save_coded(coded)

    # Summary of coding quality
    unclear = sum(1 for c in coded if c["sycophancy_code"] == -1)
    errors = sum(1 for c in coded if c["has_error"])
    print(f"  Total responses: {len(coded)}")
    print(f"  Unclear/unparseable: {unclear} ({unclear/len(coded)*100:.1f}%)")
    print(f"  API errors: {errors}")

    # ── Step 4: Analysis ──────────────────────────────────────────────────────
    print("\nStep 6: Running analysis...")
    df = pd.DataFrame(coded)
    df = df[df["sycophancy_code"] != -1]  # exclude unclear

    # Item-level variance (manipulation check)
    print("\nItem-level variance (manipulation check):")
    compute_item_variance(df)

    # Sycophancy rates by condition
    rates = compute_sycophancy_rate(df)
    print("\nSycophancy rates by condition:")
    print(rates.to_string(index=False))

    # Core effect: Condition A vs B
    cond_a = df[df["condition"] == "A_evaluation_explicit"]["sycophancy_code"]
    cond_b = df[df["condition"] == "B_deployment_explicit"]["sycophancy_code"]
    if len(cond_a) > 0 and len(cond_b) > 0:
        effect = compute_effect_size(cond_a, cond_b)
        print(f"\nCore effect size (Condition A vs B): Cohen's d = {effect['cohens_d']}")

    # Repeated-measures ANOVA
    print("\nRepeated-measures ANOVA:")
    try:
        aov_results = run_repeated_measures_anova(df)
        for row in aov_results:
            print(f"  {row}")
        print("\nPost-hoc tests (Bonferroni corrected):")
        posthoc = run_posthoc_tests(df)
        print(posthoc[["A", "B", "T", "dof", "p-unc", "p-corr", "hedges"]].to_string(index=False))
    except Exception as e:
        print(f"  ANOVA failed (need more data or pingouin install): {e}")

    # Full results summary
    print_results_summary(df)

    # ── Step 5: Plots ─────────────────────────────────────────────────────────
    print("\nStep 7: Generating plots...")
    try:
        generate_all_plots(df, rates)
    except Exception as e:
        print(f"  Plotting failed: {e}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("Results: data/coded_responses.jsonl")
    print("Plots:   plots/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()