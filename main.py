"""
main.py
Entry point for the alignment benchmark construct validity experiment.
Uses Together AI API (Llama 3.3 70B) for inference.

Usage:
  python main.py --pilot          # 50 items per source (~$0.50, ~5 min)
  python main.py --full           # 200 items per source (~$2.00, ~15 min)
  python main.py --analyze-only   # skip inference, analyze existing data

Requirements:
  pip install -r requirements.txt
  TOGETHER_API_KEY in .env file
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data.loader import (
    load_raw_items,
    generate_condition_items,
    validate_condition_balance,
    validate_question_consistency,
)
from inference.model_runner import run_experiment, load_responses
from coding.rater import code_all, compute_cohen_kappa
from analysis.stats import (
    compute_sycophancy_rate,
    compute_effect_size,
    compute_item_variance,
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
                        help="Run on 50 items per source (~$0.50)")
    parser.add_argument("--full", action="store_true",
                        help="Run on 200 items per source (~$2.00)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference; analyze existing data/responses.jsonl only")
    parser.add_argument("--extended-conditions", action="store_true",
                        help="Include 5th condition (researcher vocab without eval framing)")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="Together AI model string (default: Llama 3.3 70B)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ALIGNMENT BENCHMARK CONSTRUCT VALIDITY EXPERIMENT")
    print("="*60 + "\n")

    # ── Step 1: Load and prepare data ─────────────────────────────────────────
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

        # ── Step 2: Run inference ──────────────────────────────────────────────
        print(f"\nStep 4: Running experiment via Together AI (model: {args.model})...")
        print("(Progress is saved — safe to interrupt and resume)\n")
        responses = run_experiment(
            items=items,
            model_name=args.model,
            output_path="data/responses.jsonl",
        )
    else:
        print("Analyze-only mode: loading existing responses...")
        if not os.path.exists("data/responses.jsonl"):
            print("ERROR: data/responses.jsonl not found. Run without --analyze-only first.")
            sys.exit(1)
        responses = load_responses("data/responses.jsonl")

    # ── Step 3: Code responses ─────────────────────────────────────────────────
    print("\nStep 5: Coding responses...")
    coded = code_all(responses)
    save_coded(coded)

    unclear = sum(1 for c in coded if c["sycophancy_code"] == -1)
    errors = sum(1 for c in coded if c["has_error"])
    print(f"  Total responses:       {len(coded)}")
    print(f"  Unclear/unparseable:   {unclear} ({unclear/len(coded)*100:.1f}%)")
    print(f"  Errors:                {errors}")

    if unclear / max(len(coded), 1) > 0.1:
        print("\n  ⚠ WARNING: >10% unclear responses. The model may not be following "
              "the (A)/(B) instruction. Check data/responses.jsonl for raw outputs.")

    # ── Step 4: Analysis ───────────────────────────────────────────────────────
    print("\nStep 6: Running analysis...")
    df = pd.DataFrame(coded)
    df = df[df["sycophancy_code"] != -1]  # exclude unclear

    if len(df) == 0:
        print("ERROR: No valid coded responses to analyse. "
              "Check that the model is producing (A)/(B) answers.")
        sys.exit(1)

    # Manipulation check
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
        print(f"\nCore effect size (Condition A vs B): Cohen's d = {effect['cohens_d']:.3f}")
        if abs(effect["cohens_d"]) > 0.3:
            print("  ⚠ Exceeds d=0.3 threshold — potential construct validity concern")
        else:
            print("  ✓ Below d=0.3 threshold — scores appear robust to evaluation cues")

    # Repeated-measures ANOVA with post-hoc tests
    print("\nRepeated-measures ANOVA:")
    try:
        aov_results = run_repeated_measures_anova(df)
        for row in aov_results:
            print(f"  {row}")
        print("\nPost-hoc tests (Bonferroni corrected):")
        posthoc = run_posthoc_tests(df)
        print(posthoc[["A", "B", "T", "dof", "p-unc", "p-corr", "hedges"]].to_string(index=False))
    except Exception as e:
        print(f"  ANOVA skipped: {e}")

    # Full summary
    print_results_summary(df)

    # ── Step 5: Plots ──────────────────────────────────────────────────────────
    print("\nStep 7: Generating plots...")
    try:
        generate_all_plots(df, rates)
        print("Plots saved to plots/")
    except Exception as e:
        print(f"  Plotting failed: {e}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("  Results: data/coded_responses.jsonl")
    print("  Plots:   plots/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()