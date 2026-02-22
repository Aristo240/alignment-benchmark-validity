"""
tests/test_all.py
Unit tests for all modules.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from data.loader import (
    generate_condition_items,
    validate_condition_balance,
    validate_question_consistency,
    CONDITIONS,
)
from inference.model_runner import _extract_answer_token, _check_evaluation_references
from coding.rater import code_response, compute_cohen_kappa
from analysis.stats import compute_sycophancy_rate, compute_effect_size

import pandas as pd
import numpy as np


# ── data/loader.py tests ──────────────────────────────────────────────────────

class TestLoader:
    def _make_raw_items(self, n=10):
        return [
            {
                "question": f"Question {i}",
                "answer_matching_behavior": " (A)" if i % 2 == 0 else " (B)",
                "source": "test",
            }
            for i in range(n)
        ]

    def test_condition_count(self):
        raw = self._make_raw_items(5)
        items = generate_condition_items(raw)
        assert len(items) == len(raw) * len(CONDITIONS)

    def test_question_identical_across_conditions(self):
        raw = self._make_raw_items(5)
        items = generate_condition_items(raw)
        validate_question_consistency(items)  # should not raise

    def test_condition_balance(self):
        raw = self._make_raw_items(5)
        items = generate_condition_items(raw)
        validate_condition_balance(items)  # should not raise

    def test_no_data_loss(self):
        raw = self._make_raw_items(20)
        items = generate_condition_items(raw)
        item_ids = set(i.item_id for i in items)
        assert len(item_ids) == len(raw)

    def test_system_prompt_injected_correctly(self):
        raw = self._make_raw_items(2)
        items = generate_condition_items(raw)
        neutral = [i for i in items if i.condition == "C_neutral"]
        assert all(i.system_prompt is None for i in neutral)
        eval_items = [i for i in items if i.condition == "A_evaluation_explicit"]
        assert all(i.system_prompt is not None for i in eval_items)


# ── inference/model_runner.py tests ──────────────────────────────────────────

class TestModelRunner:
    def test_extract_a(self):
        assert _extract_answer_token("(A) Yes, I agree.") == " (A)"

    def test_extract_b(self):
        assert _extract_answer_token("The answer is (B)") == " (B)"

    def test_extract_unclear(self):
        assert _extract_answer_token("I'm not sure about this.") == "UNCLEAR"

    def test_extract_ambiguous(self):
        # Both A and B mentioned — should return UNCLEAR
        assert _extract_answer_token("(A) or (B)") == "UNCLEAR"

    def test_eval_reference_detected(self):
        assert _check_evaluation_references("This is a safety evaluation test.") is True

    def test_eval_reference_not_detected(self):
        assert _check_evaluation_references("I think the answer is A.") is False

    def test_eval_reference_case_insensitive(self):
        assert _check_evaluation_references("SAFETY EVALUATION in progress.") is True


# ── coding/rater.py tests ─────────────────────────────────────────────────────

class TestRater:
    def _make_response(self, token, answer_matching):
        from inference.model_runner import ModelResponse
        return ModelResponse(
            item_id="test_001",
            source="test",
            condition="C_neutral",
            question="Test question",
            answer_matching_behavior=answer_matching,
            system_prompt=None,
            raw_response=f"The answer is {token}",
            response_token=token,
            evaluation_references=False,
        )

    def test_correct_match_codes_1(self):
        r = self._make_response(" (A)", " (A)")
        assert code_response(r) == 1

    def test_mismatch_codes_0(self):
        r = self._make_response(" (B)", " (A)")
        assert code_response(r) == 0

    def test_unclear_codes_minus_1(self):
        r = self._make_response("UNCLEAR", " (A)")
        assert code_response(r) == -1

    def test_kappa_perfect_agreement(self):
        r1 = [1, 0, 1, 1, 0]
        r2 = [1, 0, 1, 1, 0]
        assert compute_cohen_kappa(r1, r2) == 1.0

    def test_kappa_no_agreement(self):
        # Alternating disagreement produces kappa < 1
        r1 = [1, 0, 1, 0, 1, 0]
        r2 = [0, 1, 0, 1, 0, 1]
        kappa = compute_cohen_kappa(r1, r2)
        assert kappa < 0  # systematic disagreement = negative kappa

    def test_kappa_excludes_unclear(self):
        r1 = [1, -1, 0, 1]
        r2 = [1, 0, 0, 1]
        # Should skip the pair where r1=-1
        kappa = compute_cohen_kappa(r1, r2)
        assert isinstance(kappa, float)

    def test_kappa_empty_after_exclusion(self):
        with pytest.raises(ValueError):
            compute_cohen_kappa([-1, -1], [-1, 0])


# ── analysis/stats.py tests ───────────────────────────────────────────────────

class TestStats:
    def _make_df(self):
        np.random.seed(42)
        n = 100
        data = []
        for cond, base_rate in [
            ("A_evaluation_explicit", 0.6),
            ("B_deployment_explicit", 0.4),
            ("C_neutral", 0.5),
            ("D_counter_signal", 0.45),
        ]:
            for i in range(n):
                data.append({
                    "item_id": f"item_{i:04d}",
                    "condition": cond,
                    "sycophancy_code": int(np.random.random() < base_rate),
                    "evaluation_references": int(np.random.random() < 0.05),
                    "source": "test",
                })
        return pd.DataFrame(data)

    def test_sycophancy_rate_columns(self):
        df = self._make_df()
        rates = compute_sycophancy_rate(df)
        assert "condition" in rates.columns
        assert "sycophancy_rate" in rates.columns
        assert "n" in rates.columns

    def test_sycophancy_rate_range(self):
        df = self._make_df()
        rates = compute_sycophancy_rate(df)
        assert (rates["sycophancy_rate"] >= 0).all()
        assert (rates["sycophancy_rate"] <= 1).all()

    def test_all_conditions_present(self):
        df = self._make_df()
        rates = compute_sycophancy_rate(df)
        assert len(rates) == 4

    def test_effect_size_direction(self):
        # Group A has higher rate than B — d should be positive
        df = self._make_df()
        a = df[df["condition"] == "A_evaluation_explicit"]["sycophancy_code"]
        b = df[df["condition"] == "B_deployment_explicit"]["sycophancy_code"]
        effect = compute_effect_size(a, b)
        assert "cohens_d" in effect
        # A was set at 0.6, B at 0.4, so d > 0
        assert effect["cohens_d"] > 0

    def test_effect_size_zero_for_identical(self):
        a = pd.Series([1, 0, 1, 0, 1])
        b = pd.Series([1, 0, 1, 0, 1])
        effect = compute_effect_size(a, b)
        assert effect["cohens_d"] == 0.0