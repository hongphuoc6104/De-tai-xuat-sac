"""Tests for Medical AI Domain Guardrails defined in AGENTS.md.

Invariants verified:
1. Zero Patient Data Leakage: assert_disjoint strictly catches overlapping patient_ids.
2. Objective Lens QC: strict_lens permits only 4, 10, 20, 40.
3. ISUP Grade QC: strict_grade permits only 0-5.
4. Stratified Patient Split: split_patients ensures zero data leakage.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add Script_Colab to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "Script_Colab"))

from data_integrity import assert_disjoint, split_patients, strict_grade, strict_lens  # noqa: E402


class TestMedicalGuardrails:
    """Test suite ensuring zero patient leakage and strict data integrity."""

    def test_strict_grade_valid(self):
        """Verify strict ISUP grade parsing for valid 0-5 grades."""
        for grade in range(6):
            assert strict_grade(str(grade)) == grade
            assert strict_grade(f"ISUP {grade}") == grade
            assert strict_grade(f"Grade: {grade}") == grade

    def test_strict_grade_invalid(self):
        """Verify non-ISUP grades (e.g. Gleason sum > 5, invalid strings) are rejected."""
        with pytest.raises(ValueError) as exc1:
            strict_grade("7")  # Gleason 3+4=7 is not ISUP
        assert "Invalid/missing ISUP grade" in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            strict_grade("Gleason 8")
        assert "Invalid/missing ISUP grade" in str(exc2.value)

        with pytest.raises(ValueError) as exc3:
            strict_grade("")
        assert "Invalid/missing ISUP grade" in str(exc3.value)

    def test_strict_lens_valid(self):
        """Verify objective lens parsing for 4X, 10X, 20X, 40X."""
        assert strict_lens("4") == 4
        assert strict_lens("10X") == 10
        assert strict_lens("20x") == 20
        assert strict_lens("40") == 40

    def test_strict_lens_invalid(self):
        """Verify non-standard magnifications are rejected."""
        with pytest.raises(ValueError) as exc1:
            strict_lens("100")
        assert "Invalid/missing objective lens" in str(exc1.value)

        with pytest.raises(ValueError) as exc2:
            strict_lens("60x")
        assert "Invalid/missing objective lens" in str(exc2.value)

        with pytest.raises(ValueError) as exc3:
            strict_lens("unknown")
        assert "Invalid/missing objective lens" in str(exc3.value)

    def test_zero_patient_leakage_assertion_passes_on_disjoint(self):
        """Verify assert_disjoint succeeds when patients are completely distinct."""
        split_a = pd.DataFrame({
            "slide_id": ["s1", "s2"],
            "patient_id": ["p1", "p2"],
            "label": [0, 1]
        })
        split_b = pd.DataFrame({
            "slide_id": ["s3", "s4"],
            "patient_id": ["p3", "p4"],
            "label": [0, 1]
        })
        # Should execute cleanly without error
        assert_disjoint(split_a, split_b)
        assert len(split_a) == 2
        assert len(split_b) == 2

    def test_zero_patient_leakage_assertion_fails_on_overlap(self):
        """Verify assert_disjoint raises ValueError immediately if any patient overlaps."""
        split_train = pd.DataFrame({
            "slide_id": ["s1", "s2"],
            "patient_id": ["p_shared", "p1"],
            "label": [0, 1]
        })
        split_val = pd.DataFrame({
            "slide_id": ["s3", "s4"],
            "patient_id": ["p_shared", "p2"],  # LEAKAGE: p_shared is in both!
            "label": [0, 1]
        })
        with pytest.raises(ValueError, match="Data leakage: overlapping patient_id") as exc:
            assert_disjoint(split_train, split_val)
        assert "overlapping patient_id" in str(exc.value)

    def test_patient_split_produces_disjoint_sets(self):
        """Verify split_patients guarantees disjoint splits with no patient leakage."""
        df = pd.DataFrame({
            "slide_id": [f"s{i}" for i in range(10)],
            "patient_id": [f"p{i}" for i in range(10)],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        train, val = split_patients(df, ratio=0.2, seed=42)
        assert len(train) > 0
        assert len(val) > 0
        # Assert no overlap
        train_patients = set(train["patient_id"])
        val_patients = set(val["patient_id"])
        assert train_patients.isdisjoint(val_patients)
