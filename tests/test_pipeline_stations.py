"""Tests for the 6-station development conveyor pipeline.

Verifies:
1. Station 1: Adversarial Critique & Lavish HTML Dashboard generation.
2. Station 2: AST Assertion Integrity & Proof logging.
3. Station 3: Documentation & PLAYBOOK.md Schema Validation.
4. Station 4: Linter & Clean Code (ruff, compileall).
5. Station 5: PR preparation and treehouse worktree status.
6. Station 6: CI monitoring readiness.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from conveyor import (  # noqa: E402
    run_station_1_critique,
    run_station_2_tests,
    run_station_3_docs,
    run_station_4_lint,
    run_station_5_pr,
    run_station_6_ci,
    validate_playbook_schema,
    verify_ast_assertion_integrity,
)


class TestPipelineStations:
    """Verifies all 6 stations of the development conveyor pipeline."""

    def test_station_1_critique_generates_lavish_html(self):
        """Verify Station 1 runs critique checks and writes .lavish/pipeline_review.html."""
        assert run_station_1_critique(launch_lavish=False) is True
        lavish_html = ROOT_DIR / ".lavish" / "pipeline_review.html"
        assert lavish_html.exists()
        content = lavish_html.read_text(encoding="utf-8")
        assert "No-Mistake Pipeline" in content
        assert "Mermaid" in content or "mermaid" in content

    def test_station_2_ast_integrity_verifier(self):
        """Verify AST integrity checker detects valid test suites and flags empty tests."""
        valid, count, empty = verify_ast_assertion_integrity()
        assert valid is True
        assert count > 0
        assert len(empty) == 0

    def test_station_2_execution_generates_proof_log(self):
        """Verify Station 2 executes tests and produces immutable proof log."""
        assert run_station_2_tests(target_tests="tests/test_medical_guardrails.py") is True
        proofs_dir = ROOT_DIR / "Results" / "proofs"
        assert proofs_dir.exists()
        proof_files = list(proofs_dir.glob("test_evidence_*.log"))
        assert len(proof_files) > 0
        latest = sorted(proof_files)[-1]
        content = latest.read_text(encoding="utf-8")
        assert "VERIFIABLE TEST EXECUTION PROOF LOG" in content
        assert "AST Integrity   : PASSED" in content

    def test_station_3_playbook_schema_validation(self):
        """Verify Station 3 validates PLAYBOOK.md format."""
        assert run_station_3_docs() is True
        is_valid, issues = validate_playbook_schema()
        assert is_valid is True
        assert len(issues) == 0

    def test_station_4_lint_and_compile(self):
        """Verify Station 4 runs compileall and ruff cleanly."""
        assert run_station_4_lint() is True

    def test_station_5_pr_preparation(self):
        """Verify Station 5 formats PR title, body, and checks worktree."""
        assert run_station_5_pr(create_pr=False) is True

    def test_station_6_ci_readiness(self):
        """Verify Station 6 CI monitoring interface."""
        assert run_station_6_ci(watch=False) is True
