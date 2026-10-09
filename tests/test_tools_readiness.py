"""Tests verifying that all installed tools are operational in the environment.

Tools:
1. treehouse: worktree manager CLI
2. gh-axi: token-optimized GitHub CLI for agents
3. lavish-axi: interactive HTML review & visual report generator
4. ruff: fast Python linter
5. pytest: testing framework
"""
import shutil
import subprocess


class TestToolsReadiness:
    """Verifies that each CLI tool is installed, accessible in PATH, and executable."""

    def test_treehouse_installed_and_runs(self):
        """Verify treehouse CLI is executable and responds."""
        assert shutil.which("treehouse") is not None
        result = subprocess.run(["treehouse", "list"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "Git Worktrees" in result.stdout

    def test_gh_axi_installed_and_runs(self):
        """Verify gh-axi CLI is executable and outputs version or help."""
        assert shutil.which("gh-axi") is not None
        result = subprocess.run(["gh-axi", "--version"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_lavish_axi_installed_and_runs(self):
        """Verify lavish-axi CLI is executable and outputs version or help."""
        assert shutil.which("lavish-axi") is not None
        result = subprocess.run(["lavish-axi", "--help"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "lavish-axi" in result.stdout or "Lavish" in result.stdout

    def test_ruff_installed_and_runs(self):
        """Verify ruff linter is executable."""
        assert shutil.which("ruff") is not None
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "ruff" in result.stdout.lower()

    def test_pytest_installed_and_runs(self):
        """Verify pytest is executable."""
        assert shutil.which("pytest") is not None
        result = subprocess.run(["pytest", "--version"], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert "pytest" in result.stdout.lower()
