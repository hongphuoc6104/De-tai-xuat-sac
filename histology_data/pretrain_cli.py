"""CLI for metadata review, nested case-by-lens bundles and readiness checks."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .governance import (
    GovernanceError,
    build_governance,
    create_case_review_draft,
    write_case_review_draft,
)
from .readiness import build_pretrain_bundle, verify_pretrain_bundle
from .splits import SplitError


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


@contextmanager
def _writer_lock(output_root: Path) -> Iterator[None]:
    """Enforce one writer for governance and bundle outputs; stale locks fail closed."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ".pretrain.lock"
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                path.unlink()
        except FileNotFoundError:
            pass


def _print_result(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m histology_data.pretrain_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    draft = commands.add_parser("draft", help="create a metadata/source-scoped human review template")
    draft.add_argument("--metadata", type=Path, required=True)
    draft.add_argument("--source-root", type=Path, required=True)
    draft.add_argument("--output-root", type=Path, required=True)
    draft.add_argument("--source-kind", choices=("zip", "directory"), default="zip")
    draft.add_argument("--expected-sources", type=_positive_int, default=None)
    draft.add_argument("--expected-patches", type=_positive_int, default=148991)

    build = commands.add_parser("build", help="build reviewed governance and a full feature bundle")
    build.add_argument("--metadata", type=Path, required=True)
    build.add_argument("--case-review", type=Path, required=True)
    build.add_argument("--feature-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--cohort", choices=("common", "all"), default="common")
    build.add_argument("--duplicate-dispositions", type=Path)
    build.add_argument("--expected-sources", type=_positive_int, default=25)
    build.add_argument("--expected-vectors", type=_positive_int, default=148991)
    build.add_argument("--seed", type=int, default=42)

    verify = commands.add_parser("verify", help="verify a governed bundle and its source feature release")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--feature-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "draft":
            expected_sources = args.expected_sources
            if expected_sources is None:
                expected_sources = 25 if args.source_kind == "zip" else 1
            draft = create_case_review_draft(
                args.metadata,
                args.source_root,
                source_kind=args.source_kind,
                expected_sources=expected_sources,
                expected_patches=args.expected_patches,
            )
            with _writer_lock(args.output_root):
                directory = write_case_review_draft(draft, args.output_root)
            summary = {key: value for key, value in draft.items() if key not in {"review_rows", "source_plan"}}
            _print_result({"status": "draft_created", **summary, "path": str(directory),
                           "case_review_path": str(directory / "case_review.csv")})
            return 0
        if args.command == "build":
            with _writer_lock(args.output_root):
                governance = build_governance(args.metadata, args.case_review, args.output_root)
                result = build_pretrain_bundle(
                    Path(governance["path"]),
                    args.feature_root,
                    args.output_root,
                    cohort_mode=args.cohort,
                    duplicate_dispositions=args.duplicate_dispositions,
                    expected_sources=args.expected_sources,
                    expected_vectors=args.expected_vectors,
                    seed=args.seed,
                )
            _print_result({"governance": governance, "bundle": result})
            return 0 if result.get("training_ready") else 2
        result = verify_pretrain_bundle(args.bundle, args.feature_root)
        _print_result(result)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, GovernanceError, SplitError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
