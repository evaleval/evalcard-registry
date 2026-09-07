"""Publish the registry's canonical_* parquets to evaleval/entity-registry-data.

Replaces the inline Python in `.github/workflows/seed.yml` with a
dedicated, testable publish path:

  1. Run the seed CLI end-to-end against `seed/`, materialising every
     canonical_* table to `fixtures/`.
  2. Compute a content hash over the parquet bytes — used as the
     idempotency key. If the previous publish on HF carries the same
     hash, skip the upload (a no-op merge to main shouldn't churn the
     dataset history).
  3. Write a `manifest.json` sidecar carrying:
       - schema_version (registry.<MAJOR>.<MINOR>; major bump on
         removed/renamed columns, minor on additions).
       - content_hash.
       - seed_git_sha — the registry repo's HEAD SHA at publish time.
       - generated_at — ISO timestamp.
       - per-table row counts (sanity surface for consumers).
  4. Upload `fixtures/` + `manifest.json` to the HF Dataset repo as a
     single revision via HfApi.upload_folder.

Usage:
    uv run python scripts/publish_registry_data.py [--dry-run]

`--dry-run` writes the manifest and skips the push. ``--skip-seed`` is for CI
after it has already seeded and tested ``fixtures/``; it guarantees the bytes
validated by the workflow are the bytes offered to the Hub.

Backend consumers (eval_card_backend) should assert
`manifest.json.schema_version`'s major matches their expected major.
That guards against a registry breaking change that ships before the
producer is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

# Bump when removing/renaming columns in any canonical_* table.
# Bump minor when ADDING columns (backward-compatible). The producer
# asserts the major matches what it expects.
# 3.1: added canonical_orgs.logo_url (org brand-mark pointer).
# 3.2: added canonical_benchmarks.preferred_metric_id + the
#      benchmark_metric_folds table (merged benchmark view).
SCHEMA_VERSION = "registry.3.2"

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"
FIXTURES_DIR = REPO_ROOT / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
HF_DATASET_REPO = "evaleval/entity-registry-data"

# Cron-owned artifacts that must NEVER ride this flat publish — they are
# produced + published by their own dedicated cron (e.g. hub_stats_index by
# refresh-hub-stats-index). CI seeds into empty fixtures so they are absent
# here anyway, but a dev who ran the build script locally would otherwise leak
# a stray copy into the producer layout. (Mirrors hf_store._CRON_OWNED_TABLES.)
_CRON_OWNED_PARQUETS = {"hub_stats_index.parquet"}

# YAML overrides that ship alongside the parquets. The producer reads
# these when present in the registry data cache (slice_overrides drives
# slice→benchmark promotion; display_overrides feeds the display name
# polish layer). Storing them in the published dataset removes the
# producer's dependency on a sibling registry checkout.
SHIPPED_YAML = ("slice_overrides.yaml", "display_overrides.yaml")


def _git_sha() -> Optional[str]:
    """HEAD SHA of the registry repo at publish time. None when not in
    a git checkout (fresh CI runner with shallow clone? handled
    gracefully)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _origin_main_sha() -> Optional[str]:
    """Current ``origin/main`` SHA, read from the remote immediately before
    production mutation. ``None`` means the freshness check could not be made
    and must fail closed when requested by CI."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", "--exit-code", "origin", "refs/heads/main"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.split()[0] if out.stdout.split() else None
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None


def _assert_checkout_is_current_main() -> None:
    """Fail rather than let a delayed older workflow roll dataset HEAD back."""
    local_sha = _git_sha()
    remote_sha = _origin_main_sha()
    if not local_sha or not remote_sha:
        raise RuntimeError(
            "cannot verify checkout freshness against origin/main; refusing publish"
        )
    if local_sha != remote_sha:
        raise RuntimeError(
            f"stale checkout {local_sha} is not current origin/main {remote_sha}; "
            "refusing publish"
        )


def _content_hash(fixtures_dir: Path) -> str:
    """SHA-256 over the deterministically-sorted file bytes for every
    artifact we publish.

    Includes parquets and YAML overrides; excludes manifest.json (would
    create a circular dependency: hash depends on manifest, manifest
    contains hash). Hashes file PATH + file BYTES so a missing/added
    file changes the hash and the next publish doesn't get skipped.
    """
    h = hashlib.sha256()
    files = sorted(
        [p for p in fixtures_dir.glob("*.parquet") if p.name not in _CRON_OWNED_PARQUETS]
        + [p for p in fixtures_dir.glob("*.yaml") if p.name != "manifest.json"]
    )
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _row_counts(fixtures_dir: Path) -> dict[str, int]:
    """Per-table row counts for the manifest sanity surface. Uses
    pyarrow's metadata-only read so we don't materialise the whole
    parquet for a count."""
    import pyarrow.parquet as pq

    out: dict[str, int] = {}
    for p in sorted(fixtures_dir.glob("*.parquet")):
        if p.name in _CRON_OWNED_PARQUETS:
            continue
        try:
            md = pq.read_metadata(p)
            out[p.stem] = md.num_rows
        except Exception as exc:
            # Don't let a broken parquet block publish — log and skip.
            print(f"  [warn] couldn't read row count for {p.name}: {exc}", file=sys.stderr)
    return out


def _run_seed() -> int:
    """Invoke the seed CLI in LOCAL_MODE so it writes to fixtures/
    without pushing. This script handles the upload to HF itself
    (see upload step in main)."""
    env = dict(os.environ)
    env["LOCAL_MODE"] = "true"
    proc = subprocess.run(
        ["uv", "run", "eval-card-registry", "seed", "--local"],
        cwd=REPO_ROOT,
        env=env,
    )
    return proc.returncode


_T = TypeVar("_T")


def _is_retryable_http(exc: Exception) -> bool:
    """True for transient HF errors worth a longer wait: rate limits (429)
    and server errors (5xx). Auth/permission (401/403) and missing-repo (404)
    won't heal on retry, so we let those propagate immediately."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code == 429 or (isinstance(code, int) and 500 <= code < 600)


def _with_backoff(
    fn: Callable[[], _T],
    *,
    what: str,
    attempts: int = 5,
    base_seconds: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run fn(), retrying transient HF rate-limit / server errors with a long
    exponential backoff (15s, 30s, 60s, …). huggingface_hub already retries
    internally, but its backoff caps at 8s and gives up after a few tries —
    too shallow for a sustained 429 burst (which is what fails the publish
    when several push runs hit HF at once). This gives HF time to recover."""
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except HfHubHTTPError as exc:
            if attempt == attempts or not _is_retryable_http(exc):
                raise
            wait = base_seconds * (2 ** (attempt - 1))
            print(
                f"  [retry] {what}: {type(exc).__name__} "
                f"(attempt {attempt}/{attempts}); waiting {wait:.0f}s…",
                file=sys.stderr,
            )
            sleep(wait)
    # Unreachable: the final attempt either returns or raises above.
    raise RuntimeError(f"{what}: retry loop exited without a result")


def _read_remote_manifest() -> Optional[dict]:
    """Pull the existing manifest.json from HF (if any) so we can
    compare content_hash and skip no-op publishes. Returns None when
    the dataset hasn't been published before, the manifest doesn't
    exist yet (legacy revisions), or any HF call fails."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError:
        return None

    try:
        local = _with_backoff(
            lambda: hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename="manifest.json",
                repo_type="dataset",
            ),
            what="read remote manifest",
        )
        with open(local) as f:
            return json.load(f)
    except (
        RepositoryNotFoundError,
        EntryNotFoundError,
        HfHubHTTPError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        # Only reached for a genuine not-found / persistent failure: a
        # transient 429 or 5xx is retried by _with_backoff first. Failing
        # open here means we publish rather than skip — safe, just not a no-op.
        print(f"  [info] no remote manifest available ({type(exc).__name__}); will publish",
              file=sys.stderr)
        return None


def _push(fixtures_dir: Path, manifest: dict) -> None:
    """Upload fixtures/ + manifest.json to HF as a single dataset
    revision. Commit message includes the seed git SHA so consumers
    can pin to a specific registry source state."""
    from huggingface_hub import HfApi

    api = HfApi()
    sha = manifest.get("seed_git_sha") or "unknown"
    commit_msg = f"Publish registry data (seed @ {sha[:8] if sha != 'unknown' else 'unknown'})"
    api.upload_folder(
        folder_path=str(fixtures_dir),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=commit_msg,
        # Ship parquets + manifest + the YAML overrides that the
        # producer reads alongside (slice_overrides.yaml,
        # display_overrides.yaml). Excluding everything else keeps
        # the dataset small and the artifact list reviewable.
        allow_patterns=["*.parquet", "manifest.json", "*.yaml"],
        # Cron-owned artifacts are published by their own cron, never here.
        ignore_patterns=sorted(_CRON_OWNED_PARQUETS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run seed + write manifest locally; skip HF push. Used in PR CI.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Use already materialized fixtures (CI only; avoids re-seeding after tests).",
    )
    parser.add_argument(
        "--require-origin-main-head",
        action="store_true",
        help="Fail unless checkout HEAD is the current remote main (production CI).",
    )
    args = parser.parse_args()

    if args.skip_seed:
        print("[1/4] Using pre-seeded fixtures…", file=sys.stderr)
    else:
        print("[1/4] Running seed CLI…", file=sys.stderr)
        rc = _run_seed()
        if rc != 0:
            print(f"seed failed with exit code {rc}", file=sys.stderr)
            return rc

    if not FIXTURES_DIR.is_dir():
        print(f"fixtures/ missing after seed; expected {FIXTURES_DIR}", file=sys.stderr)
        return 1

    print("[2/4] Staging YAML overrides into fixtures/…", file=sys.stderr)
    import shutil as _shutil
    for name in SHIPPED_YAML:
        src = SEED_DIR / name
        if src.exists():
            _shutil.copy2(src, FIXTURES_DIR / name)
            print(f"  staged: {name}", file=sys.stderr)
        else:
            print(f"  [warn] {src} missing; consumers will fall back to defaults",
                  file=sys.stderr)

    print("[2/4] Computing content hash…", file=sys.stderr)
    content_hash = _content_hash(FIXTURES_DIR)
    print(f"  content_hash: {content_hash[:16]}…", file=sys.stderr)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "content_hash": content_hash,
        "seed_git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_counts": _row_counts(FIXTURES_DIR),
    }

    print("[3/4] Writing manifest.json…", file=sys.stderr)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  schema_version: {SCHEMA_VERSION}", file=sys.stderr)
    print(f"  row_counts: {sum(manifest['row_counts'].values())} rows across "
          f"{len(manifest['row_counts'])} tables", file=sys.stderr)

    if args.dry_run:
        print("[4/4] --dry-run: skipping HF upload.", file=sys.stderr)
        return 0

    if args.require_origin_main_head:
        print("[4/4] Verifying checkout is current origin/main…", file=sys.stderr)
        try:
            _assert_checkout_is_current_main()
        except RuntimeError as exc:
            print(f"  [error] {exc}", file=sys.stderr)
            return 1

    print("[4/4] Checking remote manifest for idempotency…", file=sys.stderr)
    remote = _read_remote_manifest()
    if remote and remote.get("seed_git_sha") == manifest.get("seed_git_sha"):
        print("  remote seed_git_sha matches local; skipping push (no-op).",
              file=sys.stderr)
        return 0
    if remote and remote.get("content_hash") == content_hash:
        print("  remote content_hash matches local; skipping push (no-op).",
              file=sys.stderr)
        return 0

    print(f"  pushing to {HF_DATASET_REPO}…", file=sys.stderr)
    _with_backoff(lambda: _push(FIXTURES_DIR, manifest), what="upload to HF")
    print("  done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
