"""Shared config, path resolution, and logging helpers for Phase 7a."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_utf8_stdio() -> None:
    """Configure stdout/stderr UTF-8 encoding on Windows."""
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.stderr.encoding != "utf-8":
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def get_repo_root() -> Path:
    """Return active repository/worktree root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load config.yaml using UTF-8 encoding."""
    root = get_repo_root()
    cfg_file = Path(config_path) if config_path else (root / "config.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve relative paths against base_dir (defaults to active repo root)."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return (base_dir or get_repo_root()) / p


def resolve_phase7a_paths(
    cfg: Dict[str, Any],
    merged_dir_override: str | Path | None = None,
) -> Dict[str, Path]:
    """Resolve all Phase 7a input/output paths from config.yaml."""
    repo_root = get_repo_root()
    paths_cfg = cfg.get("paths", {})
    project_root = Path(cfg.get("project_root", str(repo_root)))

    catalog_path = resolve_path(paths_cfg["catalog"], repo_root)
    checkpoint_dir = resolve_path(paths_cfg["checkpoint_dir"], repo_root)
    eval_dir = resolve_path(paths_cfg["eval_dir"], repo_root)
    reports_dir = resolve_path(paths_cfg["reports_dir"], repo_root)

    if merged_dir_override:
        merged_dir = Path(merged_dir_override)
    elif "merged_dataset" in paths_cfg:
        merged_raw = Path(paths_cfg["merged_dataset"])
        merged_dir = merged_raw if merged_raw.is_absolute() else (project_root / merged_raw)
    else:
        merged_dir = project_root / "_build" / "Hoc_sau_ver4" / "data" / "merged_dataset"

    eval_p7 = eval_dir / "phase7"
    reports_p7 = reports_dir / "phase7"
    shards_dir = eval_p7 / "shards"
    contact_sheets_dir = reports_p7 / "contact_sheets"
    candidates_out = catalog_path.parent / "candidates_v3.parquet"

    return {
        "repo_root": repo_root,
        "project_root": project_root,
        "catalog_path": catalog_path,
        "checkpoint_dir": checkpoint_dir,
        "eval_dir": eval_dir,
        "reports_dir": reports_dir,
        "merged_dir": merged_dir,
        "eval_p7": eval_p7,
        "reports_p7": reports_p7,
        "shards_dir": shards_dir,
        "contact_sheets_dir": contact_sheets_dir,
        "candidates_out": candidates_out,
        "build_script": repo_root / "scripts" / "build_merged_dataset.py",
    }
