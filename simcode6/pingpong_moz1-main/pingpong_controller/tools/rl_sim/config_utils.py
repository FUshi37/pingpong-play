from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def apply_yaml_defaults(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    *,
    section: str,
    default_config_path: Path,
) -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=default_config_path)
    pre_args, _ = pre_parser.parse_known_args(argv)

    cfg_path = Path(pre_args.config).resolve()
    cfg = _read_yaml_config(cfg_path)
    common_cfg = cfg.get("common", {})
    section_cfg = cfg.get(section, {})
    if common_cfg and not isinstance(common_cfg, dict):
        raise ValueError(f"'common' section must be a mapping: {cfg_path}")
    if section_cfg and not isinstance(section_cfg, dict):
        raise ValueError(f"'{section}' section must be a mapping: {cfg_path}")
    merged = _merge_dicts(common_cfg or {}, section_cfg or {})
    parser.set_defaults(**merged)
