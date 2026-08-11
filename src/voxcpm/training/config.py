from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Any

import argbind
import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment_variables(value: Any) -> Any:
    """Recursively expand environment placeholders and reject missing values."""

    if isinstance(value, dict):
        return {key: _expand_environment_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted({name for name in _ENV_VAR_PATTERN.findall(value) if name not in os.environ})
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing environment variable(s) required by training config: {names}")
    return _ENV_VAR_PATTERN.sub(lambda match: os.environ[match.group(1)], value)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary suitable for argbind.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a top-level mapping.")
    return _expand_environment_variables(data)


def parse_args_with_config(config_path: str | Path | None = None):
    """
    Helper to unify CLI arguments and YAML configuration.

    Usage mirrors minicpm-audio:
        args = parse_args_with_config("conf/voxcpm/finetune.yml")
        with argbind.scope(args):
            ...
    """
    cli_args = argbind.parse_args()
    if config_path is None:
        return cli_args

    yaml_args = load_yaml_config(config_path)
    with argbind.scope(cli_args):
        yaml_args = argbind.parse_args(yaml_args=yaml_args, argv=[])
    cli_args.update(yaml_args)
    return cli_args
