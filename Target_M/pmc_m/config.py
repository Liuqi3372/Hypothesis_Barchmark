from __future__ import annotations

import os
from pathlib import Path


DEFAULT_NCBI_EMAIL = "lllmeqi77@gmail.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NCBI_KEY_FILE = PROJECT_ROOT / "key" / "NCBI_API_KEY.txt"
DEFAULT_DEEPSEEK_KEY_FILE = PROJECT_ROOT / "key" / "DEEPSEEK_API_KEY.txt"


def get_ncbi_email() -> str:
    return os.getenv("NCBI_EMAIL", "").strip() or DEFAULT_NCBI_EMAIL


def _load_key(environment_name: str, path: Path) -> str:
    value = os.getenv(environment_name, "").strip()
    if value:
        return value
    if not path.is_file():
        return ""
    value = path.read_text(encoding="utf-8-sig").strip()
    if value and any(character.isspace() for character in value):
        raise ValueError(f"The API key file must contain exactly one key line: {path}")
    return value


def load_ncbi_api_key(path: Path = DEFAULT_NCBI_KEY_FILE) -> str:
    return _load_key("NCBI_API_KEY", path)


def load_deepseek_api_key(path: Path = DEFAULT_DEEPSEEK_KEY_FILE) -> str:
    return _load_key("DEEPSEEK_API_KEY", path)
