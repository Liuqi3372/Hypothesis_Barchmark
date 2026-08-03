from pathlib import Path

from pmc_m.config import DEFAULT_NCBI_EMAIL, load_deepseek_api_key, load_ncbi_api_key


def test_default_email_is_configured():
    assert DEFAULT_NCBI_EMAIL == "lllmeqi77@gmail.com"


def test_key_file_is_trimmed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    key_file = tmp_path / "NCBI_API_KEY.txt"
    key_file.write_text("abc123\n", encoding="utf-8")
    assert load_ncbi_api_key(key_file) == "abc123"


def test_deepseek_key_file_is_loaded(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    key_file = tmp_path / "DEEPSEEK_API_KEY.txt"
    key_file.write_text("ds-test-key\n", encoding="utf-8")
    assert load_deepseek_api_key(key_file) == "ds-test-key"
