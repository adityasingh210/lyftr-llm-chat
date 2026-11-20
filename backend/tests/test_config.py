import pytest

from app.config import Settings, sqlite_path_from_url


@pytest.mark.parametrize(
    "url, path",
    [
        ("sqlite:////data/app.db", "/data/app.db"),
        ("sqlite:///./app.db", "./app.db"),
        ("sqlite:///:memory:", ":memory:"),
        ("/tmp/x.db", "/tmp/x.db"),
    ],
)
def test_sqlite_path_from_url(url, path):
    assert sqlite_path_from_url(url) == path


def test_unsupported_url():
    with pytest.raises(ValueError):
        sqlite_path_from_url("sqlite://host/db")


def test_key_pool_ids_follow_order():
    s = Settings.from_env({"LLM_API_KEYS": " a , b ,c", "LLM_MODEL": "gemini/x"})
    assert [k.key_id for k in s.keys] == ["key_1", "key_2", "key_3"]
    assert [k.api_key for k in s.keys] == ["a", "b", "c"]
    assert {k.model for k in s.keys} == {"gemini/x"}


def test_per_key_models():
    s = Settings.from_env({"LLM_API_KEYS": "a,b", "LLM_KEY_MODELS": "gemini/x,groq/y"})
    assert [k.model for k in s.keys] == ["gemini/x", "groq/y"]


def test_bool_and_origins():
    s = Settings.from_env({"ENABLE_FAILURE_SIMULATION": "TRUE", "ALLOWED_ORIGINS": "https://a.app, https://b.app"})
    assert s.enable_failure_simulation is True
    assert s.allowed_origins == ["https://a.app", "https://b.app"]
