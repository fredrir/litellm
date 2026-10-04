import click
import pytest

from litellm_manager.core import Model
from litellm_manager.metadata import token_limits
from litellm_manager.settings import Settings
from litellm_manager.url_setup import unit_text


def test_dotenv_controls_url_and_user_key_from_any_directory(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "LITELLM_URL=http://models.localhost:8080\nLITELLM_API_KEY=my-chosen-key\nLITELLM_PORT=4020\n"
    )
    monkeypatch.setenv("LITELLM_ENV_FILE", str(env))
    monkeypatch.chdir(tmp_path.parent)
    settings = Settings()
    assert settings.api_url == "http://models.localhost:8080/v1"
    assert settings.port == 4020
    assert settings.require_key() == "my-chosen-key"


def test_missing_key_fails_with_editable_location(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_ENV_FILE", str(tmp_path / ".env"))
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(click.ClickException, match="Set LITELLM_API_KEY"):
        Settings().require_key()


def test_auto_output_budget_and_live_context_follow_backend(monkeypatch):
    model = Model("model", "local", "local", "/model", None, "1B", 0, -1, 131072, False, 8100)
    assert token_limits(model)["max_output_tokens"] == 131071
    monkeypatch.setattr("litellm_manager.metadata.live_context", lambda port: 8192)
    live = token_limits(model, True)
    assert live["max_input_tokens"] == 8191
    assert live["max_output_tokens"] == 8191
    assert live["fitted"] is True


def test_forwarder_binds_both_loopback_families_and_configured_ports():
    socket, service = unit_text(80, 4020)
    assert "ListenStream=127.0.0.1:80" in socket
    assert "ListenStream=[::1]:80" in socket
    assert "127.0.0.1:4020" in service
    assert "0.0.0.0" not in socket
