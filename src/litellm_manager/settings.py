import os
from pathlib import Path
from urllib.parse import urlsplit

import click
from dotenv import dotenv_values

PROJECT = Path(__file__).resolve().parents[2]


class Settings:
    def __init__(self):
        self.env_file = (
            Path(os.environ.get("LITELLM_ENV_FILE", PROJECT / ".env")).expanduser().resolve()
        )
        self.values = {**dotenv_values(self.env_file), **os.environ}
        self.url = (self.values.get("LITELLM_URL") or "http://litellm.localhost").rstrip("/")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or not (parsed.hostname == "localhost" or parsed.hostname.endswith(".localhost"))
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise click.ClickException(
                "LITELLM_URL must be an http://*.localhost URL with an optional port."
            )
        try:
            self.public_port = parsed.port or 80
            self.port = int(self.values.get("LITELLM_PORT", "4010"))
        except ValueError as exc:
            raise click.ClickException("Invalid port in .env.") from exc
        if not 1024 <= self.port <= 65535:
            raise click.ClickException("LITELLM_PORT must be between 1024 and 65535.")
        self.key = self.values.get("LITELLM_API_KEY") or ""

    def require_key(self):
        if not self.key.strip() or "replace-with" in self.key or self.key == "sk-1234":
            raise click.ClickException(f"Set LITELLM_API_KEY to your own key in {self.env_file}.")
        return self.key

    @property
    def api_url(self):
        return self.url + "/v1"
