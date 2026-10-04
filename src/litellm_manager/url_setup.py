"""Install a loopback-only, socket-activated forwarder for the configured localhost URL."""

import os
import subprocess
from pathlib import Path

from .settings import Settings


def unit_text(public_port: int, backend_port: int) -> tuple[str, str]:
    socket = (
        "[Unit]\nDescription=LiteLLM localhost URL\n\n[Socket]\n"
        f"ListenStream=127.0.0.1:{public_port}\nListenStream=[::1]:{public_port}\n"
        "BindIPv6Only=ipv6-only\nNoDelay=true\n\n[Install]\nWantedBy=sockets.target\n"
    )
    service = (
        "[Unit]\nDescription=LiteLLM loopback forwarding\n\n[Service]\n"
        f"ExecStart=/usr/lib/systemd/systemd-socket-proxyd --exit-idle-time=30s 127.0.0.1:{backend_port}\n"
        "NoNewPrivileges=true\nPrivateTmp=true\nProtectSystem=strict\nProtectHome=true\n"
    )
    if os.geteuid() == 0:
        service += "DynamicUser=true\n"
    return socket, service


def main():
    settings = Settings()
    root = os.geteuid() == 0
    if settings.public_port < 1024 and not root:
        raise SystemExit(
            "Port 80 requires administrator privileges. Run: sudo ./scripts/setup-url.sh"
        )
    if settings.public_port == settings.port:
        raise SystemExit("Public forwarding port must differ from LITELLM_PORT.")
    directory = Path("/etc/systemd/system") if root else Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    socket, service = unit_text(settings.public_port, settings.port)
    for suffix, content in [("socket", socket), ("service", service)]:
        (directory / f"litellm-localhost.{suffix}").write_text(content)
    command = ["systemctl"] + ([] if root else ["--user"])
    subprocess.run([*command, "daemon-reload"], check=True)
    subprocess.run([*command, "stop", "litellm-localhost.service"], check=False)
    subprocess.run([*command, "enable", "litellm-localhost.socket"], check=True)
    subprocess.run([*command, "restart", "litellm-localhost.socket"], check=True)
    print(f"Forwarding {settings.url} to loopback port {settings.port} (IPv4 + IPv6).")


if __name__ == "__main__":
    main()
