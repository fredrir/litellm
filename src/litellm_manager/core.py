import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

import click
import yaml

from .catalog import preset_for, service_for
from .metadata import model_metadata, token_limits
from .settings import PROJECT, Settings


@dataclass
class Model:
    name: str
    repo: str
    revision: str
    path: str
    mmproj: str | None
    params: str
    context: int
    output: int
    trained_context: int | None
    tools: bool
    port: int
    cache: str = "f16"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    @property
    def unit(self) -> str:
        return "litellm-model-" + sha256(self.name.encode()).hexdigest()[:16] + ".service"


@dataclass
class Service:
    name: str
    project: str
    script: str
    route: str
    extra: str
    port: int
    env: list[str] = field(default_factory=list)
    kind: str = "service"

    @property
    def unit(self) -> str:
        return "litellm-service-" + sha256(self.name.encode()).hexdigest()[:16] + ".service"


Entry = Model | Service


class Store:
    def __init__(self):
        self.settings = Settings()
        home = Path.home()
        self.config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "litellm-manager"
        self.data = Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "litellm-manager"
        self.state = (
            Path(os.environ.get("XDG_STATE_HOME", home / ".local/state")) / "litellm-manager"
        )
        self.units = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "systemd/user"
        self.registry = self.config / "models.json"
        self.proxy_config = self.config / "proxy.yaml"
        self.proxy_port = self.settings.port

    def read(self) -> list[Entry]:
        if not self.registry.exists():
            return []
        try:
            return [
                Service(**m) if m.get("kind") == "service" else Model(**m)
                for m in json.loads(self.registry.read_text())
            ]
        except (ValueError, TypeError, OSError) as exc:
            raise click.ClickException(f"Cannot read {self.registry}: {exc}") from exc

    @contextmanager
    def transaction(self):
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / "lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def save(self, models: list[Entry]):
        atomic_write(self.registry, json.dumps([asdict(m) for m in models], indent=2) + "\n")


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def resolve(models: list[Entry], name: str) -> Entry:
    exact = [m for m in models if m.name.casefold() == name.casefold()]
    matches = exact or [m for m in models if m.name.split("/")[-1].casefold() == name.casefold()]
    if len(matches) != 1:
        reason = "Ambiguous model" if matches else "Unknown model"
        raise click.ClickException(f"{reason}: {name}. Use 'litellm list' and the full model name.")
    return matches[0]


def available_port(models: list[Entry], proxy_port: int) -> int:
    reserved = {m.port for m in models} | {proxy_port}
    for port in range(8100, 9000):
        if port not in reserved and port_free(port):
            return port
    raise click.ClickException("No free backend port in 8100–8999.")


def port_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def gguf_file(path: str) -> str:
    p = Path(path).expanduser().resolve()
    try:
        with p.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise click.ClickException(f"Not a GGUF model: {p}")
    except OSError as exc:
        raise click.ClickException(f"Cannot open {p}: {exc}") from exc
    return str(p)


def download_model(
    name: str,
    models: list[Model],
    proxy_port: int,
    *,
    repo: str | None = None,
    filename: str | None = None,
    mmproj: str | None = None,
    context: int | None = None,
    output: int | None = None,
    params: str | None = None,
    tools: bool | None = None,
) -> Model:
    # Keep Hugging Face imports off the list/completion/start path.
    from huggingface_hub import HfApi, hf_hub_download

    found = preset_for(name)
    preset = found[1] if found else None
    canonical = found[0] if found else name
    if any(m.name.casefold() == canonical.casefold() for m in models):
        raise click.ClickException(f"Already added: {canonical}")
    if context and output and output >= context:
        raise click.ClickException(
            "--output must be smaller than --context (prompt/image tokens need space)."
        )
    port = available_port(models, proxy_port)
    if Path(name).expanduser().is_file():
        path = gguf_file(name)
        projector = gguf_file(mmproj) if mmproj else None
        canonical = Path(path).stem
        if any(m.name.casefold() == canonical.casefold() for m in models):
            raise click.ClickException(f"Already added: {canonical}")
        source, revision = "local", "local"
    else:
        source = repo or (preset.repo if preset else name)
        token = Settings().values.get("HF_TOKEN")
        info = HfApi(token=token).model_info(
            source, revision=preset.revision if preset and not repo else None
        )
        revision = info.sha  # Pin every download, including custom models, to one snapshot.
        files = [s.rfilename for s in info.siblings]
        model_file = filename or (preset.filename if preset and not repo else None)
        projector_file = mmproj or (preset.mmproj if preset and not repo else None)
        if not model_file:
            candidates = [
                f
                for f in files
                if f.endswith(".gguf")
                and not any(s in f.lower() for s in ("mmproj", "mtp", "draft"))
            ]
            if len(candidates) != 1:
                raise click.ClickException(
                    "Select a GGUF with --file (and --mmproj for vision). "
                    f"Candidates: {', '.join(candidates) or 'none; use --repo with a GGUF repository'}"
                )
            model_file = candidates[0]
        if not projector_file:
            candidates = [f for f in files if f.endswith(".gguf") and "mmproj" in f.lower()]
            if len(candidates) == 1:
                projector_file = candidates[0]
            elif candidates:
                raise click.ClickException("Select the matching vision projector with --mmproj.")
        for file in [model_file, projector_file]:
            if file and file not in files:
                raise click.ClickException(f"File not found in {source}@{revision}: {file}")
        path = gguf_file(hf_hub_download(source, model_file, revision=revision, token=token))
        projector = (
            gguf_file(hf_hub_download(source, projector_file, revision=revision, token=token))
            if projector_file
            else None
        )
    metadata = model_metadata(path, projector)
    trained = metadata["trained_context"]
    if context and context > trained:
        raise click.ClickException(f"--context exceeds the model's trained context ({trained}).")
    if output and output >= (context or trained):
        raise click.ClickException("--output must be smaller than the model context.")
    return Model(
        canonical,
        source,
        revision,
        path,
        projector,
        params or metadata["params"],
        context or 0,
        output or -1,
        trained,
        tools if tools is not None else metadata["tools"],
        port,
        preset.cache if preset else "f16",
        preset.temperature if preset else 0.0,
        preset.top_k if preset else 0,
        preset.top_p if preset else 1.0,
    )


def uv_binary() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise click.ClickException("uv not found. Install uv: https://docs.astral.sh/uv/")
    return str(Path(uv).resolve())


def add_service(name: str, models: list[Entry], proxy_port: int) -> Service:
    canonical, preset = service_for(name)
    if any(m.name.casefold() == canonical.casefold() for m in models):
        raise click.ClickException(f"Already added: {canonical}")
    project = PROJECT / preset.project
    extra = "gpu" if shutil.which("nvidia-smi") else "cpu"
    result = subprocess.run(
        [uv_binary(), "sync", "--project", str(project), "--frozen", "--extra", extra],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise click.ClickException(f"uv sync failed:\n{result.stderr.strip()[-2000:]}")
    return Service(
        canonical,
        str(project),
        preset.script,
        preset.route,
        extra,
        available_port(models, proxy_port),
        list(preset.env),
    )


def pass_through(service: Service) -> dict:
    return {
        "path": service.route,
        "target": f"http://127.0.0.1:{service.port}",
        "include_subpath": True,
        "auth": True,
        "timeout": 300,
    }


class Services:
    proxy_unit = "litellm-manager-proxy.service"

    def __init__(self, store: Store):
        self.store = store
        self.env = os.environ.copy()
        runtime = self.env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        self.env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")

    def ctl(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["systemctl", "--user", *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if check and result.returncode:
            raise click.ClickException(
                result.stderr.strip() or result.stdout.strip() or "systemctl failed"
            )
        return result

    def active(self, unit: str) -> bool:
        return self.ctl("is-active", "--quiet", unit, check=False).returncode == 0

    def ready(self, model: Entry) -> bool:
        return self.active(model.unit) and healthy(model.port, "/health")

    def backend_command(self, model: Model) -> list[str]:
        server = os.environ.get("LLAMA_SERVER") or str(
            self.store.data / "runtime/llama.cpp/build/bin/llama-server"
        )
        if not Path(server).is_file():
            server = shutil.which(server) or server
        if not Path(server).is_file():
            raise click.ClickException(
                "llama-server is missing. Run scripts/install.sh or set LLAMA_SERVER."
            )
        gguf_file(model.path)
        # Non-causal vision attention requires each image batch to fit in one
        # microbatch. 2048 also holds Gemma 4's default 1120 image tokens.
        batch = 2048 if model.mmproj else 512
        ubatch = batch if model.mmproj else 256
        args = [
            server,
            "--model",
            model.path,
            "--alias",
            model.name,
            "--host",
            "127.0.0.1",
            "--port",
            str(model.port),
            "--n-predict",
            str(model.output),
            "--parallel",
            "1",
            "--threads",
            "8",
            "--threads-batch",
            "8",
            "--flash-attn",
            "on",
            "--cache-type-k",
            model.cache,
            "--cache-type-v",
            model.cache,
            "--batch-size",
            str(batch),
            "--ubatch-size",
            str(ubatch),
            "--jinja",
            "--no-context-shift",
            "--fit",
            "on",
            "--fit-target",
            "1024",
            "--temp",
            str(model.temperature),
            "--top-k",
            str(model.top_k),
            "--top-p",
            str(model.top_p),
            "--min-p",
            "0",
            "--repeat-penalty",
            "1",
        ]
        # Omit --ctx-size in auto mode. Explicit zero disables the upstream fitter!
        if model.context:
            args.extend(["--ctx-size", str(model.context)])
        # Leave n-gpu-layers unset: llama.cpp's fitter uses current free VRAM,
        # offloading every layer when possible and falling back to CPU when necessary.
        if model.mmproj:
            args.extend(["--mmproj", gguf_file(model.mmproj)])
        return args

    def service_command(self, service: Service) -> list[str]:
        if not (Path(service.project) / "pyproject.toml").is_file():
            raise click.ClickException(f"Service project missing: {service.project}")
        return [
            uv_binary(),
            "run",
            "--project",
            service.project,
            "--frozen",
            "--extra",
            service.extra,
            service.script,
            "--host",
            "127.0.0.1",
            "--port",
            str(service.port),
        ]

    def command(self, entry: Entry) -> list[str]:
        if isinstance(entry, Service):
            return self.service_command(entry)
        return self.backend_command(entry)

    def write_unit(
        self, unit: str, args: list[str], *, backend: bool = False, extra_env: tuple[str, ...] = ()
    ):
        command = " ".join(systemd_quote(arg) for arg in args)
        environment = (
            "Environment=CUDA_VISIBLE_DEVICES=0\n"
            if backend
            else ("Environment=LITELLM_LOCAL_MODEL_COST_MAP=True\nEnvironment=DO_NOT_TRACK=1\n")
        )
        environment += "".join(f"Environment={systemd_quote(e)}\n" for e in extra_env)
        if not backend:
            environment += (
                "Environment="
                + systemd_quote("LITELLM_ENV_FILE=" + str(self.store.settings.env_file))
                + "\n"
            )
        atomic_write(
            self.store.units / unit,
            "[Unit]\nDescription=Local LiteLLM model service\n\n[Service]\n"
            f"Type=exec\nExecStart={command}\n{environment}"
            "Restart=no\nTimeoutStopSec=20\nKillMode=control-group\n"
            "UMask=0077\n",
        )

    def configure_proxy(self, entries: list[Entry]):
        self.store.settings.require_key()
        rows = []
        general = {"master_key": "os.environ/LITELLM_MASTER_KEY"}
        routes = [pass_through(e) for e in entries if isinstance(e, Service)]
        if routes:
            general["pass_through_endpoints"] = routes
        for model in (e for e in entries if isinstance(e, Model)):
            rows.append(
                {
                    "model_name": model.name,
                    "litellm_params": {
                        "model": "openai/" + model.name,
                        "api_base": f"http://127.0.0.1:{model.port}/v1",
                        "api_key": "local",
                    },
                    "model_info": {
                        "mode": "chat",
                        "supported_endpoints": ["/v1/chat/completions", "/v1/responses"],
                        "supports_vision": bool(model.mmproj),
                        "supports_function_calling": model.tools,
                        "max_input_tokens": token_limits(model)["max_input_tokens"],
                        "max_output_tokens": token_limits(model)["max_output_tokens"],
                        "input_cost_per_token": 0,
                        "output_cost_per_token": 0,
                    },
                }
            )
            if model.output > 0:
                rows[-1]["litellm_params"]["max_tokens"] = model.output
        atomic_write(
            self.store.proxy_config,
            yaml.safe_dump(
                {
                    "model_list": rows,
                    "general_settings": general,
                    "litellm_settings": {"telemetry": False, "request_timeout": 300},
                    "router_settings": {"num_retries": 0},
                },
                sort_keys=False,
            ),
        )
        proxy = self.store.data / "runtime/proxy/bin/litellm"
        if not proxy.is_file():
            raise click.ClickException("LiteLLM runtime is missing. Run scripts/install.sh.")
        self.write_unit(
            self.proxy_unit,
            [
                sys.executable,
                "-m",
                "litellm_manager.runtime",
                str(proxy),
                "--config",
                str(self.store.proxy_config),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.store.proxy_port),
            ],
        )
        self.ctl("daemon-reload")
        atomic_write(self.store.state / "settings-fingerprint", self.settings_fingerprint())

    def settings_fingerprint(self):
        settings = self.store.settings
        return sha256(
            f"{settings.url}\n{settings.port}\n{settings.require_key()}".encode()
        ).hexdigest()

    def refresh(self, models: list[Entry]):
        was_active = self.active(self.proxy_unit)
        unit_path = self.store.units / self.proxy_unit
        previous = {
            path: path.read_text() if path.exists() else None
            for path in (self.store.proxy_config, unit_path)
        }
        try:
            self.configure_proxy(models)
            if was_active:
                if models:
                    self.ctl("restart", self.proxy_unit)
                    self.wait_ready(
                        self.proxy_unit, self.store.proxy_port, "/health/liveliness", 90
                    )
                else:
                    self.ctl("stop", self.proxy_unit)
        except (Exception, KeyboardInterrupt):
            for path, content in previous.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write(path, content)
            self.ctl("daemon-reload", check=False)
            if was_active:
                self.ctl("restart", self.proxy_unit, check=False)
            raise

    def wait_ready(self, unit: str, port: int, route: str, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.active(unit):
                raise click.ClickException(
                    f"Service exited before it was ready.\n{self.logs(unit)}"
                )
            if healthy(port, route):
                return
            time.sleep(0.5)
        raise click.ClickException(f"Startup timed out after {timeout:g}s.\n{self.logs(unit)}")

    def logs(self, unit: str) -> str:
        return subprocess.run(
            ["journalctl", "--user", "-u", unit, "-n", "20", "--no-pager", "-o", "cat"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()

    def start(self, model: Entry, models: list[Entry], timeout: float):
        started_here = not self.active(model.unit)
        if started_here:
            if not port_free(model.port):
                raise click.ClickException(
                    f"Backend port {model.port} is occupied. Stop its owner first."
                )
            self.write_unit(
                model.unit,
                self.command(model),
                backend=True,
                extra_env=tuple(model.env) if isinstance(model, Service) else (),
            )
            self.ctl("daemon-reload")
            self.ctl("reset-failed", model.unit, check=False)
            self.ctl("start", model.unit)
        try:
            self.wait_ready(model.unit, model.port, "/health", timeout)
            marker = self.store.state / "settings-fingerprint"
            if self.active(self.proxy_unit) and (
                not marker.exists() or marker.read_text() != self.settings_fingerprint()
            ):
                self.refresh(models)
            if not self.active(self.proxy_unit):
                if not port_free(self.store.proxy_port):
                    raise click.ClickException(
                        f"Proxy port {self.store.proxy_port} is occupied; set LITELLM_PORT."
                    )
                self.configure_proxy(models)
                self.ctl("reset-failed", self.proxy_unit, check=False)
                self.ctl("start", self.proxy_unit)
            self.wait_ready(self.proxy_unit, self.store.proxy_port, "/health/liveliness", 90)
        except (Exception, KeyboardInterrupt):
            # A failed/interrupted start must not leave a large GPU allocation behind.
            if started_here:
                self.ctl("stop", model.unit, check=False)
            if not any(self.active(m.unit) for m in models):
                self.ctl("stop", self.proxy_unit, check=False)
            raise

    def stop(self, model: Entry, models: list[Entry]):
        if (self.store.units / model.unit).exists():
            self.ctl("stop", model.unit)
        if (
            not any(self.active(m.unit) for m in models)
            and (self.store.units / self.proxy_unit).exists()
        ):
            self.ctl("stop", self.proxy_unit)


def healthy(port: int, route: str, timeout: float = 0.3) -> bool:
    try:
        # Bypass HTTP_PROXY: these services always bind loopback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}{route}", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def get_json(port: int, route: str, timeout: float = 0.3) -> dict:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}{route}", timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError):
        return {}


def systemd_quote(arg: str) -> str:
    return (
        '"'
        + arg.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        + '"'
    )
