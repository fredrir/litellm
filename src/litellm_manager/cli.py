import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .catalog import CATALOG
from .core import Services, Store, available_port, download_model, healthy, resolve

console = Console()
if not console.is_terminal:
    console = Console(width=140)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.1.0")
def cli():
    """Manage local llama.cpp models through LiteLLM at http://127.0.0.1:4010/v1.

    Full Hugging Face IDs and unique short model names are accepted.
    LITELLM_PORT overrides the gateway port; LLAMA_SERVER overrides the runtime.
    """


@cli.command()
@click.argument("model")
@click.option("--repo", help="GGUF repository override for a Hugging Face model ID.")
@click.option("--file", "filename", help="Exact GGUF filename in the repository.")
@click.option("--mmproj", help="Matching vision projector filename (or path for a local model).")
@click.option(
    "--context", type=click.IntRange(min=512), help="Total tokens per request, including output."
)
@click.option("--output", type=click.IntRange(min=1), help="Default maximum generated tokens.")
@click.option("--params", help="Parameter count for the table, e.g. 12B.")
@click.option("--tools/--no-tools", default=None, help="Declare function calling support.")
def add(model, **options):
    """Download and register a model and its vision projector. Existing downloads are reused."""
    store = Store()
    with console.status(f"Adding {model}…"):
        from huggingface_hub.errors import HfHubHTTPError

        # Download outside the mutation lock so existing models remain controllable.
        try:
            new = download_model(model, store.read(), store.proxy_port, **options)
        except HfHubHTTPError as exc:
            raise click.ClickException(
                f"Hugging Face download failed: {exc}. "
                "For gated models accept the repository terms and set HF_TOKEN."
            ) from exc
        with store.transaction():
            models = store.read()
            if any(m.name.casefold() == new.name.casefold() for m in models):
                raise click.ClickException(f"Already added: {new.name}")
            new.port = available_port(models, store.proxy_port)
            models.append(new)
            services = Services(store)
            services.refresh(models)
            store.save(models)
    console.print(f"[green]Added[/green] {new.name}", markup=True, highlight=False)


@cli.command()
@click.argument("model")
def remove(model):
    """Stop and unregister a model. Shared Hugging Face cache files are kept for reuse."""
    store = Store()
    with store.transaction():
        models = store.read()
        target = resolve(models, model)
        services = Services(store)
        services.stop(target, models)
        remaining = [m for m in models if m.name != target.name]
        services.refresh(remaining)
        (store.units / target.unit).unlink(missing_ok=True)
        services.ctl("daemon-reload")
        store.save(remaining)
        console.print(f"[yellow]Removed[/yellow] {target.name}; downloaded cache retained.")


@cli.command("list")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable model details.")
def list_models(as_json):
    """Show registered models, configured token budgets, API support, and live status."""
    store = Store()
    models = store.read()
    services = Services(store)
    proxy_ready = services.active(services.proxy_unit) and healthy(
        store.proxy_port, "/health/liveliness"
    )
    rows = [(m, proxy_ready and services.ready(m)) for m in models]
    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        **asdict(m),
                        "vision": bool(m.mmproj),
                        "api": ["responses", "chat-completions"],
                        "status": "on" if on else "off",
                    }
                    for m, on in rows
                ],
                indent=2,
            )
        )
        return
    table = Table(header_style="bold cyan", border_style="bright_black", expand=False)
    for column in ("model", "params", "context", "output", "vision", "tools", "api", "status"):
        table.add_column(column, no_wrap=column not in ("model", "api"), overflow="fold")
    for m, on in rows:
        table.add_row(
            m.name,
            m.params,
            f"{m.context:,}",
            f"{m.output:,}",
            "[green]yes[/green]" if m.mmproj else "[dim]no[/dim]",
            "[green]yes[/green]" if m.tools else "[dim]no[/dim]",
            "responses | chat-completions",
            "[green]on[/green]" if on else "[dim]off[/dim]",
        )
    console.print(table)
    console.print(
        f"[dim]Context = prompt + images + output. Output = default generation limit.\n"
        f"Gateway: http://127.0.0.1:{store.proxy_port}/v1[/dim]"
    )


@cli.command()
@click.argument("model")
@click.option(
    "--timeout",
    type=click.FloatRange(min=1, max=3600),
    default=180,
    show_default=True,
    help="Maximum seconds to wait for the model to load.",
)
def start(model, timeout):
    """Start a model and the shared gateway; wait until both are ready."""
    store = Store()
    with store.transaction():
        models = store.read()
        target = resolve(models, model)
        with console.status(f"Starting {target.name}…"):
            Services(store).start(target, models, timeout)
        console.print(
            f"[green]Serving[/green] {target.name} at http://127.0.0.1:{store.proxy_port}/v1"
        )


@cli.command()
@click.argument("model")
def stop(model):
    """Stop a model and release its GPU memory. Stop the gateway after the last model."""
    store = Store()
    with store.transaction():
        models = store.read()
        target = resolve(models, model)
        Services(store).stop(target, models)
        console.print(f"[yellow]Stopped[/yellow] {target.name}")


@cli.command()
@click.argument("model", required=False)
@click.option("--follow", "-f", is_flag=True, help="Follow new log entries.")
def logs(model, follow):
    """Show a model's logs, or gateway logs when no model is given."""
    store = Store()
    services = Services(store)
    unit = resolve(store.read(), model).unit if model else services.proxy_unit
    args = ["journalctl", "--user", "-u", unit, "--no-pager", "-o", "cat", "-n", "50"]
    if follow:
        args.append("-f")
    subprocess.run(args, env=services.env, check=False)


@cli.command()
def completion():
    """Print the full zsh completion script (installed automatically by scripts/install.sh)."""
    click.echo((Path(__file__).with_name("_litellm")).read_text(), nl=False)


@cli.command("_complete", hidden=True)
@click.argument("command", required=False)
def complete(command):
    # No network, systemctl, or heavyweight imports during TAB completion.
    names = {m.name for m in Store().read()}
    if command == "add":
        names |= set(CATALOG)
    leaves = [name.split("/")[-1] for name in names]
    names |= {leaf for leaf in leaves if leaves.count(leaf) == 1}
    click.echo("\n".join(sorted(names)))


def main():
    try:
        cli()
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        click.ClickException(str(exc)).show()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
