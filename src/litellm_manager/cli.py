import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .catalog import CATALOG, SERVICES, service_for
from .core import (
    Service,
    Services,
    Store,
    add_service,
    available_port,
    download_model,
    healthy,
    resolve,
)
from .metadata import token_limits

console = Console()
if not console.is_terminal:
    console = Console(width=140)


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option("0.1.0")
@click.pass_context
def cli(ctx):
    """Local models, native CUDA, one LiteLLM gateway.

    Full Hugging Face IDs and unique short model names are accepted.
    Configure LITELLM_URL and LITELLM_API_KEY in .env.
    """
    if ctx.invoked_subcommand is None:
        store = Store()
        console.print(
            Panel(
                Text.assemble(
                    ("Lite", "bold cyan"),
                    ("LLM", "bold magenta"),
                    (f"\n{store.settings.api_url}", "bright_blue"),
                ),
                border_style="cyan",
                padding=(1, 2),
            )
        )
        ctx.invoke(list_models)
        commands = Text()
        for name, style, argument in (
            ("add", "bold green", "<model>"),
            ("start", "bold cyan", "<model>"),
            ("stop", "bold yellow", "<model>"),
            ("remove", "bold red", "<model>"),
            ("list", "bold blue", None),
            ("logs", "bold magenta", None),
            ("--help", "bold", None),
        ):
            if len(commands):
                commands.append("\n")
            commands.append(name.ljust(7), style)
            if argument:
                commands.append(argument, "dim")
        console.print(
            Panel(
                commands,
                title="[bold cyan]Commands[/]",
                border_style="bright_black",
            )
        )


@cli.command()
@click.argument("model")
@click.option("--repo", help="GGUF repository override for a Hugging Face model ID.")
@click.option("--file", "filename", help="Exact GGUF filename in the repository.")
@click.option("--mmproj", help="Matching vision projector filename (or path for a local model).")
@click.option(
    "--context",
    type=click.IntRange(min=1),
    help="Optional context ceiling; defaults to automatic memory fit.",
)
@click.option(
    "--output",
    type=click.IntRange(min=1),
    help="Optional output ceiling; defaults to remaining context.",
)
@click.option("--params", help="Parameter count for the table, e.g. 12B.")
@click.option("--tools/--no-tools", default=None, help="Declare function calling support.")
def add(model, **options):
    """Download and register a model and its vision projector, or a managed service."""
    store = Store()
    with console.status(f"Adding {model}…"):
        from huggingface_hub.errors import HfHubHTTPError

        # Download outside the mutation lock so existing models remain controllable.
        try:
            if service_for(model):
                if any(value is not None for value in options.values()):
                    raise click.ClickException("Options apply to GGUF models only.")
                new = add_service(model, store.read(), store.proxy_port)
            else:
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
                        "api": [f"pass-through {m.route}"],
                        "status": "on" if on else "off",
                    }
                    if isinstance(m, Service)
                    else {
                        **asdict(m),
                        "vision": bool(m.mmproj),
                        "api": ["responses", "chat-completions"],
                        "status": "on" if on else "off",
                        "limits": token_limits(m, on),
                    }
                    for m, on in rows
                ],
                indent=2,
            )
        )
        return
    table = Table(
        header_style="bold cyan", border_style="blue", expand=False, row_styles=["", "on grey7"]
    )
    for column in ("model", "params", "context", "output", "vision", "tools", "api", "status"):
        table.add_column(column, no_wrap=column not in ("model", "api"), overflow="fold")
    for m, on in rows:
        label = Text()
        if "/" in m.name:
            owner, leaf = m.name.rsplit("/", 1)
            label.append(owner + "/", "dim cyan")
            label.append(leaf, "bold white")
        else:
            label.append(m.name, "bold white")
        status = "[bold green]● on[/]" if on else "[dim]○ off[/]"
        if isinstance(m, Service):
            na = "[dim]n/a[/]"
            table.add_row(label, na, na, na, na, na, f"[blue]pass-through[/] {m.route}", status)
            continue
        limits = token_limits(m, on)
        context = f"{limits['max_context_tokens']:,}" if limits["max_context_tokens"] else "unknown"
        if not m.context and not on:
            context = "auto ≤ " + context
        table.add_row(
            label,
            f"[magenta]{m.params}[/]",
            f"[cyan]{context}[/]",
            f"[yellow]{m.output:,}[/]" if m.output > 0 else "[yellow]remaining[/]",
            "[green]✓[/]" if m.mmproj else "[dim]—[/]",
            "[green]✓[/]" if m.tools else "[dim]—[/]",
            "[blue]responses[/] | [bright_blue]chat-completions[/]",
            status,
        )
    console.print(table)


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
            f"[bold green]● Ready[/] [bold]{target.name}[/] at [bright_blue]{store.settings.api_url}[/]"
        )
        if not healthy(store.settings.public_port, "/health/liveliness"):
            console.print(
                "[yellow]Activate the configured URL:[/] [bold]sudo ./scripts/setup-url.sh[/]\n"
                f"[dim]Internal API is ready at http://127.0.0.1:{store.proxy_port}/v1[/]"
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
@click.argument("query", default="")
@click.option("--describe", is_flag=True)
def complete(command, query, describe):
    store = Store()
    models = store.read()
    if command == "add":
        from .hub import normalize_query, search_models

        results = search_models(query, store.state / "hub-completion")
        remote = [r["id"] for r in results]
        compact = normalize_query(query).replace("-", "")
        local = [
            name
            for name in [*CATALOG, *SERVICES]
            if compact in normalize_query(name).replace("-", "")
        ]
        names = list(dict.fromkeys([*remote, *sorted(local)]))
        by_id = {r["id"]: r for r in results}
        for name in names:
            description = "service" if name in SERVICES else "curated preset"
            if name in by_id:
                r = by_id[name]
                description = f"↓ {r.get('downloads', 0):,}  ♥ {r.get('likes', 0):,}"
            click.echo(f"{name}:{description}" if describe else name)
        return
    if command == "stop":
        services = Services(store)
        models = [m for m in models if services.active(m.unit)]
    names = {m.name for m in models}
    # Offer the short name when it uniquely identifies a model, otherwise the full ID.
    leaves = [name.split("/")[-1].casefold() for name in names]
    unique = {leaf for leaf in leaves if leaves.count(leaf) == 1}
    names = {
        name.split("/")[-1] if name.split("/")[-1].casefold() in unique else name for name in names
    }
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
