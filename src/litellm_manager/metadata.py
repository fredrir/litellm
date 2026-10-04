from pathlib import Path

import click


def model_metadata(path: str, mmproj: str | None = None) -> dict:
    from gguf import GGUFReader

    try:
        reader = GGUFReader(path)
        architecture = reader.fields["general.architecture"].contents()
        context = int(reader.fields[f"{architecture}.context_length"].contents())
        template = reader.fields.get("tokenizer.chat_template")
        template = template.contents() if template else ""
        parameters = sum(int(t.n_elements) for t in reader.tensors)
        if mmproj:
            parameters += sum(int(t.n_elements) for t in GGUFReader(mmproj).tensors)
        if context < 1:
            raise ValueError("missing positive context length")
        scale = 1e9 if parameters >= 1e9 else 1e6
        return {
            "trained_context": context,
            "params": f"{parameters / scale:.3g}{'B' if scale == 1e9 else 'M'}",
            "tools": "tools" in template and "tool_calls" in template,
        }
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise click.ClickException(
            f"Cannot derive model limits from {Path(path).name}: {exc}"
        ) from exc


def live_context(port: int) -> int | None:
    from .core import get_json

    props = get_json(port, "/props")
    settings = props.get("default_generation_settings", {})
    context = settings.get("n_ctx") or props.get("n_ctx")
    return int(context) if context else None


def token_limits(model, running: bool = False) -> dict:
    context = live_context(model.port) if running else None
    context = context or model.context or model.trained_context
    return {
        "min_input_tokens": 1,
        "min_output_tokens": 1,
        "max_context_tokens": context,
        "max_input_tokens": max(1, context - 1) if context else None,
        "max_output_tokens": min(model.output, context - 1)
        if model.output > 0 and context
        else max(1, context - 1)
        if context
        else None,
        "automatic": model.context == 0,
        "fitted": running,
        "output_budget": "context - tokenized input (including images)",
        "trained_context": model.trained_context,
    }
