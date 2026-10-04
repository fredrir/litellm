"""Opt-in live checks: uv run python scripts/verify.py [full-model-id ...]."""

import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from litellm_manager.core import Service, Services, Store, resolve


def request(port, route, body=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}" + (route if route.startswith("/") else f"/v1/{route}"),
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + Store().settings.require_key(),
        },
    )
    try:
        with opener.open(req, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode()) from exc


def main():
    store = Store()
    models = store.read()
    services = Services(store)
    targets = [resolve(models, name) for name in sys.argv[1:]] if sys.argv[1:] else models
    fixture = Path(__file__).parents[1] / "tests/fixtures/ocr.png"
    image = "data:image/png;base64," + base64.b64encode(fixture.read_bytes()).decode()
    for model in targets:
        already_active = services.active(model.unit)
        try:
            subprocess.run(
                [sys.executable, "-m", "litellm_manager.cli", "start", model.name], check=True
            )
            if isinstance(model, Service):
                health = request(store.proxy_port, f"{model.route}/health")
                assert health.get("ready"), health
                layout = request(
                    store.proxy_port,
                    f"{model.route}/v1/layout",
                    {"png": base64.b64encode(fixture.read_bytes()).decode()},
                )
                assert layout["blocks"] or layout["lines"], layout
                print(
                    model.name,
                    "layout:",
                    len(layout["blocks"]),
                    "blocks,",
                    len(layout["lines"]),
                    "lines",
                    flush=True,
                )
                continue
            prompt = (
                "Convert this page to docling."
                if "docling" in model.name
                else "OCR:"
                if "PaddleOCR" in model.name
                else "Read the text in this image."
            )
            content = [{"type": "text", "text": prompt}]
            inputs = [{"type": "input_text", "text": prompt}]
            if model.mmproj:
                content.append({"type": "image_url", "image_url": {"url": image}})
                inputs.append({"type": "input_image", "image_url": image})
            chat = request(
                store.proxy_port,
                "chat/completions",
                {
                    "model": model.name,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 128,
                },
            )
            text = chat["choices"][0]["message"]["content"]
            assert text, chat
            print(model.name, "chat + vision:", text, flush=True)
            responses = request(
                store.proxy_port,
                "responses",
                {
                    "model": model.name,
                    "input": [{"role": "user", "content": inputs}],
                    "max_output_tokens": 128,
                    "store": False,
                },
            )
            assert responses.get("output"), responses
            print(model.name, "responses + vision:", json.dumps(responses["output"]), flush=True)
            if model.tools:
                tool = request(
                    store.proxy_port,
                    "chat/completions",
                    {
                        "model": model.name,
                        "max_tokens": 256,
                        "messages": [{"role": "user", "content": "Call add with a=2 and b=3."}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "description": "Add two integers.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "a": {"type": "integer"},
                                            "b": {"type": "integer"},
                                        },
                                        "required": ["a", "b"],
                                        "additionalProperties": False,
                                    },
                                },
                            }
                        ],
                        "tool_choice": {"type": "function", "function": {"name": "add"}},
                    },
                )
                calls = tool["choices"][0]["message"]["tool_calls"]
                assert calls[0]["function"]["name"] == "add", tool
                assert json.loads(calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}, tool
                print(model.name, "tools:", json.dumps(calls), flush=True)
        finally:
            if not already_active:
                subprocess.run(
                    [sys.executable, "-m", "litellm_manager.cli", "stop", model.name], check=True
                )


if __name__ == "__main__":
    main()
