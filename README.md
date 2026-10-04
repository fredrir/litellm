# litellm

Native CUDA llama.cpp models through a local LiteLLM gateway.

```zsh
litellm                         # dashboard
litellm --help
litellm list --json
litellm add <TAB>               # popular Hugging Face GGUF models
litellm add gemma4<TAB>         # model search, ranked by relevance/downloads/likes
litellm add unsloth/gemma-4-12b-it-GGUF
litellm start gemma-4-12b-it-GGUF
litellm stop gemma-4-12b-it-GGUF
litellm remove gemma-4-12b-it-GGUF
litellm logs gemma-4-12b-it-GGUF --follow
```

## Configuration

Edit the repository's **`.env`**. It is gitignored and loaded from any working directory. `.env.example` contains the supported settings; the CLI never generates an API key.

| Env | Default / behavior |
| --- | --- |
| `LITELLM_URL` | `http://litellm.localhost`; HTTP localhost/subdomains, optional port |
| `LITELLM_API_KEY` | Your chosen key; required for the gateway |
| `LITELLM_PORT` | `4010`; internal loopback gateway port |
| `LITELLM_ENV_FILE` | Repository `.env`; optional alternate file |
| `HF_TOKEN` | Optional Hugging Face download/search authentication |
| `LLAMA_SERVER` | Managed native CUDA build; optional override |
| `XDG_CONFIG_HOME` | `~/.config`; registry and credential-free proxy config |
| `XDG_DATA_HOME` | `~/.local/share`; managed runtimes |
| `XDG_STATE_HOME` | `~/.local/state`; locks and Hugging Face completion cache |

The gateway reads the API key directly from `.env` at startup. Changing `.env` and running `start` refreshes gateway settings. Generated units and proxy YAML contain no API key. Shared downloaded weights remain cached after `remove`.

## Local URL

```zsh
sudo ./scripts/setup-url.sh
```

The script installs two small systemd units: a socket on IPv4/IPv6 loopback and `systemd-socket-proxyd` forwarding to `LITELLM_PORT`. Port 80 needs administrator privileges. Rerun after changing URL/ports. For a public port above 1023 (e.g. `http://litellm.localhost:8080`), run the script without sudo to install user units.

`.localhost` names resolve to loopback automatically on this machine. Clients use `LITELLM_URL` plus `/v1` as their SDK base URL and `LITELLM_API_KEY` from `.env` as their key.

## Automatic token budgets

| Value | Source / behavior |
| --- | --- |
| Trained context | Architecture context length in GGUF metadata |
| Serving context | llama.cpp memory fitter, using free device memory on start |
| Live context in `list` | Backend `/props`; stopped models show `auto ≤ trained maximum` |
| Minimum input/output | One token; structural minimum, not a generation target |
| Maximum input/output | Serving context minus space for the other side |
| Per-request output | Remaining context after backend tokenization, including image tokens |
| Output stopping | EOS or context exhaustion; context shifting disabled |
| Explicit ceilings | Optional `add --context N --output N`; validated against metadata |
| Detailed limits | `litellm list --json` → `limits` |

Model context includes input and output. There is no arbitrary default output cap. Limits change when the model is started with a different available memory budget; the allocated context stays stable while serving.

## Hardware and models

| Setting | Value |
| --- | --- |
| Hardware | Ryzen 7 9800X3D, RTX 5070 Ti 16 GB, 32 GB RAM |
| Build | llama.cpp v0.5.0, native CPU instructions, CUDA SM 120a, Release |
| CPU threads | 8 generation / 8 prompt processing |
| GPU | Memory fitter, 1 GiB margin, CPU fallback when necessary |
| Attention | Flash Attention; Q8 KV for Gemma, F16 for OCR |
| Request slots / batch / microbatch | 1 / 512 / 256 |
| Granite | F16 + matching vision projector |
| PaddleOCR | Official GGUF + matching vision projector |
| Google Gemma | Official QAT Q4_0 + matching vision projector |
| Unsloth Gemma | `UD-Q8_K_XL` (12.7 GiB weights) + F16 projector |
| Gemma sampling | temperature 1, top-p 0.95, top-k 64, min-p 0 |
| OCR sampling | temperature 0, repetition penalty 1 |

Unsloth Q8 weights exceed current free VRAM while other GPU services are resident, so the fitter offloads as many layers as possible. It cannot match the fully GPU-resident QAT model's speed under that memory pressure.

Custom GGUF repositories with several quantizations require `--file`; `--mmproj` selects the matching projector. Model parameter counts, trained context, and tool template support are read from the actual GGUF. Local GGUF paths are also accepted. API requests use the full registered ID.

Granite's prompt is `Convert this page to docling.` and its output is DocTags. PaddleOCR uses `OCR:`, `Table Recognition:`, or `Formula Recognition:`. Send images as OpenAI data-URL image content blocks. PDF rasterization/layout and Markdown conversion belong to the Docling/PaddleOCR pipelines.

## Completion and verification

Hugging Face completions search GGUF repositories, normalize inputs such as `gemma4`, rank name relevance ahead of popularity, display downloads/likes, and retain API results for 15 minutes. Network requests have bounded timeouts; stale cached results and curated models remain available offline. `start`, `stop`, `remove`, and `logs` complete registered models without network access.

```zsh
./scripts/install.sh
source ~/.zfunc/_litellm       # or open a new shell
uv run pytest -q
uv run ruff check src tests scripts/verify.py
uv run python scripts/verify.py   # live API/vision/tool checks; restores prior running states
```

Add/remove refreshes a running gateway's collection. Do that between requests. Model processes are systemd user services and survive terminal exit.

## Primary sources

| Source | Used for |
| --- | --- |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/server/README.md) | APIs, memory fit, context, Flash Attention, Jinja |
| [LiteLLM compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible) | Local provider routing |
| [Hugging Face search](https://huggingface.co/docs/huggingface_hub/en/guides/search) | Search and popularity sorting |
| [Unsloth Gemma GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | Exact quantization and matching projector |
| [Gemma model card](https://huggingface.co/google/gemma-4-12B-it) | Sampling |
