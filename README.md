# litellm

Local llama.cpp CUDA inference through a LiteLLM gateway. Research and pinned releases: **2026-10-04**.

```zsh
./scripts/install.sh
litellm add ibm-granite/granite-docling-258M
litellm add PaddlePaddle/PaddleOCR-VL-1.6
litellm add google/gemma-4-12B-it
litellm list
litellm start gemma-4-12B-it
litellm stop gemma-4-12B-it
litellm remove gemma-4-12B-it
litellm --help
```

| Model | Weights | Context tokens | Default output tokens | Vision | Tools |
| --- | --- | ---: | ---: | --- | --- |
| granite-docling-258M | F16 | 8,192 | 4,096 | yes | no |
| PaddleOCR-VL-1.6 | Official GGUF | 16,384 | 8,192 | yes | no |
| gemma-4-12B-it | Official Google QAT Q4_0 | 16,384 | 4,096 | yes | yes |

Context includes prompt, image tokens, and generated output. Output is a configurable default, bounded by remaining context; clients can request a different limit. Registered models can run together if memory permits. Idle models allocate no GPU memory.

| Setting | Value |
| --- | --- |
| Machine | Ryzen 7 9800X3D, 32 GB RAM, RTX 5070 Ti 16 GB |
| Build | llama.cpp v0.5.0, native CPU instructions, CUDA SM 120a, Release |
| Generation / prompt threads | 8 / 8 |
| Flash Attention | on |
| Concurrent requests per model | 1; additional requests queue |
| Batch / microbatch | 512 / 256 |
| Gemma KV cache | Q8_0 |
| OCR KV cache | F16 |
| GPU offload | llama.cpp memory fitter, 1 GiB margin, CPU fallback |
| Gemma sampling | temperature 1, top-p 0.95, top-k 64, min-p 0 |
| OCR sampling | temperature 0, repetition penalty 1 |
| Gateway | `http://127.0.0.1:4010/v1` |
| API | `/v1/chat/completions`, `/v1/responses` |
| Lifecycle | systemd user services; survives terminal exit, stops on logout; no boot autostart |

## Requests

Use the full registered model ID in API requests; CLI commands also accept unique short names. The first add generates a local gateway key stored with owner-only permissions. Use it as the SDK's `api_key`.

```zsh
export LITELLM_API_KEY="$(cat ~/.config/litellm-manager/api-key)"

curl http://127.0.0.1:4010/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-12B-it","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

curl http://127.0.0.1:4010/v1/responses \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-12B-it","input":"Hello","max_output_tokens":128,"store":false}'
```

Send images using OpenAI `image_url` content blocks with data URLs. Granite's prompt is `Convert this page to docling.`; its output is DocTags. PaddleOCR uses prompts such as `OCR:`, `Table Recognition:`, or `Formula Recognition:`. These models serve the recognition step; PDF rasterization, document layout, and conversion into Markdown are separate Docling/PaddleOCR pipeline steps.

## Other models and tuning

```zsh
litellm add owner/model --repo owner/model-GGUF --file model-Q4_K_M.gguf \
  --mmproj mmproj-model-f16.gguf --context 16384 --output 4096 --params 7B --tools
litellm add /absolute/path/model.gguf --mmproj /absolute/path/mmproj.gguf
litellm list --json
litellm logs gemma-4-12B-it
litellm logs --follow
```

To change a preset's token budgets, remove and re-add it with `--context`/`--output`; downloads are reused. Custom repositories with multiple GGUFs require `--file`. Split GGUF shards require a pre-downloaded local first shard and its sibling shards. Custom models' parameter counts/tool support are supplied with `--params`/`--tools`; `?` means unknown.

| Env | Default / behavior |
| --- | --- |
| `LITELLM_PORT` | `4010`; stop all managed models before changing |
| `LLAMA_SERVER` | Managed native build; accepts executable path or name |
| `HF_TOKEN` | Optional Hugging Face authentication; required for gated repositories |
| `XDG_CONFIG_HOME` | `~/.config`; registry in `litellm-manager/models.json` |
| `XDG_DATA_HOME` | `~/.local/share`; runtimes in `litellm-manager/runtime` |
| `XDG_STATE_HOME` | `~/.local/state`; mutation lock in `litellm-manager` |

`remove` stops and unregisters a model while retaining shared Hugging Face cache files. Add/remove restarts a running gateway to load its new model collection; perform those operations between requests. Start/stop does not restart a running gateway. Readiness checks determine the displayed on/off status.

## Completion and verification

The installer writes `~/.zfunc/_litellm`, which your zsh configuration already loads. Open a new shell, or run:

```zsh
autoload -Uz compinit
compinit
source ~/.zfunc/_litellm
uv run pytest -q
uv run ruff check src tests
uv run python scripts/verify.py
```

Commands, flags, token budget choices, local GGUF paths, and registered model IDs complete without contacting Hugging Face or starting services.

## Sources

| Source | Used for |
| --- | --- |
| [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/v0.5.0/tools/server/README.md) | API, memory fitter, Flash Attention, KV cache, Jinja tools |
| [llama.cpp build](https://github.com/ggml-org/llama.cpp/blob/v0.5.0/docs/build.md) | Native CUDA build |
| [LiteLLM compatible endpoints](https://docs.litellm.ai/docs/providers/openai_compatible) | Local OpenAI provider routing |
| [LiteLLM Responses](https://docs.litellm.ai/docs/response_api) | Responses gateway |
| [Gemma overview](https://ai.google.dev/gemma/docs/core) / [QAT GGUF](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) | Official QAT, memory planning, modalities |
| [Gemma model card](https://huggingface.co/google/gemma-4-12B-it) | Sampling and model context |
| [Granite model](https://huggingface.co/ibm-granite/granite-docling-258M) / [GGUF](https://huggingface.co/ggml-org/granite-docling-258M-GGUF) | DocTags, context, weights/projector |
| [PaddleOCR GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF) | Official weights/projector and OCR prompts |
