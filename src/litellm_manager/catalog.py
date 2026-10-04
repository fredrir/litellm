from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    repo: str
    revision: str
    filename: str
    mmproj: str
    params: str
    tools: bool = False
    cache: str = "f16"
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0


# Verified against upstream model configs and GGUF repositories, 2026-10-04.
CATALOG = {
    "ibm-granite/granite-docling-258M": Preset(
        "ggml-org/granite-docling-258M-GGUF",
        "684c1800c5fbd614e29b4d1a82177e6c122af1f3",
        "granite-docling-258M-f16.gguf",
        "mmproj-granite-docling-258M-f16.gguf",
        "258M",
    ),
    "PaddlePaddle/PaddleOCR-VL-1.6": Preset(
        "PaddlePaddle/PaddleOCR-VL-1.6-GGUF",
        "511b09642bb324401f15f97cc23bc67e8f0a291d",
        "PaddleOCR-VL-1.6-GGUF.gguf",
        "PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
        "900M",
    ),
    "google/gemma-4-12B-it": Preset(
        "google/gemma-4-12B-it-qat-q4_0-gguf",
        "29d097773436b69ff9feafd636ab4cf873786537",
        "gemma-4-12b-it-qat-q4_0.gguf",
        "mmproj-gemma-4-12b-it-qat-q4_0.gguf",
        "12B",
        tools=True,
        cache="q8_0",
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    ),
    "unsloth/gemma-4-12b-it-GGUF": Preset(
        "unsloth/gemma-4-12b-it-GGUF",
        "fc034cfff751157913579611efad8462ac1be606",
        "gemma-4-12b-it-UD-Q8_K_XL.gguf",
        "mmproj-F16.gguf",
        "12B",
        tools=True,
        cache="q8_0",
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    ),
}


def preset_for(name: str) -> tuple[str, Preset] | None:
    for key, value in CATALOG.items():
        if name.casefold() in (key.casefold(), key.split("/")[-1].casefold()):
            return key, value
    return None
