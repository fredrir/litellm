import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .settings import Settings


def hub_query(query: str, *, sort: str = "downloads") -> list[dict]:
    params = {"search": query, "filter": "gguf", "sort": sort, "direction": "-1", "limit": "80"}
    headers = {"User-Agent": "litellm-manager/0.1"}
    token = Settings().values.get("HF_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=3
    ) as response:
        return json.load(response)


def rank_models(models: list[dict], query: str) -> list[dict]:
    normalized = normalize_query(query)
    terms = re.findall(r"[a-z0-9]+", normalized)

    def score(model):
        name = model["id"].casefold()
        leaf = name.split("/")[-1]
        relevance = 6 * (name.startswith(normalized) and bool(query))
        relevance += 5 * (leaf.startswith(normalized) and bool(query))
        relevance += sum(
            3 if term in re.findall(r"[a-z0-9]+", leaf) else 1 if term in name else -8
            for term in terms
        )
        popularity = math.log10(1 + model.get("downloads", 0))
        popularity += 0.4 * math.log10(1 + model.get("likes", 0))
        return relevance + popularity

    return sorted(models, key=lambda m: (-score(m), -m.get("downloads", 0), m["id"]))


def normalize_query(query: str) -> str:
    if "/" in query:
        owner, leaf = query.split("/", 1)
        return owner.casefold() + "/" + normalize_query(leaf)
    spaced = re.sub(r"([a-z])([0-9])", r"\1-\2", query.casefold())
    return "-".join(re.findall(r"[a-z0-9]+", spaced))


def search_models(query: str, cache_dir: Path) -> list[dict]:
    from hashlib import sha256

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (sha256(query.casefold().encode()).hexdigest() + ".json")
    cached = []
    try:
        saved = json.loads(path.read_text())
        cached = saved["models"]
        if time.time() - saved["time"] < 900:
            return cached
    except (OSError, ValueError, KeyError):
        pass
    if os.environ.get("LITELLM_COMPLETION_OFFLINE") == "1":
        return cached
    try:
        rows = hub_query(query)
        # Hub substring search misses punctuation changes such as "gemma 4 12b".
        if not rows and query:
            normalized = normalize_query(query)
            if normalized != query:
                rows = hub_query(normalized)
        rows = rank_models(rows, query)[:30]
        from .core import atomic_write

        atomic_write(path, json.dumps({"time": time.time(), "models": rows}))
        return rows
    except (OSError, ValueError):
        # TAB remains usable offline; never emit network errors into the terminal.
        return cached
