import json
import time

from litellm_manager.hub import rank_models, search_models


def test_exact_prefix_beats_popular_unrelated_name():
    models = [
        {"id": "someone/gemma-4", "downloads": 100},
        {"id": "someone/llama-gemma-adapter", "downloads": 1000000},
    ]
    assert rank_models(models, "gemma-4")[0]["id"] == "someone/gemma-4"


def test_popularity_breaks_relevance_ties():
    models = [{"id": "a/model-GGUF", "downloads": 10}, {"id": "b/model-GGUF", "downloads": 1000}]
    assert rank_models(models, "model")[0]["id"] == "b/model-GGUF"


def test_completion_cache_avoids_repeated_http_calls(tmp_path, monkeypatch):
    calls = []

    def query(value):
        calls.append(value)
        return [{"id": "a/gemma-GGUF", "downloads": 100}]

    monkeypatch.delenv("LITELLM_COMPLETION_OFFLINE", raising=False)
    monkeypatch.setattr("litellm_manager.hub.hub_query", query)
    assert search_models("gemma", tmp_path) == search_models("gemma", tmp_path)
    assert calls == ["gemma"]


def test_offline_returns_stale_results_without_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_COMPLETION_OFFLINE", raising=False)
    monkeypatch.setattr("litellm_manager.hub.hub_query", lambda query: [{"id": "a/model"}])
    results = search_models("model", tmp_path)
    path = next(tmp_path.glob("*.json"))
    data = json.loads(path.read_text())
    data["time"] = time.time() - 3600
    path.write_text(json.dumps(data))

    def failure(query):
        raise OSError("offline")

    monkeypatch.setattr("litellm_manager.hub.hub_query", failure)
    assert search_models("model", tmp_path) == results
