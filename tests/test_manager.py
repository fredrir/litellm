import json
import subprocess
from dataclasses import replace

import click
import pytest
import yaml
from click.testing import CliRunner

from litellm_manager.catalog import CATALOG, SERVICES
from litellm_manager.cli import cli
from litellm_manager.core import (
    Model,
    Service,
    Services,
    Store,
    add_service,
    download_model,
    resolve,
)
from litellm_manager.settings import PROJECT


@pytest.fixture
def store(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LITELLM_API_KEY=my-test-key\nLITELLM_URL=http://litellm.localhost\n")
    monkeypatch.setenv("LITELLM_ENV_FILE", str(env_file))
    monkeypatch.setenv("LITELLM_COMPLETION_OFFLINE", "1")
    for env, directory in [
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
    ]:
        monkeypatch.setenv(env, str(tmp_path / directory))
    monkeypatch.delenv("LITELLM_PORT", raising=False)
    return Store()


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.gguf"
    write_gguf(path)
    return Model(
        "test/model", "local", "local", str(path), None, "1B", 8192, 2048, None, False, 8100
    )


@pytest.fixture
def service(tmp_path):
    project = tmp_path / "pp-structure"
    project.mkdir()
    (project / "pyproject.toml").touch()
    return Service(
        "PaddlePaddle/PP-StructureV3",
        str(project),
        "pp-structure",
        "/pp-structure",
        "gpu",
        8103,
        ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"],
    )


def write_gguf(path):
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(8192)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_registry_round_trip_and_unique_short_names(store, model):
    store.save([model])
    assert resolve(store.read(), "MODEL") == model
    other = replace(model, name="another/model", port=8101)
    with pytest.raises(click.ClickException, match="Ambiguous"):
        resolve([model, other], "model")
    assert resolve([model, other], "test/model") == model


def test_unknown_model_has_actionable_error(store):
    result = CliRunner().invoke(cli, ["stop", "missing"])
    assert result.exit_code == 1
    assert "Unknown model" in result.output
    assert "litellm list" in result.output


def test_corrupt_registry_is_not_silently_overwritten(store):
    store.registry.parent.mkdir(parents=True)
    store.registry.write_text("broken")
    result = CliRunner().invoke(cli, ["list"])
    assert result.exit_code == 1
    assert "Cannot read" in result.output
    assert store.registry.read_text() == "broken"


def test_reject_output_larger_than_context_before_network(store):
    with pytest.raises(click.ClickException, match="smaller than"):
        download_model("google/gemma-4-12B-it", [], 4010, context=8192, output=20000)


def test_reject_context_beyond_model_limit(store, model):
    with pytest.raises(click.ClickException, match="trained context"):
        download_model(model.path, [], 4010, context=16384)


def test_local_add_requires_gguf_and_detects_duplicates(store, tmp_path):
    path = tmp_path / "local.gguf"
    path.write_bytes(b"invalid")
    with pytest.raises(click.ClickException, match="Not a GGUF"):
        download_model(str(path), [], 4010)
    write_gguf(path)
    added = download_model(str(path), [], 4010)
    assert added.name == "local"
    with pytest.raises(click.ClickException, match="Already added"):
        download_model(str(path), [added], 4010)


def test_add_completion_reads_registry_without_network_or_systemd(store, model, monkeypatch):
    store.save([model])

    def forbidden(*args, **kwargs):
        pytest.fail("Add completion must not call systemctl or the network")

    monkeypatch.setattr(subprocess, "run", forbidden)
    result = CliRunner().invoke(cli, ["_complete", "add"])
    assert result.exit_code == 0
    assert set(result.output.splitlines()) == set(CATALOG) | set(SERVICES)


def test_stop_completion_only_lists_running_models(store, model, monkeypatch):
    running = replace(model, name="test/running", port=8101)
    stopped = replace(model, name="test/stopped", port=8102)
    store.save([running, stopped])
    monkeypatch.setattr(Services, "active", lambda self, unit: unit == running.unit)
    result = CliRunner().invoke(cli, ["_complete", "stop"])
    assert result.exit_code == 0
    assert set(result.output.splitlines()) == {"running"}


def test_completion_prefers_full_names_for_ambiguous_short_names(store, model):
    store.save([model, replace(model, name="another/model", port=8101)])
    result = CliRunner().invoke(cli, ["_complete", "logs"])
    assert result.exit_code == 0
    assert set(result.output.splitlines()) == {"test/model", "another/model"}


def test_failed_registry_change_restores_gateway_config(store, monkeypatch):
    services = Services(store)
    store.proxy_config.parent.mkdir(parents=True)
    store.proxy_config.write_text("previous config")
    monkeypatch.setattr(services, "active", lambda unit: True)
    monkeypatch.setattr(services, "ctl", lambda *args, **kwargs: None)

    def broken_config(models):
        store.proxy_config.write_text("broken new config")
        raise click.ClickException("new configuration failed")

    monkeypatch.setattr(services, "configure_proxy", broken_config)
    with pytest.raises(click.ClickException, match="configuration failed"):
        services.refresh([])
    assert store.proxy_config.read_text() == "previous config"


def test_list_reports_off_when_gateway_is_down_even_with_backend_up(store, model, monkeypatch):
    store.save([model])
    monkeypatch.setattr(Services, "active", lambda self, unit: unit == model.unit)
    monkeypatch.setattr(Services, "ready", lambda self, m: True)
    result = CliRunner().invoke(cli, ["list", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)[0]["status"] == "off"


def test_proxy_routes_only_to_local_backends(store, model, monkeypatch):
    runtime = store.data / "runtime/proxy/bin/litellm"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(Services, "ctl", lambda *args, **kwargs: None)
    Services(store).configure_proxy([model])
    assert not (store.config / "api-key").exists()
    assert "my-test-key" not in store.proxy_config.read_text()
    assert (
        yaml.safe_load(store.proxy_config.read_text())["general_settings"]["master_key"]
        == "os.environ/LITELLM_MASTER_KEY"
    )
    row = yaml.safe_load(store.proxy_config.read_text())["model_list"][0]
    assert row["model_name"] == model.name
    assert row["litellm_params"]["model"] == "openai/test/model"
    assert row["litellm_params"]["api_base"] == "http://127.0.0.1:8100/v1"
    assert row["litellm_params"]["max_tokens"] == 2048
    assert '--host" "127.0.0.1' in (store.units / Services.proxy_unit).read_text()


def test_failed_start_cleans_up_gpu_process_and_unused_gateway(store, model, monkeypatch):
    services = Services(store)
    calls = []
    monkeypatch.setattr(services, "active", lambda unit: False)
    monkeypatch.setattr(services, "ctl", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(services, "backend_command", lambda m: ["/fake/llama-server"])
    monkeypatch.setattr(services, "write_unit", lambda *a, **kw: None)
    monkeypatch.setattr("litellm_manager.core.port_free", lambda port: True)

    def failure(*args):
        raise click.ClickException("load failed")

    monkeypatch.setattr(services, "wait_ready", failure)
    with pytest.raises(click.ClickException, match="load failed"):
        services.start(model, [model], 1)
    assert ("stop", model.unit) in calls
    assert ("stop", services.proxy_unit) in calls


def test_stop_preserves_gateway_for_other_running_models(store, model, monkeypatch):
    other = replace(model, name="other", port=8101)
    services = Services(store)
    store.units.mkdir(parents=True)
    (store.units / model.unit).touch()
    (store.units / services.proxy_unit).touch()
    calls = []
    monkeypatch.setattr(services, "active", lambda unit: unit == other.unit)
    monkeypatch.setattr(services, "ctl", lambda *a, **kw: calls.append(a))
    services.stop(model, [model, other])
    assert calls == [("stop", model.unit)]


def test_stop_last_model_stops_gateway(store, model, monkeypatch):
    services = Services(store)
    store.units.mkdir(parents=True)
    (store.units / model.unit).touch()
    (store.units / services.proxy_unit).touch()
    calls = []
    monkeypatch.setattr(services, "active", lambda unit: False)
    monkeypatch.setattr(services, "ctl", lambda *a, **kw: calls.append(a))
    services.stop(model, [model])
    assert calls == [("stop", model.unit), ("stop", services.proxy_unit)]


def test_occupied_port_never_stops_its_owner(store, model, monkeypatch):
    services = Services(store)
    monkeypatch.setattr(services, "active", lambda unit: False)
    monkeypatch.setattr("litellm_manager.core.port_free", lambda port: False)
    monkeypatch.setattr(services, "ctl", lambda *a, **kw: pytest.fail("Must not touch services"))
    with pytest.raises(click.ClickException, match="occupied"):
        services.start(model, [model], 1)


def test_vision_batches_fit_non_causal_attention_without_splitting_images(
    store, model, tmp_path, monkeypatch
):
    server = tmp_path / "llama-server"
    server.touch()
    monkeypatch.setenv("LLAMA_SERVER", str(server))
    vision = replace(model, name="google/gemma-4-12B-it", mmproj=model.path)
    args = Services(store).backend_command(vision)
    batch = int(args[args.index("--batch-size") + 1])
    ubatch = int(args[args.index("--ubatch-size") + 1])
    # The default Gemma 4 image budget must fit in one non-causal decode.
    assert batch >= 1120
    assert ubatch >= batch


def test_remove_unregisters_and_stops_model_without_deleting_shared_weights(
    store, model, monkeypatch
):
    store.save([model])
    stopped = []
    monkeypatch.setattr(Services, "stop", lambda self, target, models: stopped.append(target.name))
    monkeypatch.setattr(Services, "refresh", lambda *args: None)
    monkeypatch.setattr(Services, "ctl", lambda *args, **kwargs: None)
    result = CliRunner().invoke(cli, ["remove", "model"])
    assert result.exit_code == 0
    assert stopped == [model.name]
    assert store.read() == []
    from pathlib import Path

    assert Path(model.path).exists()


def test_add_preserves_models_registered_during_download(store, model, monkeypatch):
    other = replace(model, name="other/model", port=8100)

    def concurrent_download(*args, **kwargs):
        store.save([other])
        return replace(model, port=8100)

    monkeypatch.setattr("litellm_manager.cli.download_model", concurrent_download)
    monkeypatch.setattr("litellm_manager.core.port_free", lambda port: True)
    monkeypatch.setattr(Services, "refresh", lambda *args: None)
    result = CliRunner().invoke(cli, ["add", "test/model"])
    assert result.exit_code == 0
    saved = store.read()
    assert {m.name for m in saved} == {model.name, other.name}
    assert len({m.port for m in saved}) == 2


def test_registry_keeps_services_beside_models(store, model, service):
    store.save([model, service])
    assert store.read() == [model, service]
    assert resolve(store.read(), "pp-structurev3") == service
    assert service.unit.startswith("litellm-service-")
    assert json.loads(store.registry.read_text())[1]["kind"] == "service"


def test_add_service_syncs_project_venv_without_downloading_weights(store, model, monkeypatch):
    calls = []
    monkeypatch.setattr("litellm_manager.core.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("litellm_manager.core.port_free", lambda port: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: calls.append(args) or subprocess.CompletedProcess(args, 0),
    )
    added = add_service("PP-StructureV3", [model], 4010)
    project = str(PROJECT / "services/pp-structure")
    assert calls == [["/usr/bin/uv", "sync", "--project", project, "--frozen", "--extra", "gpu"]]
    assert (added.name, added.project, added.route, added.port) == (
        "PaddlePaddle/PP-StructureV3",
        project,
        "/pp-structure",
        8101,
    )
    with pytest.raises(click.ClickException, match="Already added"):
        add_service("PaddlePaddle/PP-StructureV3", [added], 4010)


def test_add_service_rejects_gguf_options(store):
    result = CliRunner().invoke(cli, ["add", "PaddlePaddle/PP-StructureV3", "--context", "8192"])
    assert result.exit_code == 1
    assert "GGUF models only" in result.output


def test_service_unit_runs_project_script_on_its_port(store, service, monkeypatch):
    monkeypatch.setattr("litellm_manager.core.shutil.which", lambda name: "/usr/bin/uv")
    services = Services(store)
    args = services.command(service)
    assert args == [
        "/usr/bin/uv",
        "run",
        "--project",
        service.project,
        "--frozen",
        "--extra",
        "gpu",
        "pp-structure",
        "--host",
        "127.0.0.1",
        "--port",
        "8103",
    ]
    services.write_unit(service.unit, args, backend=True, extra_env=tuple(service.env))
    unit = (store.units / service.unit).read_text()
    assert 'Environment="PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"' in unit
    assert "Environment=CUDA_VISIBLE_DEVICES=0" in unit


def test_proxy_exposes_services_as_authenticated_pass_through(store, model, service, monkeypatch):
    runtime = store.data / "runtime/proxy/bin/litellm"
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    monkeypatch.setattr(Services, "ctl", lambda *args, **kwargs: None)
    Services(store).configure_proxy([model, service])
    config = yaml.safe_load(store.proxy_config.read_text())
    assert [row["model_name"] for row in config["model_list"]] == [model.name]
    assert config["general_settings"]["pass_through_endpoints"] == [
        {
            "path": "/pp-structure",
            "target": "http://127.0.0.1:8103",
            "include_subpath": True,
            "auth": True,
            "timeout": 300,
        }
    ]
    Services(store).configure_proxy([model])
    assert (
        "pass_through_endpoints"
        not in yaml.safe_load(store.proxy_config.read_text())["general_settings"]
    )


def test_list_shows_service_route_without_token_limits(store, model, service, monkeypatch):
    store.save([model, service])
    monkeypatch.setattr(Services, "active", lambda self, unit: True)
    monkeypatch.setattr(Services, "ready", lambda self, m: True)
    monkeypatch.setattr("litellm_manager.cli.healthy", lambda port, route: True)
    monkeypatch.setattr("litellm_manager.metadata.live_context", lambda port: None)
    result = CliRunner().invoke(cli, ["list", "--json"])
    assert result.exit_code == 0
    row = json.loads(result.output)[1]
    assert row["api"] == ["pass-through /pp-structure"]
    assert row["status"] == "on"
    assert "limits" not in row
    table = CliRunner().invoke(cli, ["list"])
    assert table.exit_code == 0
    assert "pass-through /pp-structure" in table.output
