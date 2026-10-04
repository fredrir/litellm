import json
import os
import re
import select
import shlex
import shutil
import signal
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which("zsh"), reason="zsh is required")
def test_real_zsh_tab_completes_commands_flags_and_catalog_models(tmp_path, monkeypatch):
    import pty

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("LITELLM_COMPLETION_OFFLINE", "1")
    cache = tmp_path / "state/litellm-manager/hub-completion"
    cache.mkdir(parents=True)
    (cache / (sha256(b"g4test").hexdigest() + ".json")).write_text(
        json.dumps(
            {"time": time.time(), "models": [{"id": "publisher/gemma-4-GGUF", "downloads": 1234}]}
        )
    )
    # Use this test's Python environment, independent of the installed user launcher.
    launcher = tmp_path / "litellm"
    launcher.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m litellm_manager.cli "$@"\n'
    )
    launcher.chmod(0o700)
    completion = Path(__file__).parents[1] / "src/litellm_manager/_litellm"
    pid, master = pty.fork()
    if pid == 0:
        os.execlp("zsh", "zsh", "-f")

    def until(pattern):
        collected = ""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                collected += os.read(master, 65536).decode(errors="replace")
                match = re.search(pattern, collected)
                if match:
                    return match
        pytest.fail(f"Completion timed out: {collected!r}")

    try:
        command = (
            f"PATH={shlex.quote(str(tmp_path))}:$PATH; autoload -Uz compinit; compinit -D; "
            f"source {shlex.quote(str(completion))}; bindkey '^I' complete-word; "
            "show-buffer() { print -r -- RESULT:$BUFFER; zle reset-prompt; }; "
            "zle -N show-buffer; bindkey '^X' show-buffer; PROMPT='READY> '\n"
        )
        os.write(master, command.encode())
        until(r"READY> .*?\x1b\[K")
        for typed, expected in [
            ("litellm sta", "litellm start "),
            ("litellm start --ti", "litellm start --timeout "),
            ("litellm add --co", "litellm add --context "),
            ("litellm add unsl", "litellm add unsloth/gemma-4-12b-it-GGUF "),
            ("litellm add goog", "litellm add google/gemma-4-12B-it "),
            ("litellm add g4test", "litellm add publisher/gemma-4-GGUF "),
        ]:
            os.write(master, b"\x15" + typed.encode() + b"\t\x18")
            assert until(r"RESULT:([^\r\n]*)\r\n").group(1) == expected
    finally:
        os.kill(pid, signal.SIGTERM)
        os.close(master)
        os.waitpid(pid, 0)
