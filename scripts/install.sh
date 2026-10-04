#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${XDG_DATA_HOME:-$HOME/.local/share}/litellm-manager/runtime"
cd "$project_dir"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) is required for this RTX 5070 Ti build.'; exit 1; }
uv sync --locked
mkdir -p "$runtime_dir" "$HOME/.local/bin" "$HOME/.zfunc"
if [[ ! -x "$runtime_dir/proxy/bin/python" ]]; then
  uv venv --python 3.12 "$runtime_dir/proxy"
fi
uv pip sync --python "$runtime_dir/proxy/bin/python" "$project_dir/proxy-requirements.txt"
if [[ ! -x "$runtime_dir/build/bin/cmake" ]]; then
  uv venv --python 3.12 "$runtime_dir/build"
  uv pip install --python "$runtime_dir/build/bin/python" cmake==4.4.4 ninja==1.13.2
fi
if [[ ! -d "$runtime_dir/llama.cpp/.git" ]]; then
  git clone --depth 1 --branch v0.5.0 https://github.com/ggml-org/llama.cpp.git "$runtime_dir/llama.cpp"
fi
actual_commit="$(git -C "$runtime_dir/llama.cpp" rev-parse HEAD)"
if [[ "$actual_commit" != 7fe450e19305b828c199d602c23a8337aaa1f03b ]]; then
  echo "Unexpected llama.cpp checkout: $actual_commit; expected pinned v0.5.0 commit."
  exit 1
fi
"$runtime_dir/build/bin/cmake" -S "$runtime_dir/llama.cpp" -B "$runtime_dir/llama.cpp/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$runtime_dir/build/bin/ninja" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON -DGGML_CCACHE=OFF \
  -DCMAKE_CUDA_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= -DCMAKE_C_COMPILER_LAUNCHER= \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
"$runtime_dir/build/bin/cmake" --build "$runtime_dir/llama.cpp/build" --target llama-server -j 4
ln -sfn "$project_dir/.venv/bin/litellm" "$HOME/.local/bin/litellm"
cp "$project_dir/src/litellm_manager/_litellm" "$HOME/.zfunc/_litellm"
echo 'Installed litellm and zsh completions. Open a new shell to load completions.'
echo 'Add initial models:'
echo '  litellm add ibm-granite/granite-docling-258M'
echo '  litellm add PaddlePaddle/PaddleOCR-VL-1.6'
echo '  litellm add google/gemma-4-12B-it'
