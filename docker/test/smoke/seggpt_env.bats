#!/usr/bin/env bats
#
# Repo-specific runtime smoke tests. Exercise the `devel` image built
# from this repo's Dockerfile, via the `test` stage. Use the shared
# helpers in test_helper.bash (assert_cmd_installed, assert_file_exists,
# assert_dir_exists, assert_file_owned_by, assert_pip_pkg, ...) to keep
# assertions terse. Add one assertion per meaningful installation
# artifact.

setup() {
  load "${BATS_TEST_DIRNAME}/test_helper"
}

# -------------------- baseline --------------------

@test "entrypoint.sh is installed and executable" {
  assert_file_exists /entrypoint.sh
  assert [ -x /entrypoint.sh ]
}

@test "bash is available on PATH" {
  assert_cmd_installed bash
}

# -------------------- conda + python --------------------

@test "conda is on PATH (miniconda installed)" {
  assert_cmd_installed conda
}

@test "mamba is on PATH (faster conda solver)" {
  assert_cmd_installed mamba
}

@test "python resolves to conda Python 3.11" {
  run python --version
  assert_success
  assert_line --partial "3.11"
}

@test "/opt/conda is owned by the runtime user" {
  assert_file_owned_by "${USER}" /opt/conda
}

# -------------------- pytorch ecosystem (conda) --------------------

@test "torch imports as 2.0.1 with CUDA 11.8 build" {
  run python -c "import torch; print(torch.__version__, torch.version.cuda)"
  assert_success
  assert_line --partial "2.0.1"
  assert_line --partial "11.8"
}

@test "torchvision imports as 0.15.2" {
  run python -c "import torchvision; print(torchvision.__version__)"
  assert_success
  assert_line --partial "0.15.2"
}

@test "torchaudio imports as 2.0.2" {
  run python -c "import torchaudio; print(torchaudio.__version__)"
  assert_success
  assert_line --partial "2.0.2"
}

@test "cv2 imports as 4.7.x" {
  run python -c "import cv2; print(cv2.__version__)"
  assert_success
  assert_line --partial "4.7"
}

# -------------------- SegGPT pip deps --------------------

@test "timm 0.9.7 importable" {
  run python -c "import timm; print(timm.__version__)"
  assert_success
  assert_line --partial "0.9.7"
}

@test "fairscale importable" {
  run python -c "import fairscale; print(fairscale.__version__)"
  assert_success
}

@test "fvcore importable" {
  run python -c "import fvcore"
  assert_success
}

@test "yacs importable" {
  run python -c "from yacs.config import CfgNode"
  assert_success
}

@test "fastapi installed" { assert_pip_pkg fastapi; }
@test "uvicorn installed" { assert_pip_pkg uvicorn; }
@test "classy-fastapi installed" { assert_pip_pkg classy-fastapi; }

# -------------------- detectron2 (built from upstream v0.6) --------------------

@test "detectron2 imports as v0.6" {
  run python -c "import detectron2; print(detectron2.__version__)"
  assert_success
  assert_line --partial "0.6"
}

# -------------------- CUDA build env --------------------

@test "CUDA_HOME points to /usr/local/cuda-11.8" {
  [ "${CUDA_HOME}" = "/usr/local/cuda-11.8" ]
}

@test "TORCH_CUDA_ARCH_LIST includes Ampere (8.6)" {
  echo "${TORCH_CUDA_ARCH_LIST}" | grep -q "8.6"
}
