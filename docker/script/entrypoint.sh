#!/usr/bin/env bash
#
# Container entrypoint.
#
# 1. Editable-install the seggpt package on first start so
#    `from seggpt.api import SegGPTBackend` works without the user
#    having to remember `pip install -e .` after every fresh container.
#    Mount lives at $HOME/work (compose.yaml's mount_1) — that's the
#    repo root, where pyproject.toml + src/seggpt/ live.
#
#    This step has to run at container start (not Dockerfile build)
#    because the docker build context is the docker/ subdirectory; src/
#    and pyproject.toml at the repo root are outside it and can't be
#    COPYed in. Re-running pip install -e on a working install is
#    a fast no-op so we don't gate behind a cache check.

set -euo pipefail

if [[ -f "${HOME}/work/pyproject.toml" ]]; then
    pip install --quiet --no-deps -e "${HOME}/work" || true
fi

exec "${@}"
