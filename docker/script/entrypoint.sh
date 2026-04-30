#!/usr/bin/env bash
#
# Container entrypoint.
#
# The seggpt package is installed --editable into the image at build
# time (Dockerfile pulls pyproject.toml + src/ from the `repo_root`
# named context — see docker/setup.conf [additional_contexts]). The
# install path matches setup.conf's mount_1 target so the host
# bind-mount overlays the same layout at run time and edits on the
# host reflect inside the container without re-installing.
#
# This script just execs the container command. Kept as a separate
# file so the docker/template alias-dance ARGs can find it.

set -euo pipefail

exec "${@}"
