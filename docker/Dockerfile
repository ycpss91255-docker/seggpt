# SegGPT backend Dockerfile (derived from ycpss91255-docker/template).
#
# Stages:
#   sys     - User/group, locale, timezone (template default)
#   base    - CUDA system deps + miniconda + Python 3.11 +
#             pytorch ecosystem (conda) + SegGPT pip deps + detectron2 v0.6
#   devel   - Application entrypoint (interactive shell)
#   test    - Lint + smoke test (ephemeral, discarded after build)
#   runtime - Minimal runtime image (FastAPI server) - TODO when Layer 3 lands

ARG BASE_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"
ARG TEST_TOOLS_IMAGE="test-tools:local"

############################## sys ##############################
FROM ${BASE_IMAGE} AS sys

# CI's build-worker.yaml passes USER/GROUP/UID/GID, while compose.yaml
# (local flow) passes USER_NAME/USER_GROUP/USER_UID/USER_GID. Declare
# the short forms first so each long form can default to its short
# counterpart — both flows then converge on the long names below.
ARG USER="user"
ARG GROUP="user"
ARG UID=1000
ARG GID=1000
ARG USER_NAME="${USER}"
ARG USER_GROUP="${GROUP}"
ARG USER_UID="${UID}"
ARG USER_GID="${GID}"
ARG TZ="Asia/Taipei"
ARG APT_MIRROR_UBUNTU="archive.ubuntu.com"
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-x", "-euo", "pipefail", "-c"]

RUN sed -i "s@archive.ubuntu.com@${APT_MIRROR_UBUNTU}@g" /etc/apt/sources.list || true; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        locales \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    sed -i 's/^# *en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen && \
    locale-gen && \
    update-locale LANG="en_US.UTF-8" && \
    ln -snf /usr/share/zoneinfo/"${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone

ENV LC_ALL="en_US.UTF-8"
ENV LANG="en_US.UTF-8"

# Create user (handle UID/GID conflicts)
RUN if getent group "${USER_GID}" >/dev/null; then \
        groupmod -n "${USER_GROUP}" "$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        groupadd -g "${USER_GID}" "${USER_GROUP}"; \
    fi && \
    if getent passwd "${USER_UID}" >/dev/null; then \
        usermod -l "${USER_NAME}" -d "/home/${USER_NAME}" -m \
            "$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
    else \
        useradd -m -s /bin/bash -u "${USER_UID}" -g "${USER_GID}" "${USER_NAME}"; \
    fi && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

############################## base ##############################
FROM sys AS base

ARG DEBIAN_FRONTEND=noninteractive
# Same USER/USER_NAME alias dance as sys — ARGs do not flow across
# stages, so each stage that references the user has to redeclare.
ARG USER="user"
ARG GROUP="user"
ARG USER_NAME="${USER}"
ARG USER_GROUP="${GROUP}"

# CUDA / cuDNN versions pinned to match SegGPT's pytorch 2.0.1 + cu118 wheel
ARG NV_CUDNN_VERSION=8.6.0.163
ARG NV_CUDNN_PACKAGE="libcudnn8=${NV_CUDNN_VERSION}-1+cuda11.8"
ARG NV_CUDNN_PACKAGE_DEV="libcudnn8-dev=${NV_CUDNN_VERSION}-1+cuda11.8"

ENV CUDA_HOME="/usr/local/cuda-11.8"
ENV USE_CUDA=1
ENV BUILD_WITH_CUDA=1
# detectron2 builds CUDA kernels for the listed compute caps; covers
# Maxwell (5.0) through Ampere (8.6+PTX). Forward-compat via PTX.
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        sudo \
        git \
        vim \
        tmux \
        terminator \
        curl \
        wget \
        ca-certificates \
        build-essential \
        cmake \
        ffmpeg \
        libjpeg-dev \
        libpng-dev \
        ${NV_CUDNN_PACKAGE} \
        ${NV_CUDNN_PACKAGE_DEV} \
        && \
    apt-mark hold libcudnn8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Miniconda + Python 3.11 + mamba (faster solver)
ARG PYTHON_VERSION=3.11
ARG MAMBA_VERSION=1.5.0
ARG MINICONDA_VERSION=py311_23.5.2-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV MAMBA_NO_LOW_SPEED_LIMIT=0

RUN case "${TARGETPLATFORM}" in \
        "linux/arm64") MINICONDA_ARCH=aarch64 ;; \
        *)             MINICONDA_ARCH=x86_64  ;; \
    esac && \
    curl -fsSL -o /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-${MINICONDA_ARCH}.sh" && \
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    conda config --add channels conda-forge && \
    conda config --remove channels defaults && \
    conda install --quiet -y python="${PYTHON_VERSION}" mamba="${MAMBA_VERSION}" && \
    conda clean -ya

# Heavy conda packages (pytorch ecosystem + opencv).
# pytorch-cuda=11.8 forces the CUDA-enabled wheel that matches the
# nvidia/cuda:11.8.0 base image. mkl pin is workaround for
# https://github.com/pytorch/pytorch/issues/123097
RUN mamba install --quiet --yes \
        -c pytorch \
        -c nvidia \
        -c conda-forge \
        mkl=2023.2.0 \
        python="${PYTHON_VERSION}" \
        pytorch=2.0.1 \
        torchvision=0.15.2 \
        torchaudio=2.0.2 \
        pytorch-cuda=11.8 \
        opencv=4.7.0 \
        yacs=0.1.8 \
        && \
    mamba clean -ya

# SegGPT pip dependencies + detectron2 v0.6.
# detectron2 has no pre-built CUDA wheels on PyPI; install from upstream
# v0.6 tag so it compiles against the cuDNN/pytorch above. The third_party
# subtree at repo root is for source-level traceability; build pulls
# upstream by tag for reproducibility.
RUN pip install --no-cache-dir \
        timm==0.9.7 \
        fairscale==0.4.13 \
        fvcore \
        click==8.1.7 \
        classy-fastapi==0.5.1 \
        fastapi==0.110.1 \
        uvicorn==0.29.0 \
        && \
    pip install --no-cache-dir \
        'git+https://github.com/facebookresearch/detectron2.git@v0.6' && \
    pip cache purge

# Hand /opt/conda to USER so dev-time pip installs from inside the
# container don't need sudo.
RUN chown -R "${USER_NAME}":"${USER_GROUP}" "${CONDA_DIR}"

# Note: LD_LIBRARY_PATH is intentionally not set here. Conda's libs
# carry an rpath that resolves the runtime CUDA / cuDNN dependencies
# without needing the env var, and prepending /opt/conda/lib globally
# made bash pick up conda's libtinfow.so.6.4 (no version stamp) over
# the system's libtinfo.so.6.3, producing a noisy
# `no version information available` warning on every shell start.
# torch + torchvision + torchaudio + opencv all import correctly with
# the env var unset (verified: torch 2.0.1, cuda 11.8, cudnn 8700).

############################## devel ##############################
FROM base AS devel

# Repeat the alias dance from sys/base — ARGs do not flow across stages
# so each stage that references the user has to redeclare. Switching
# everything below to USER_NAME / USER_GROUP avoids the previous
# `${HOME}` substitution bug: $HOME is only populated at runtime (Docker
# reads /etc/passwd when it processes the USER instruction at run time),
# so any Dockerfile directive resolved at build time — WORKDIR, COPY
# --chown, RUN-side `${HOME}` references — saw an empty $HOME and baked
# `/work` instead of `/home/<user>/work`.
ARG USER="user"
ARG GROUP="user"
ARG USER_NAME="${USER}"
ARG USER_GROUP="${GROUP}"
ARG ENTRYPOINT_FILE="script/entrypoint.sh"
ARG CONFIG_DIR="/tmp/config"
ARG CONFIG_SRC="config"

COPY --chmod=0755 "./${ENTRYPOINT_FILE}" "/entrypoint.sh"
COPY --chown="${USER_NAME}":"${USER_GROUP}" "${CONFIG_SRC}" "${CONFIG_DIR}"

USER "${USER_NAME}"

# config/pip/setup.sh is the downstream-extensible pip hook; SegGPT
# heavy deps are pinned in base (above), so requirements.txt stays
# empty unless the user adds project-local pip extras.
RUN "${CONFIG_DIR}"/pip/setup.sh

# Setup shell, terminator, tmux. /home/${USER_NAME} is a build-time
# constant — safer than ${HOME} which is empty at build time even
# though Docker populates it at run time.
#
# Pass ${USER_NAME} / ${USER_GROUP} explicitly to terminator's setup.sh:
# the script reads $USER env when called without args, and the literal
# string "user" leaks into $USER inside this RUN (likely BuildKit
# coupling between `ARG USER="user"` and shell env). Explicit args
# bypass the env lookup.
RUN cat "${CONFIG_DIR}"/shell/bashrc >> "/home/${USER_NAME}/.bashrc" && \
    chown "${USER_NAME}":"${USER_GROUP}" "/home/${USER_NAME}/.bashrc" && \
    "${CONFIG_DIR}"/shell/terminator/setup.sh "${USER_NAME}" "${USER_GROUP}" && \
    "${CONFIG_DIR}"/shell/tmux/setup.sh && \
    sudo rm -rf "${CONFIG_DIR}"

# Build-time editable install of the seggpt package. Pulled from the
# `repo_root` named context (defined in setup.conf [additional_contexts]
# as `..`, available since template v0.16.0) — the main build context
# is `docker/`, which can't see pyproject.toml + src/ at the repo root
# without this. pip --no-deps because heavy deps already locked above.
#
# install path /home/${USER_NAME}/work matches setup.conf's mount_1
# target, so the editable install's egg-link / .pth pointers stay
# valid after the host bind-mount overlays this directory at run
# time (host repo has identical layout). Edits on the host reflect
# in the container without re-installing.
COPY --from=repo_root --chown="${USER_NAME}":"${USER_GROUP}" \
    pyproject.toml LICENSE README.md /home/${USER_NAME}/work/
COPY --from=repo_root --chown="${USER_NAME}":"${USER_GROUP}" \
    src /home/${USER_NAME}/work/src

RUN pip install --quiet --no-cache-dir --no-deps -e "/home/${USER_NAME}/work"

WORKDIR "/home/${USER_NAME}/work"

EXPOSE 22 8888

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

############################## test ##############################
# Resolves to test-tools:local (local build.sh) or ghcr.io/.../test-tools:vX.Y.Z (CI).
FROM ${TEST_TOOLS_IMAGE} AS test-tools-stage

FROM devel AS test

USER root

# Lint tools (from pre-built test-tools image; see TEST_TOOLS_IMAGE at top)
COPY --from=test-tools-stage /usr/local/bin/shellcheck /usr/local/bin/shellcheck
COPY --from=test-tools-stage /usr/local/bin/hadolint /usr/local/bin/hadolint

# Lint: ShellCheck (.sh) + Hadolint (Dockerfile)
COPY .hadolint.yaml /lint/.hadolint.yaml
COPY Dockerfile /lint/Dockerfile
COPY *.sh /lint/
# Helpers sourced by the root-level scripts. Must sit next to them so
# build.sh / run.sh / exec.sh / stop.sh / setup.sh can source _lib.sh
# (which in turn sources i18n.sh); setup.sh also sources _tui_conf.sh.
COPY template/script/docker/_lib.sh \
     template/script/docker/i18n.sh \
     template/script/docker/_tui_conf.sh \
     /lint/
RUN shellcheck -S warning /lint/*.sh
RUN cd /lint && hadolint Dockerfile

# Bats (from pre-built test-tools image; see TEST_TOOLS_IMAGE at top)
COPY --from=test-tools-stage /opt/bats /opt/bats
COPY --from=test-tools-stage /usr/lib/bats /usr/lib/bats
RUN ln -sf /opt/bats/bin/bats /usr/local/bin/bats

ENV BATS_LIB_PATH="/usr/lib/bats"

# Smoke test (shared from template + repo-specific)
COPY template/test/smoke/ /smoke_test/
COPY test/smoke/ /smoke_test/

ARG USER
USER "${USER}"

RUN bats /smoke_test/

############################## runtime (optional) ##############################
# Enabled when Layer 3 (FastAPI server) lands. Will derive from `base`
# (not `sys`) so the inference stack is available, then COPY src/ and
# launch uvicorn.
#
# FROM base AS runtime
# ARG USER="${USER_NAME}"
# COPY --chmod=0755 script/entrypoint.sh /entrypoint.sh
# COPY --chown="${USER}" src/ /workspace/src/
# COPY --chown="${USER}" pyproject.toml /workspace/
# RUN cd /workspace && pip install --no-cache-dir -e .
# USER "${USER}"
# WORKDIR /workspace
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["uvicorn", "seggpt.server:app", "--host", "0.0.0.0", "--port", "8888"]
