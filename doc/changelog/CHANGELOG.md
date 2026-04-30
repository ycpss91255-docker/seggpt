# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial repository scaffold with three-layer architecture skeleton (Layer 1 `src/runtime/`, Layer 2 `src/api/`, Layer 3 `src/server/`).
- Docker environment via `ycpss91255-docker/template` subtree at `docker/template/`.
- detectron2 v0.6 source via git subtree at `third_party/detectron2/`.
- 4-language README (en / zh-TW / zh-CN / ja).
- pytest test scaffold under `test/{unit,integration,smoke}/`.
- Dockerfile customisation: CUDA 11.8 + cuDNN 8.6 base, miniconda + Python 3.11, pytorch 2.0.1 (cu118) ecosystem via mamba, SegGPT pip deps (timm 0.9.7, fairscale 0.4.13, fvcore, yacs, classy-fastapi, fastapi, uvicorn), detectron2 v0.6 built from upstream tag.
- Smoke tests for the Dockerfile build: 20 bats checks covering conda / pytorch / opencv / detectron2 / pip deps / CUDA env (`docker/test/smoke/seggpt_env.bats`).
- Layer 1 utility port from `generative-services-server`: `seggpt.runtime.utils.{types, naming, environment_variables, logger, tools, lazy_import}` slimmed down to the symbols the SegGPT runtime actually needs (~250 LOC vs. ~2000 upstream).
- Layer 1 service port from `generative-services-server`: `seggpt.runtime.services.{abstract_service, seggpt_model, seggpt_service, import_modules, import_self, utils}` — kernel ABC + factory registry + ViT-Large model + stateful target/prompt/reset wrapper. Verbatim except for import-path rewriting onto the new `seggpt.runtime` package.
- 74 pytest unit tests collected across `runtime/utils/` and `runtime/services/` (parametrised cases expand to ~100). Heavy-dep tests (yacs / torch) are auto-skipped on hosts via `pytest.importorskip` and execute inside the docker image.
- **Layer 2 stable Python API**: `seggpt.api.SegGPTBackend.infer(target, refs, masks, *, mode='instance', class_id=None)` — stateless one-shot wrapper around Layer 1. Returns raw `{mask, class_id, inference_latency_ms, gpu_mem_mb}`. Loads the model once at construction, optional `warmup=True` for benchmark stability.
- 11 unit tests for `SegGPTBackend` (mocked Layer 1) covering reset/target/prompt order, mode/class_id pass-through, length-mismatch validation, latency/GPU-mem telemetry, CUDA synchronise + peak-memory branch.
- Layer 2 integration test under `test/integration/runtime/test_seggpt_backend_e2e.py` — loads hmbb fixtures, runs `SegGPTBackend.infer()`, locks down the upstream `mIoU > 0.9` contract. Auto-skips when weights / CUDA missing.
- **Phase 0 CLI**: `scripts/phase0.py` — runs end-to-end inference, emits a JSON record (latency / gpu_mem / mIoU). Defaults to in-repo hmbb fixtures so a no-arg invocation smoke-tests the full pipeline; CLI flags swap in real-prompt scenes.
- `model/` directory: ships the SegGPT ViT-Large architecture YAML; `.gitignore` blocks `*.pth` / `*.bin` / `*.safetensors` so the 1.48 GB checkpoint stays out of git. `model/README.md` documents provenance + expected layout.
- `test/assets/hmbb/` fixtures (3 reference images + 3 masks) and `test/assets/expected/output_hmbb_3.png` for the regression mIoU bar.
- Entrypoint auto-runs `pip install --no-deps -e ${HOME}/work` on container start so `from seggpt.api import SegGPTBackend` works without the user having to remember the editable install. Idempotent: a working install just re-registers in ~5 sec. Workaround for template's hardcoded `context: .` (build context = `docker/`) — `src/` and `pyproject.toml` at repo root are outside, so build-time install is not currently feasible. Issue to upstream `setup.conf [build] context` will replace this with a build-time install.
- Phase 0 docs (`doc/phase0-runbook.md` + `doc/phase0-test-flow.md`) committed to git as the canonical source. Notion mirrors live under `SegGPT 評估計畫（Phase 0）`; markdown headers link to the Notion pages.

### Changed

- Restructure source layout: `src/{runtime,api,server}` -> `src/seggpt/{runtime,api,server}`, with `src/seggpt/__init__.py` declaring the top-level package. `pyproject.toml` no longer needs `package-dir` mapping.
- Upgrade `docker/template/` subtree from commit `81ed9ff` to `v0.15.0`. Affects template-internal docs / tests / `build-worker.yaml`; no behavior change for seggpt's local docker flow (the reusable workflow is referenced by `@v0.15.0` tag, not pulled from the subtree).
- CI: bump template reusable workflow pin to `@v0.15.0` (closes template#195) + pass `context_path: docker` (Dockerfile in subdirectory) + pin `build_runtime: false` until Layer 3 lands so the missing `runtime` stage doesn't block CI.

### Fixed

- Dockerfile USER vs USER_NAME ARG mismatch with template's CI build-args. CI passes short forms (`USER=ci`, `GROUP=ci`, `UID=1000`, `GID=1000`) while compose.yaml passes long forms (`USER_NAME` etc.). Both forms are now declared in the sys + base stages with long forms defaulting to short, so `useradd` creates the correct user under both flows. Filed as template#198 for upstream resolution.
- `bash: /opt/conda/lib/libtinfo.so.6: no version information available` warning on every shell start. Caused by `ENV LD_LIBRARY_PATH=/opt/conda/lib:...` shoving conda's `libtinfow.so.6.4` (no version stamp) ahead of the system's `libtinfo.so.6.3` for bash's terminal handling. Dropped the env var entirely — torch / torchvision / torchaudio / opencv / cudnn (8700) all import correctly without it because conda's binaries carry an rpath that resolves CUDA libs without LD_LIBRARY_PATH. Verified inside the rebuilt image.
- `WORKDIR ${HOME}/work` baked `/work` into the image (the empty default `${HOME}` at Dockerfile build time, before Docker populates `$HOME` at run time from `/etc/passwd`). Replace with the build-time-constant `WORKDIR /home/${USER_NAME}/work` so the shell starts where the repo is actually mounted. Same fix applied to the `${HOME}/.bashrc` references in the bashrc/terminator/tmux setup RUN.
- Workspace volume mount missing from generated `compose.yaml`. Root cause: setup.sh's `_load_setup_conf` only reads `setup.conf.local` (per-repo override) and `template/setup.conf` (template default); the per-repo `docker/setup.conf` is **not** in that lookup chain despite the comment claiming it is. Template's `setup.conf` ships `mount_1 =` (empty), so when `setup.conf.local` doesn't declare `[volumes]` the workspace mount silently vanishes. Fix by mirroring `mount_1` into `setup.conf.local`'s `[volumes]` section, anchored at `${WS_PATH}/src/seggpt` (not `${WS_PATH}` alone) so the container sees the seggpt repo directly at `~/work` and the entrypoint's `${HOME}/work/pyproject.toml` auto-install check fires.
- `terminator/setup.sh` failing with `id: 'user': no such user` during the build. The script falls back to `${USER}` env when called without args; `ARG USER="user"` (the alias-dance default) leaks into the shell's `$USER` for that RUN under BuildKit, so the literal string "user" is what the script sees. Pass `${USER_NAME}` and `${USER_GROUP}` explicitly as positional args.
- Fresh CI builds failing with `libmamba.so.2: undefined symbol: solver_ruleinfo2str, version SOLV_1.0`. Root cause: `ARG MAMBA_VERSION=1.5.*` glob — solver picks newest 1.5.x (currently 1.5.1) whose libmamba ABI requires libsolv >= 0.7.29, but mamba 1.5.1's metadata only declares `libsolv >= 0.7.22` so the base env's pkgs/main `libsolv 0.7.22` is kept instead of being upgraded to conda-forge 0.7.29. PR #1 worked because mamba 1.5.0 happened to upgrade libsolv. Pin `MAMBA_VERSION=1.5.0` to lock the known-working trio (mamba 1.5.0 / libmamba 1.5.0 / libsolv 0.7.29).
