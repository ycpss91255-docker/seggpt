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

### Changed

- Restructure source layout: `src/{runtime,api,server}` -> `src/seggpt/{runtime,api,server}`, with `src/seggpt/__init__.py` declaring the top-level package. `pyproject.toml` no longer needs `package-dir` mapping.
- Upgrade `docker/template/` subtree from commit `81ed9ff` to `v0.15.0`. Affects template-internal docs / tests / `build-worker.yaml`; no behavior change for seggpt's local docker flow (the reusable workflow is referenced by `@v0.15.0` tag, not pulled from the subtree).
- CI: bump template reusable workflow pin to `@v0.15.0` (closes template#195) + pass `context_path: docker` (Dockerfile in subdirectory) + pin `build_runtime: false` until Layer 3 lands so the missing `runtime` stage doesn't block CI.

### Fixed

- Dockerfile USER vs USER_NAME ARG mismatch with template's CI build-args. CI passes short forms (`USER=ci`, `GROUP=ci`, `UID=1000`, `GID=1000`) while compose.yaml passes long forms (`USER_NAME` etc.). Both forms are now declared in the sys + base stages with long forms defaulting to short, so `useradd` creates the correct user under both flows. Filed as template#198 for upstream resolution.
