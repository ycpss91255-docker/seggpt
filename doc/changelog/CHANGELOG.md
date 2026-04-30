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

### Changed

- Restructure source layout: `src/{runtime,api,server}` -> `src/seggpt/{runtime,api,server}`, with `src/seggpt/__init__.py` declaring the top-level package. `pyproject.toml` no longer needs `package-dir` mapping.
- Upgrade `docker/template/` subtree from commit `81ed9ff` to `v0.15.0`. Affects template-internal docs / tests / `build-worker.yaml`; no behavior change for seggpt's local docker flow (the reusable workflow is referenced by `@v0.15.0` tag, not pulled from the subtree).
