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
