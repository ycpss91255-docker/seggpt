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
