# Test Inventory

Single source of truth for all tests in this repository. Update the totals and per-file count whenever a test is added, renamed, or removed.

## Totals

| Category | Count | Location |
|---|---:|---|
| Unit (pytest) | 60 | `test/unit/` |
| Integration (pytest) | 0 | `test/integration/` |
| Smoke (pytest) | 0 | `test/smoke/` |
| Smoke (bats, docker image) | 20 | `docker/test/smoke/` |
| **Total** | **80** | |

> One additional `test_tools.py` group is auto-skipped on hosts without the `yacs` package (`pytest.importorskip`); inside the docker image it runs and brings the unit total to ~74. Counts here track only what runs unconditionally on a host.

## Smoke (bats, docker image)

Located at `docker/test/smoke/seggpt_env.bats`. Exercises the `devel` image built from `docker/Dockerfile` via the template `test` stage.

| # | Test name | Asserts |
|---|---|---|
| 1 | `entrypoint.sh is installed and executable` | `/entrypoint.sh` exists and is executable |
| 2 | `bash is available on PATH` | `bash` resolves on `PATH` |
| 3 | `conda is on PATH (miniconda installed)` | `conda` resolves on `PATH` |
| 4 | `mamba is on PATH (faster conda solver)` | `mamba` resolves on `PATH` |
| 5 | `python resolves to conda Python 3.11` | `python --version` reports 3.11 |
| 6 | `/opt/conda is owned by the runtime user` | conda dir owned by container `USER` |
| 7 | `torch imports as 2.0.1 with CUDA 11.8 build` | `torch.__version__` 2.0.1, `torch.version.cuda` 11.8 |
| 8 | `torchvision imports as 0.15.2` | `torchvision.__version__` 0.15.2 |
| 9 | `torchaudio imports as 2.0.2` | `torchaudio.__version__` 2.0.2 |
| 10 | `cv2 imports as 4.7.x` | `cv2.__version__` starts with 4.7 |
| 11 | `timm 0.9.7 importable` | `timm.__version__` 0.9.7 |
| 12 | `fairscale importable` | `import fairscale` ok |
| 13 | `fvcore importable` | `import fvcore` ok |
| 14 | `yacs importable` | `from yacs.config import CfgNode` ok |
| 15 | `fastapi installed` | `pip show fastapi` ok |
| 16 | `uvicorn installed` | `pip show uvicorn` ok |
| 17 | `classy-fastapi installed` | `pip show classy-fastapi` ok |
| 18 | `detectron2 imports as v0.6` | `detectron2.__version__` 0.6 |
| 19 | `CUDA_HOME points to /usr/local/cuda-11.8` | env var equals `/usr/local/cuda-11.8` |
| 20 | `TORCH_CUDA_ARCH_LIST includes Ampere (8.6)` | env var contains `8.6` |

## Unit (pytest)

Located at `test/unit/`. Imported via `test/conftest.py` which adds `src/` to `sys.path` so the in-tree `seggpt` package resolves without `pip install -e .`.

| File | Tests | Coverage |
|---|---:|---|
| `runtime/utils/test_types.py` | 10 | `PathLike` alias accepts str/Path/bytes/os.PathLike; `class_property` resolves on class without instantiation; `ConfigLike` / `ListLikeInOut` round-trip through nested data |
| `runtime/utils/test_naming.py` | 18 | `to_snake_case` and `to_camel_case` over CamelCase / mixedCase / acronym / empty / idempotency cases |
| `runtime/utils/test_environment_variables.py` | 25 | `EnvironmentVariable` registry / get-set / type coercion / double-registration error; `BooleanEnvironmentVariable` truthy/falsy strings / non-bool default rejection / int-form persistence; `PathEnvironmentVariable` is_dir create-on-get / file-mode skip-create; `USE_CUDA` / `GSI_HOME` singletons |
| `runtime/utils/test_logger.py` | 7 | level wrappers route to `seggpt.runtime` logger at correct level; non-string payloads pass through |
| `runtime/utils/test_tools.py` | (auto-skip without `yacs`) | `check_path` GSI_HOME fallback / FileNotFoundError; `path_with_home` absolute / relative; `load_yaml` yacs-vs-dict round-trip; safe_load lockdown (rejects `!!python/object`) |

## Integration (pytest)

(none yet)

## Smoke (pytest)

(none yet)
