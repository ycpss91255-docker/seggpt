# Test Inventory

Single source of truth for all tests in this repository. Update the totals and per-file count whenever a test is added, renamed, or removed.

## Totals

| Category | Count | Location |
|---|---:|---|
| Unit (pytest) | 87 | `test/unit/` |
| Integration (pytest) | 4 | `test/integration/` |
| Smoke (pytest) | 0 | `test/smoke/` |
| Smoke (bats, docker image) | 22 | `docker/test/smoke/` |
| **Total** | **113** | |

> Counts are `pytest --collect-only` items on a host without `yacs` / `torch`. Parametrised cases expand each function-def line into multiple test items (e.g. `test_naming.py`'s 5 functions yield 17 items). Inside the docker image two file-level skips lift (`test_tools.py`, `test_abstract_service.py`) and `test_services_utils.py`'s torch group runs, bringing the unit total to ~100.

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
| 21 | `seggpt package installed editable into the image` | `pip show seggpt` reports an Editable project location (build-time install via `repo_root` named context) |
| 22 | `from seggpt.api import SegGPTBackend resolves without runtime install` | `python -c "from seggpt.api import SegGPTBackend"` succeeds without any entrypoint-side `pip install` |

## Unit (pytest)

Located at `test/unit/`. Imported via `test/conftest.py` which adds `src/` to `sys.path` so the in-tree `seggpt` package resolves without `pip install -e .`.

| File | Tests | Coverage |
|---|---:|---|
| `runtime/utils/test_types.py` | 10 | `PathLike` alias accepts str/Path/bytes/os.PathLike; `class_property` resolves on class without instantiation; `ConfigLike` / `ListLikeInOut` round-trip through nested data |
| `runtime/utils/test_naming.py` | 18 | `to_snake_case` and `to_camel_case` over CamelCase / mixedCase / acronym / empty / idempotency cases |
| `runtime/utils/test_environment_variables.py` | 25 | `EnvironmentVariable` registry / get-set / type coercion / double-registration error; `BooleanEnvironmentVariable` truthy/falsy strings / non-bool default rejection / int-form persistence; `PathEnvironmentVariable` is_dir create-on-get / file-mode skip-create; `USE_CUDA` / `GSI_HOME` singletons |
| `runtime/utils/test_logger.py` | 7 | level wrappers route to `seggpt.runtime` logger at correct level; non-string payloads pass through |
| `runtime/utils/test_tools.py` | (auto-skip without `yacs`) | `check_path` GSI_HOME fallback / FileNotFoundError; `path_with_home` absolute / relative; `load_yaml` yacs-vs-dict round-trip; safe_load lockdown (rejects `!!python/object`) |
| `runtime/utils/test_lazy_import.py` | 4 | `LazyModuleImporter` defers `importlib.import_module` until first attribute access; subsequent access does not re-import; works against real stdlib modules (`os.path`); constructor is no-op |
| `runtime/services/test_services_utils.py` | 7 + 3 (torch skip) | `contains_var_keyword` / `get_var_keyword` over arg combinations; `torch_use_cuda` returns `cpu` / `cuda` per `USE_CUDA` env + `torch.cuda.is_available()` |
| `runtime/services/test_abstract_service.py` | (auto-skip without `yacs`) | `ServiceFactory` registration via `__init_subclass__` / extra keywords / dup-check / unsupported-type / lookup miss; singleton + iter + contains protocol; `default_config` introspects `__init__` params (kwargs excluded); signature key extraction; `PathService` round-trip via YAML |
| `api/test_backend.py` | (auto-skip without `yacs`) | `SegGPTBackend.infer()` calls reset/target/prompt in order; mode + class_id pass-through to Layer 1; refs/masks length-mismatch raises; latency > 0; gpu_mem zero on cpu; CUDA branch syncs + reads peak alloc; `service` property exposes Layer 1 |
| `scripts/test_phase0_driver.py` | 13 | `_miou` over identical / disjoint / 1/3-overlap / 0-vs-255 nonzero-foreground / both-empty (eps); `_resize_to` is identity on shape-match, doubles via nearest-neighbor (no interpolation introduced), keeps binary {0,1}; `_read_mask` recovers alpha-encoded RGBA / converts 3-channel BGR / passes single-channel grayscale through; `_N_SUBSETS` indices stay within prompt_01..08 and have no duplicates per N |

## Integration (pytest)

Located at `test/integration/`. Requires the SegGPT ViT-Large checkpoint at `model/seggpt_vit_large.pth` and a CUDA-capable GPU; auto-skips otherwise.

| File | Tests | Coverage |
|---|---:|---|
| `runtime/test_seggpt_backend_e2e.py` | 4 | hmbb fixtures end-to-end: returns expected output keys; predicted mask vs `expected/output_hmbb_3.png` mIoU > 0.9 (matches upstream's `tests/services/test_seggpt_service.py` bar); two consecutive `infer()` calls with same inputs return same mask (statelessness); refs/masks length-mismatch raises ValueError |

## Smoke (pytest)

(none yet)
