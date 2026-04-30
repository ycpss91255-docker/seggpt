# Test Inventory

Single source of truth for all tests in this repository. Update the totals and per-file count whenever a test is added, renamed, or removed.

## Totals

| Category | Count | Location |
|---|---:|---|
| Unit (pytest) | 0 | `test/unit/` |
| Integration (pytest) | 0 | `test/integration/` |
| Smoke (pytest) | 0 | `test/smoke/` |
| Smoke (bats, docker image) | 20 | `docker/test/smoke/` |
| **Total** | **20** | |

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

(none yet — will be populated as Layer 1 / 2 / 3 source code lands)

## Integration (pytest)

(none yet)

## Smoke (pytest)

(none yet)
