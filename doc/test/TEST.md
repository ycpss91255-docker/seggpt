# Test Inventory

Single source of truth for all tests in this repository. Update the totals and per-file count whenever a test is added, renamed, or removed.

## Totals

| Category | Count | Location |
|---|---:|---|
| Unit (pytest) | 0 | `test/unit/` |
| Integration (pytest) | 0 | `test/integration/` |
| Smoke (pytest) | 0 | `test/smoke/` |
| Smoke (bats, docker image) | 2 | `docker/test/smoke/` |
| **Total** | **2** | |

## Smoke (bats, docker image)

Located at `docker/test/smoke/seggpt_env.bats`. Exercises the `devel` image built from `docker/Dockerfile` via the template `test` stage.

| # | Test name | Asserts |
|---|---|---|
| 1 | `entrypoint.sh is installed and executable` | `/entrypoint.sh` exists and is executable |
| 2 | `bash is available on PATH` | `bash` resolves on `PATH` |

## Unit (pytest)

(none yet — will be populated as Layer 1 / 2 / 3 source code lands)

## Integration (pytest)

(none yet)

## Smoke (pytest)

(none yet)
