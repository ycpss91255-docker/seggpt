# Jetson / L4T (JetPack 6) build

`docker/Dockerfile` carries two `devel-base` flavours and picks one from
BuildKit's `TARGETARCH`, so on a Jetson the normal flow just works:

```bash
cd docker
./build.sh    # arm64 → devel-base-arm64 (dustynv/l4t-pytorch:r36.4.0)
./run.sh      # setup.sh detects /etc/nv_tegra_release → runtime: nvidia
```

| flavour | base | python / torch | selected when |
|---|---|---|---|
| `amd64` (unchanged) | `nvidia/cuda:11.8.0-devel-ubuntu22.04` | conda 3.11 / torch 2.0.1 cu118 | `TARGETARCH=amd64` (CI, x86 hosts) |
| `arm64` | `dustynv/l4t-pytorch:r36.4.0` | system 3.10 / torch 2.4.0 cu126 | `TARGETARCH=arm64` (Jetson) |

Override with `--build-arg SEGGPT_FLAVOR=amd64|arm64` (`setup.conf`
`[build] arg_N = SEGGPT_FLAVOR=...`) if you ever need to force one.
The `test` stage (bats in `docker/test/smoke/`) asserts the amd64 conda
layout and is not built on Jetson (`./build.sh` builds `devel` only).

## What differs on arm64

- No conda: the base image already ships CUDA 12.6 + cuDNN 9 + torch /
  torchvision / OpenCV on the system `python3`. `PIP_USER=1` makes the
  devel stage's editable `pip3 install -e` go to the runtime user's
  `~/.local` (on amd64 it goes into the user-owned `/opt/conda`).
- `PIP_INDEX_URL` / `PIP_EXTRA_INDEX_URL` / `PIP_TRUSTED_HOST` /
  `TAR_INDEX_URL` are overridden: the dustynv base bakes in its private
  build-time mirror (`http://jetson.webredirect.org/jp6/cu126`), which does
  not resolve outside NVIDIA's network, so every `pip install` fails with
  `Name or service not known` (#11). PyPI is the index;
  `https://pypi.jetson-ai-lab.io/jp6/cu126` is the extra index for
  aarch64 + CUDA 12.6 wheels.
- detectron2 v0.6 is built from source with `CUDA_VISIBLE_DEVICES=""`
  (CPU-only `_C`): SegGPT only uses `detectron2.layers`' pure-python
  `Conv2d` / `get_norm` / `CNNBlockBase`, and a CUDA build would need nvcc
  for ~15 min when the daemon's default runtime is `nvidia`. The stage ends
  with an import smoke test (`from detectron2 import _C`,
  `from detectron2.layers import ...`).
- `TORCH_CUDA_ARCH_LIST=8.7` (Orin), `CUDA_HOME=/usr/local/cuda`.

## Model + data mounts

The committed `setup.conf` mounts only the repo (`mount_1`) and points the
`SEGGPT_*` env at `/workspace/...` — those paths are not created by this
repo; provide them per deployment with a local (uncommitted) edit of
`docker/config/docker/setup.conf`, keeping the mounts narrow and read-only
where possible:

```ini
[volumes]
mount_1 = ${WS_PATH}/src/seggpt:/home/${USER_NAME}/work
mount_2 = ${WS_PATH}/model:/workspace/model:ro
mount_3 = ${WS_PATH}/data/examples:/workspace/prompts:ro
mount_4 = ${WS_PATH}/results:/workspace/results
```

`setup.conf` sections replace the template's whole section, so repeat
`mount_1` when adding mounts. `./run.sh` regenerates `compose.yaml`.

## Verified (2026-09-17)

Jetson AGX Orin 64GB, JetPack 6.2.2 / L4T R36.5.0, docker.io 29.1.3 +
nvidia-container, MAXN + jetson_clocks:

| item | value |
|---|---|
| image | `seggpt:devel`, 26.1 GB (base 6.3 GB pulled + build) |
| SegGPT ViT-L, 1 ref + 1 mask, 448 px | ~0.9 s / image fp32, ~0.48 s with `torch.autocast(fp16)` (#12), mIoU 0.935 |
| model load (`SegGPTBackend`) | 11–12 s |
| RAM | 7.1 GB peak (container 6.4 GiB), no swap |
| power | 46 W avg / 50 W peak (module), GPU rail 33.6 W |

## Known limits

- `dustynv/l4t-pytorch:r36.4.0` is a mutable tag (not pinned by digest);
  the JetPack 6.2.x host runs an R36.4 container fine (same major).
- `fvcore`, `pyyaml`, `python-multipart`, `requests` are unpinned on
  arm64 (same as `fvcore` on amd64); pin them if you need a reproducible
  rebuild.
- Layer 3 HTTP server is still a stub in this repo (`src/seggpt/server`).
