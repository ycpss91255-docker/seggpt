# `model/` — SegGPT checkpoints + architecture config

Binary weights (`*.pth`) are **not** committed to git (`.gitignore` blocks them). Distribute via the BAAI SegGPT release page or an internal NAS mirror; place the file at the path below before running inference.

## Expected layout

| File | Source | Purpose |
|---|---|---|
| `seggpt_vit_large.pth` | [BAAI / SegGPT release](https://huggingface.co/BAAI/SegGPT) | ViT-Large checkpoint, ~1.48 GB. Loaded by `SegGPTService(checkpoint_path=...)` (Layer 1). |
| `seggpt_vit_large.yaml` | committed to git | Model architecture config. Loaded by `SegGPTService(config_path=...)` (Layer 1). |

## Inside the docker container

The repo root mounts to `/home/<user>/work` (see `docker/compose.yaml`'s `mount_1`). So inside the container:

```
/home/<user>/work/model/seggpt_vit_large.pth
/home/<user>/work/model/seggpt_vit_large.yaml
```

Use those paths when constructing `SegGPTService` directly, or set the `SEGGPT_MODEL_PATH` env var consumed by the (future) Layer 2 wrapper.

## Provenance — original placement

When migrating from the legacy `generative-services-server`, the same checkpoint lived at `<old-repo>/model/seggpt_vit_large.pth` and the architecture yaml at `<old-repo>/config/model/seggpt_vit_large.yaml`. The new repo flattens both into `model/` to keep weight + config side by side.
