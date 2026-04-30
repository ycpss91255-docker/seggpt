# seggpt

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

SegGPT visual prompt 分割 backend。給定一張 target image 與一組以上的 `(reference_image, reference_mask)` 對，透過穩定的 Python API 或 HTTP wrapper 回傳 raw mask。模型權重採用 BAAI SegGPT ViT-Large。

本 repo 只負責純推論。Mask 完整性判斷、bbox 計算、RLE 編碼、status code 都是下游 consumer 的責任。

## 架構

三層分層，由下往上包：

| Layer | 路徑 | 職責 |
|---|---|---|
| Layer 1 | `src/seggpt/runtime/` | Kernel — SegGPT model + service code（stateful `target` / `prompt` / `reset`）。內部用，不對外暴露。 |
| Layer 2 | `src/seggpt/api/` | 穩定 Python API：`SegGPTBackend.infer(target, refs, masks)` -> raw `{mask, class_id, inference_latency_ms, gpu_mem_mb}`。Stateless one-shot。下游 consumer 應該 import 的入口。 |
| Layer 3 | `src/seggpt/server/` | FastAPI HTTP wrapper，包 Layer 2 給外部 client / 互動式測試用。 |

## Quick Start

```bash
cd docker
./build.sh   # 建 devel + test image
./run.sh     # 啟動容器（掛載 model + prompts）
```

容器內走 Python API：

```python
from seggpt.api import SegGPTBackend

backend = SegGPTBackend(
    model_path="/workspace/model/seggpt_vit_large.pth",
    config_path="/workspace/config/seggpt_vit_large.yaml",
)
result = backend.infer(target_image, reference_images, reference_masks)
# {'mask': ndarray (C,H,W), 'class_id': ndarray, 'inference_latency_ms': float, 'gpu_mem_mb': float}
```

走 HTTP（Layer 3）：

```bash
curl -X POST http://localhost:8888/infer \
     -F target=@target.png \
     -F refs=@ref1.png -F refs=@ref2.png \
     -F masks=@mask1.png -F masks=@mask2.png
```

## 目錄結構

```
seggpt/
├── docker/                      Docker 環境（template subtree）
│   ├── template/                來自 ycpss91255-docker/template 的 git subtree
│   ├── Dockerfile               per-repo Dockerfile
│   ├── compose.yaml             setup.sh 自動產生（請勿手動編輯）
│   ├── setup.conf               per-repo runtime 設定
│   └── test/smoke/              docker image smoke tests（bats）
├── src/
│   └── seggpt/                  Python package（安裝為 `seggpt`）
│       ├── runtime/             Layer 1 kernel（移植自舊 SegGPT model + service）
│       ├── api/                 Layer 2 Python API（穩定契約）
│       └── server/              Layer 3 FastAPI HTTP wrapper
├── third_party/
│   └── detectron2/              detectron2 v0.6（git subtree）
├── test/
│   ├── unit/                    pytest unit tests
│   ├── integration/             pytest integration tests
│   └── smoke/                   pytest smoke tests
└── doc/
    ├── changelog/CHANGELOG.md
    ├── test/TEST.md
    └── README.{zh-TW,zh-CN,ja}.md
```

## 測試

詳細測試清單見 [TEST.md](test/TEST.md)。

## 授權

GPL-3.0。見 [LICENSE](../LICENSE)。
