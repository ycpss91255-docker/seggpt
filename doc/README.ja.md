# seggpt

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

SegGPT visual prompt セグメンテーション backend。target image と 1 つ以上の `(reference_image, reference_mask)` ペアを与えると、安定した Python API もしくは HTTP wrapper を介して raw mask を返す。モデル重みは BAAI SegGPT ViT-Large を使用。

このリポジトリは純粋な推論のみを担う。Mask の完全性判定、bbox 計算、RLE エンコード、status code は下流 consumer の責任。

## アーキテクチャ

3 層レイヤ、下から積み上がる:

| Layer | パス | 役割 |
|---|---|---|
| Layer 1 | `src/runtime/` | カーネル — SegGPT model + service code（stateful `target` / `prompt` / `reset`）。内部利用、外部に露出しない。 |
| Layer 2 | `src/api/` | 安定した Python API: `SegGPTBackend.infer(target, refs, masks)` -> raw `{mask, class_id, inference_latency_ms, gpu_mem_mb}`。Stateless one-shot。下流 consumer が import すべきエントリポイント。 |
| Layer 3 | `src/server/` | FastAPI HTTP wrapper、Layer 2 を包む。外部 client や対話的テスト用。 |

## Quick Start

```bash
cd docker
./build.sh   # devel + test image をビルド
./run.sh     # コンテナ起動（model + prompts をマウント）
```

コンテナ内で Python API:

```python
from seggpt.api import SegGPTBackend

backend = SegGPTBackend(
    model_path="/workspace/model/seggpt_vit_large.pth",
    config_path="/workspace/config/seggpt_vit_large.yaml",
)
result = backend.infer(target_image, reference_images, reference_masks)
# {'mask': ndarray (C,H,W), 'class_id': ndarray, 'inference_latency_ms': float, 'gpu_mem_mb': float}
```

HTTP（Layer 3）:

```bash
curl -X POST http://localhost:8888/infer \
     -F target=@target.png \
     -F refs=@ref1.png -F refs=@ref2.png \
     -F masks=@mask1.png -F masks=@mask2.png
```

## ディレクトリ構造

```
seggpt/
├── docker/                      Docker 環境（template subtree）
│   ├── template/                ycpss91255-docker/template からの git subtree
│   ├── Dockerfile               per-repo Dockerfile
│   ├── compose.yaml             setup.sh が自動生成（手動編集不可）
│   ├── setup.conf               per-repo runtime 設定
│   └── test/smoke/              docker image smoke tests（bats）
├── src/
│   ├── runtime/                 Layer 1 kernel（旧 SegGPT model + service から移植）
│   ├── api/                     Layer 2 Python API（安定契約）
│   └── server/                  Layer 3 FastAPI HTTP wrapper
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

## テスト

詳細なテストインベントリは [TEST.md](test/TEST.md) を参照。

## ライセンス

GPL-3.0。[LICENSE](../LICENSE) を参照。
