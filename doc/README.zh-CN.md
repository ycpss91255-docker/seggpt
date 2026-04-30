# seggpt

**[English](../README.md)** | **[繁體中文](README.zh-TW.md)** | **[简体中文](README.zh-CN.md)** | **[日本語](README.ja.md)**

SegGPT visual prompt 分割 backend。给定一张 target image 与一组以上的 `(reference_image, reference_mask)` 对，通过稳定的 Python API 或 HTTP wrapper 返回 raw mask。模型权重采用 BAAI SegGPT ViT-Large。

本 repo 只负责纯推理。Mask 完整性判断、bbox 计算、RLE 编码、status code 都是下游 consumer 的责任。

## 架构

三层分层，由下往上包：

| Layer | 路径 | 职责 |
|---|---|---|
| Layer 1 | `src/seggpt/runtime/` | Kernel — SegGPT model + service code（stateful `target` / `prompt` / `reset`）。内部使用，不对外暴露。 |
| Layer 2 | `src/seggpt/api/` | 稳定 Python API：`SegGPTBackend.infer(target, refs, masks)` -> raw `{mask, class_id, inference_latency_ms, gpu_mem_mb}`。Stateless one-shot。下游 consumer 应该 import 的入口。 |
| Layer 3 | `src/seggpt/server/` | FastAPI HTTP wrapper，包 Layer 2 给外部 client / 交互式测试使用。 |

## Quick Start

```bash
cd docker
./build.sh   # 构建 devel + test image
./run.sh     # 启动容器（挂载 model + prompts）
```

容器内走 Python API：

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

## 目录结构

```
seggpt/
├── docker/                      Docker 环境（template subtree）
│   ├── template/                来自 ycpss91255-docker/template 的 git subtree
│   ├── Dockerfile               per-repo Dockerfile
│   ├── compose.yaml             setup.sh 自动生成（请勿手动编辑）
│   ├── setup.conf               per-repo runtime 配置
│   └── test/smoke/              docker image smoke tests（bats）
├── src/
│   └── seggpt/                  Python package（安装为 `seggpt`）
│       ├── runtime/             Layer 1 kernel（移植自旧 SegGPT model + service）
│       ├── api/                 Layer 2 Python API（稳定契约）
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

## 测试

详细测试清单见 [TEST.md](test/TEST.md)。

## 授权

GPL-3.0。见 [LICENSE](../LICENSE)。
