# Phase 0 Runbook（環境建置 + 單次 inference）

> Notion 同步版本：[Phase 0 Runbook](https://app.notion.com/p/3526414c373e811b965dfff2965d741f)（在 `SegGPT 評估計畫（Phase 0）` 之下）

在這條 branch 上、從零開始，今天就跑一次 SegGPT end-to-end 推論的步驟。

---

## Step 1 — Build dev image

第一次 build 要拉 CUDA base + miniconda + pytorch + 從原始碼編 detectron2，預計 **20–30 min**。之後 build 都 hit BuildKit cache，秒級完成。

```bash
cd /home/yunchien/workspace/coreSAM_ws/coresam_ws/src/seggpt/docker
./build.sh
```

Build 完想再驗一次：

```bash
./build.sh --check        # 快速重驗，hits cache
docker images | grep seggpt
```

## Step 2 — 確認權重就位

1.48 GB 的 checkpoint 放在 `model/seggpt_vit_large.pth`（從舊 repo hardlink，沒 commit 進 git）。Sanity check：

```bash
ls -la /home/yunchien/workspace/coreSAM_ws/coresam_ws/src/seggpt/model/seggpt_vit_large.pth
# expect ~1.48 GB
```

如果不見了（例如另一台機器 fresh clone），看 `model/README.md` 裡的 BAAI 下載連結。

## Step 3 — 啟動容器

```bash
cd /home/yunchien/workspace/coreSAM_ws/coresam_ws/src/seggpt/docker
./run.sh
```

會把你丟進 `/home/<user>/work` 的 bash shell — 這個路徑就是 host 上 repo 根目錄 RW 掛進來的。1.48 GB checkpoint 容器內可見於 `/home/<user>/work/model/seggpt_vit_large.pth`。

## Step 4 — 安裝 `seggpt` 套件（editable）

容器內：

```bash
cd /home/$(whoami)/work
pip install -e .
```

把 `seggpt.runtime`、`seggpt.api`、`seggpt.server` 接進 Python import path，這樣 `from seggpt.api import SegGPTBackend` 才能正確 resolve。Editable install 表示之後改程式碼不用重裝。

## Step 5 — 用內建的 hmbb fixtures 跑 smoke

確認整個 pipeline 對舊專案的回歸基準（`mIoU > 0.9` vs `output_hmbb_3.png`）沒有壞。

```bash
python scripts/phase0.py
```

預期輸出（JSON 進 stdout）：

```json
{
  "target": ".../test/assets/hmbb/hmbb_3.jpg",
  "refs": [".../hmbb_1.jpg", ".../hmbb_2.jpg"],
  "masks": [".../hmbb_1_target.png", ".../hmbb_2_target.png"],
  "mode": "instance",
  "model_load_ms": <number>,
  "inference_latency_ms": <number>,
  "gpu_mem_mb": <number>,
  "mask_shape": [1, H, W],
  "mask_positive_pixels": <number>,
  "class_id": [0],
  "miou": 0.9X
}
```

`miou` 低於 0.9 → port 退化了；任何 import 錯誤（`ModuleNotFoundError`）→ Step 4 沒做或 docker image 是舊的。

## Step 6 — 換成你自己的 Phase 0 prompts

你的 prompt 場景放在例如 `prompts/<scene>/`：

```
prompts/<scene>/
├── target.png       # 要分割的圖
├── ref_1.jpg        # 參考圖 1
├── ref_2.jpg        # 參考圖 2
├── mask_1.png       # ref_1 的物件 mask
├── mask_2.png       # ref_2 的物件 mask
└── gt.png           # （選填）期望輸出 mask，用來算 mIoU
```

然後：

```bash
python scripts/phase0.py \
    --target   prompts/<scene>/target.png \
    --refs     prompts/<scene>/ref_1.jpg  prompts/<scene>/ref_2.jpg \
    --masks    prompts/<scene>/mask_1.png prompts/<scene>/mask_2.png \
    --expected prompts/<scene>/gt.png \
    --save-mask out/<scene>/pred.png
```

`--expected` 選填。沒給的話只報 latency / GPU mem / mask shape，跳過 mIoU 那一行。`--save-mask` 把預測 mask 寫成 0/255 PNG，可以跟 input 並排看。

CLI 還有：

- `--mode {instance,semantic}` — 預設 `instance`。Semantic 把所有 class 群在一起；instance 各自獨立。
- `--warmup` — 計時前先跑一次 dummy forward 把 CUDA kernel JIT 起來。比較不同設定的 latency 時用。
- `--model PATH` / `--config PATH` — override checkpoint / 架構 YAML。

## Step 7 — 跑完整 integration test

要嚴格 gate（4 條檢查，含 mIoU > 0.9 鎖死跟連續 `infer()` 的 statelessness）：

```bash
pytest test/integration/runtime/test_seggpt_backend_e2e.py -v
```

沒 GPU 自動 skip（例如 CPU-only host）。

---

## Troubleshooting

| 症狀 | 可能原因 | 解法 |
|---|---|---|
| `RuntimeError: CUDA out of memory` on `infer()` | 圖太大 ViT-Large 吃不下 | 縮 input、`target` / `refs` 長邊 ≤ 1024 px |
| `mIoU < 0.9` on hmbb smoke | port 改壞了某個 forward 細節 | 對 `git log src/seggpt/runtime/services/seggpt_*.py` bisect |
| `ModuleNotFoundError: seggpt.api` | Step 4 沒做 | 容器內 `pip install -e .` |
| `FileNotFoundError: ...seggpt_vit_large.pth` | 權重沒在 host | 從舊 repo 重新 hardlink，或從 BAAI 重新下載 |
| `unable to find user ci: no matching entries in passwd file`（CI only） | template Dockerfile USER vs USER_NAME 命名不一致 — 已用 alias commit `30db9bd` workaround，upstream 還在 template#198 | 確認 `30db9bd` 在這條 branch（`git log --oneline 30db9bd`）|

## Phase 0 跑完之後

- Layer 3（`src/seggpt/server/`）：FastAPI HTTP wrapper、`POST /infer`。讓 runtime 常駐在 container，CoreSAM / curl client 透過網路拉預測。
- CoreSAM 的 ROS 2 backend `import seggpt.api.SegGPTBackend`（已支援），自己再做 `has_mask` / `mask_rle` / bbox / status_code 那層 wrap（按 CLAUDE.md SegGPT Backend Repo 邊界，那些不在這個 repo 做）。
- 4 個 P0 安全債（path traversal / upload size / file lock / yaml injection）跟著 Layer 3 從 `app/` port 過來的 PR 一次修。
