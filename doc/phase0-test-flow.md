# Phase 0 測試流程

> Notion 同步版本：[Phase 0 測試流程](https://app.notion.com/p/3526414c373e817faacac8f237fe337f)（在 `SegGPT 評估計畫（Phase 0）` 之下）
>
> 配套：`doc/phase0-runbook.md`（環境建置 / 單次 inference 怎麼跑）。
> 這份文件處理的是**測試方法論**：要比什麼、用什麼資料、判讀準則為何、結果怎麼存。

---

## 1. 目的

驗證 SegGPT (ViT-Large) 在**後推式貨架應用情景**下能不能達到「給 image + reference (image+mask)，回傳該物件 mask」的服務承諾。

具體是 4 個分割對象：
- 鐵色擋板 / 綠色擋板 / 藍色擋板（共 3 種顏色）
- 棧板下緣（位置點）

通過 Phase 0 → 才有資格進 A-Phase 1（Learnable Prompt Tuning）/ A-Phase 2（ROS 2 服務化）。

不通過 → 必須先做 prompt 多樣性調整、N 數量 sweep、或評估換 backend（VRP-SAM / PerSAM-F / SAM 2 Tiny）。

## 2. 受測對象

| 項目 | 版本 | 來源 |
|---|---|---|
| Backend | `seggpt.api.SegGPTBackend.infer()` | `src/seggpt/api/backend.py` |
| Model | SegGPT ViT-Large | `model/seggpt_vit_large.pth` (1.48 GB, BAAI release) |
| Architecture config | `img_size=[896, 448]`, `embed_dim=1024`, `depth=24`, `num_heads=16` | `model/seggpt_vit_large.yaml` |
| Container env | CUDA 11.8 + cuDNN 8.6 + pytorch 2.0.1 + detectron2 v0.6 | `docker/Dockerfile` |
| Inference mode | `instance` | Phase 0 預設；`semantic` 留 A-Phase 1 比 |

## 3. 資料準備

### 3.1 Prompt pool（reference image + mask 對）

**每個分割對象（4 個）獨立準備一個 prompt pool**：

```
prompts/
├── iron_board/         # 鐵色擋板
│   ├── ref_01.jpg + mask_01.png    # 不同光照
│   ├── ref_02.jpg + mask_02.png    # 不同角度
│   ├── ref_03.jpg + mask_03.png    # 不同距離
│   ├── ref_04.jpg + mask_04.png
│   ├── ref_05.jpg + mask_05.png
│   ├── ref_06.jpg + mask_06.png
│   ├── ref_07.jpg + mask_07.png
│   └── ref_08.jpg + mask_08.png
├── green_board/        # 同上 N=8
├── blue_board/         # 同上 N=8
└── pallet_bottom/      # 棧板下緣 N=8
```

**選圖原則**（`.claude/skills/seggpt-prompt-pool/SKILL.md` 的硬規則）：

1. **多樣性 > 數量** — 8 張同角度 prompt ≈ 2 張的效果。每組 8 張要涵蓋：光照變化（亮 / 暗 / 反光）、角度變化（正面 / 斜 / 仰）、距離變化（近 / 中 / 遠）。
2. **同類別不混色** — 鐵色擋板的 pool 不可混進綠色擋板（Feature Ensemble = 語意平均，混入會串色）
3. **長寬比警示** — SegGPT 內部 resize 到 `[896, 448]` 不保 aspect ratio。窄長型 prompt（高遠大於寬）會嚴重變形 → 挑 prompt 時先看寬高比，盡量挑接近 2:1（896/448=2:1）的構圖
4. **Mask 邊界清楚** — 不要含 occlusion / shadow ambiguous 區域

### 3.2 Test set（target 圖 + ground-truth mask）

**真實照片優先**（國泰廠 / 日月光 k7 取得，**絕對不進訓練 split**，hook `check_test_set_isolation.sh` 強制這條規則）：

```
test_set/
├── iron_board/
│   ├── target_001.jpg + gt_001.png
│   ├── target_002.jpg + gt_002.png
│   └── ... (建議 ≥ 20 張，含正常 / 邊界 / 遮擋情況)
├── green_board/   # ≥ 20
├── blue_board/    # ≥ 20
└── pallet_bottom/ # ≥ 20
```

合成圖（Isaac SIM）暫不加進 Phase 0 — 那是 C-Phase 5 的 SIM-vs-real domain gap 量化要做的事。

### 3.3 Baseline 鎖定點

- `test/assets/hmbb/` 已內建的 hmbb fixture，搭配 `expected/output_hmbb_3.png` 是**回歸基準**。任何 Phase 0 跑都先跑這個確認 pipeline 沒壞 (mIoU > 0.9)，然後才信任真實場景的數字。

## 4. 比較項目（4 個面向）

### 4.1 Pipeline 正確性（pass / fail）

```bash
python scripts/phase0.py
```

預期 `miou` ≥ 0.9。低於門檻 → **暫停其他項目**，先 bisect 哪個 commit 引入退化。

### 4.2 多場景目視判讀（每個分割對象）

> **GT 不可得**：real-photo 場景下，要對每張 target 標出所有 visible 實例（讓 raw SegGPT 輸出能被公平打分）成本不切實際。Phase 0 本地端不算 mIoU；mIoU 量化指標**引 SegGPT 論文**：DAVIS ~0.85（video 同物件分割），COCO-20i ~0.59（few-shot semantic）。本地端只做：

| 量測 | 預期 / 來源 |
|---|---|
| 目視 overlay 是否合理涵蓋目標 | 看 `output/<run>/image/<stem>_N<n>.png` |
| `mask_positive_pixels > 0`（pipeline 沒退化）| `per_image_N<n>.csv` |
| `latency` / `gpu_mem` 是否在預算內 | 見 §4.4 |
| hmbb 回歸基準 mIoU > 0.9 | §4.1（fixed test fixture，repo 內附 GT） |

**判讀**：
- overlay 視覺上吃到目標、N=8 比 N=1 邊界更乾淨 → **通過 Phase 0**，進 A-Phase 1
- overlay 全黑或範圍亂跳（多 instance 串色 / 邊界破碎）→ **prompt pool 多樣性不夠**，補不同光照 / 角度的 prompt
- 多場景連 overlay 都救不回來 → **可能要換 backend** 或進 prompt tuning

### 4.3 Feature Ensemble — N=1/2/4/8 sweep

對 prompt pool 大小 N ∈ {1, 2, 4, 8} 各跑一輪：

| N | 預期效果 |
|---|---|
| 1 | baseline，無 Feature Ensemble |
| 4 | 應該明顯優於 N=1（語意平均開始發揮）|
| 8 | DAVIS 上的甜蜜點；倉儲場域可能 N=4 就夠 |

**判讀（全靠目視，因 GT 不可得）**：
- N=4 / N=8 overlay 跟 N=1 看不出差別 → **prompt pool 多樣性不夠**（同質性高），補不同光照 / 角度的 prompt 才會有意義
- N=8 邊界明顯比 N=4 乾淨、漏標補齊 → 留在 N=8
- N=4 ≈ N=8（hard to tell）→ 用 N=4（推論成本低）

### 4.4 Latency / GPU mem 分佈

對 N=8（或上一步選定的 N）跑 ≥ 30 次 `infer()`，扔掉 warm-up（前 3 次），記錄：

| 統計量 | 預期 |
|---|---|
| `median(latency_ms)` | ≤ 500 ms（消費級 GPU + ViT-Large + N=8 + 896×448）|
| `p99(latency_ms)` | ≤ 1.5 × median |
| `peak gpu_mem_mb` | ≤ 12 GB（fits RTX 3090 / 4080 / A6000）|

**判讀**：
- `median > 500` → 可考慮 dropping N、降 input resolution、或 evaluate VRP-SAM
- `gpu_mem > 12 GB` → 同上，或上 model parallel

## 5. 執行流程

### 5.1 Pre-flight

```bash
# 1. 確認 CI 綠（image 已驗證）
gh pr checks 1 --repo ycpss91255-docker/seggpt

# 2. 本機 build image（首次 ~25 min；之後 cache hit 秒級）
cd docker && ./build.sh

# 3. 啟動 container
./run.sh
```

### 5.2 Smoke（必跑，鎖死回歸）

```bash
# 容器內
cd ~/work
pip install -e .
pytest test/integration/runtime/test_seggpt_backend_e2e.py -v
# 4 過 → pipeline OK
```

### 5.3 對每個分割對象跑 4 個面向

`scripts/phase0_driver.py` 是 multi-target / multi-N driver — 對 `<targets-dir>` 內每張 image × `<n-values>` 各跑一次 `infer()`，model 只 load 一次，輸出符合 §6 格式的 run dir。

第一個具體實例：**iron_beam_prompt × target**（鐵色擋板 + 3548 張 raw target，無 GT），完整流程見 [`doc/phase0-iron-beam-test.md`](phase0-iron-beam-test.md)。

跑法（容器內，無參即 default 讀 `data/phase_0_test/prompt`（symlink）+ `data/phase_0_test/target/`）：

```bash
python scripts/phase0_driver.py
```

要對其他分割對象跑同樣 4 個面向 → override `--prompts-dir` / `--targets-dir` / `--gt-dir` 指向對應 pool（綠擋板 / 藍擋板 / 棧板下緣，prompt pool 還沒備齊）。

## 6. 結果記錄格式

每次完整 Phase 0 run 產生 1 個目錄：

```
output/
└── 2026-05-04_1530/                                # YYYY-MM-DD_HHMM
    ├── meta.json                                   # commit / model_path / prompts_dir / N / mode / no_gt / overlay
    ├── per_image_N1.csv                            # latency_ms, gpu_mem_mb, mask_positive_pixels, [miou]
    ├── per_image_N2.csv
    ├── per_image_N4.csv
    ├── per_image_N8.csv
    ├── stats.json                                  # 每個 N 的 latency / GPU mem / [mIoU] 統計
    ├── n_sweep.csv                                 # N=1/2/4/8 各自的 latency / GPU mem / [mIoU]
    ├── image/                                      # 全部 PNG 都在這
    │   ├── <target_stem>.png                       # 原圖（每 target 一張）
    │   ├── <target_stem>_N1.png                    # 染色 overlay (N=1)
    │   ├── <target_stem>_N2.png                    # N=2
    │   ├── <target_stem>_N4.png                    # N=4
    │   └── <target_stem>_N8.png                    # N=8
    └── SUMMARY.md
```

mIoU / failures / pass-rate 由 `no_gt` 開關控制（YAML 默認 `true`）；現階段只看 latency / GPU-mem / overlay。

換 prompt set 的工作流：

```bash
# 把 active prompts 指向某組已 curate 的 prompt pool
ln -sfn iron_beam_prompts data/phase_0_test/prompt
python scripts/phase0_driver.py        # 寫 output/<timestamp>/

# 換另一組
ln -sfn pallet_lower_edge data/phase_0_test/prompt
python scripts/phase0_driver.py        # 寫 output/<timestamp_2>/，前一次保留

# 並排比對 N=8 overlay
diff output/<timestamp>/image output/<timestamp_2>/image
```

判讀完寫一句總結進 `output/<run>/SUMMARY.md`，註記：
- 通過 / 不通過
- worst case 觀察
- 下一輪要調整 prompt pool 的哪幾張 / 加哪幾類

## 7. 判讀準則總表

| 通過 Phase 0 | 條件（4 個對象都要過）| 量測方式 |
|---|---|---|
| Pipeline | 必過：hmbb mIoU > 0.9 | repo 內附 GT，在 `test/integration/` 跑 |
| 多場景目視 | overlay 視覺上吃到目標，N 增大邊界更乾淨 | 看 `output/<run>/image/<stem>_N<n>.png` |
| `mask_positive_pixels` | > 0（沒退化成全黑）| `per_image_N<n>.csv` |
| Latency median | ≤ 500 ms | `n_sweep.csv` |
| GPU mem peak | ≤ 12 GB | `n_sweep.csv` |

> mIoU 量化指標**不在本地驗**（real-photo 場景每張 visible pallet 都要標出來才公平、不切實際）。要引用 SegGPT 在這個任務上的代表數字時以論文為準（DAVIS ~0.85、COCO-20i ~0.59）。

任何一條沒過 → 列出 root cause（prompt 多樣性不足？特定場景持續 fail？資源吃太重？）→ 決定是調 pool / 換 backend / 改解析度。

## 8. 待補（暫不阻擋 Phase 0 開始）

- **Feature Ensemble 失效偵測**：當 N=8 跟 N=1 的 overlay 視覺上沒差時自動 flag「prompt pool 多樣性不足」（目前靠人眼）。
- **SIM 端 GT 流**：Isaac SIM 場景能自動生成 GT mask，未來 SIM Phase 0 可重啟 mIoU pass 準則（C-Phase）；real-photo 端永遠目視。

---

## 附：跨場域數字對齊（防止對齊錯誤）

- DAVIS（學術 video object segmentation）SegGPT 論文報 mIoU ~0.85，N=8 sweet spot
- COCO-20i（few-shot semantic）SegGPT 論文報 mIoU ~0.59
- 倉儲場域**沒有量化基準**（GT 不可得），只能視覺判斷 + 推論成本是否符合
- 如果論文數字 + 倉儲視覺結果都能說服自己，就進 A-Phase 1；否則 prompt tuning（A-Phase 1）才是真正的 quantitative bar 來源（小 train set + 自家 GT）
