# Phase 0 子流程：iron_beam_prompt × target

> Notion 同步版本：[Phase 0 子流程：iron_beam_prompt × target](https://app.notion.com/p/3526414c373e81bc9e7ec7e8ce020555)（在 `Phase 0 測試流程` 之下）
>
> 配套：`doc/phase0-runbook.md`（環境建置）/ `doc/phase0-test-flow.md`（主流程方法論）。

---

## 1. 為什麼跑這組（縮小子集 vs 完整 Phase 0）

完整 Phase 0（見 `phase0-test-flow.md` §3）需要 4 個分割對象（鐵 / 綠 / 藍擋板 + 棧板下緣），每個對象 ≥ 20 張 test set。

本子流程只先跑「**鐵色擋板**」一類：

- prompt pool：N=8（已備齊）
- target：3548 張 raw（`target/` 整包餵 driver；想跑 subset 就把 `target/` 改 symlink）

通過這組（目視合格 + latency / GPU mem 在預算內）→ 才擴大到綠 / 藍擋板 + 棧板下緣。沒過 → 調 prompt pool / 評估換 backend，不要先擴大。

> **GT 不可得**：real-photo 場景下，要對每張 target 標出**所有**可見鐵梁（讓 raw SegGPT 輸出能被公平打分）成本不切實際。本子流程**不算 mIoU**；想引用 SegGPT 在這個任務上的代表數字，以論文為準（DAVIS ~0.85、COCO-20i ~0.59）。

## 2. 資料盤點

### 2.1 資料位置（host 端）

```
coresam_ws/src/seggpt/data/                       # .gitignore'd
└── phase_0_test/
    ├── prompt -> 2_iron_beam_prompt/   # symlink（每次跑前 ln -sfn 換指向）
    ├── 1_pallet_film_prompt/           # 棧板膜 8 對 prompt + mask
    ├── 2_iron_beam_prompt/             # 鐵擋板 8 對
    ├── 3_green_beam_prompt/            # 綠擋板 8 對
    ├── 4_orange_beam_prompt/           # 橘擋板 8 對
    ├── 5_pallet_bottom_prompt/         # 棧板下緣 8 對
    └── target/                         # 3548 張 raw（targets-dir 預設指這裡）
```

`data/` 整個 `.gitignore`，理由：

1. **Backend repo 通用化**（CLAUDE.md「SegGPT Backend Repo 職責邊界」）：iron_beam / target 都是 CoreSAM 後推式貨架 application 的概念。`ycpss91255-docker/seggpt` 必須能被其他應用 reuse，不能綁死在這個應用。
2. **資料量**：3548 張 raw 圖會把 git history 灌爆。

容器內可見路徑：`~/work/data/phase_0_test/...`（整個 repo 已透過 `setup.conf` `mount_1` 掛在 `~/work`，driver 預設指向這裡）。

### 2.2 Prompt pool 切換 — `data/phase_0_test/prompt/`

driver 默認讀 `data/phase_0_test/prompt/` 內的 `prompt_01..prompt_08` 一對 image+mask。這個資料夾**設計成 symlink**，每次跑前手動換指向 5 套 prompt 之一：

```bash
# 鐵擋板
ln -sfn 2_iron_beam_prompt data/phase_0_test/prompt
python scripts/phase0_driver.py        # 寫 output/<timestamp>/

# 綠擋板
ln -sfn 3_green_beam_prompt data/phase_0_test/prompt
python scripts/phase0_driver.py        # 寫 output/<timestamp_2>/

# 橘擋板 / 棧板膜 / 棧板下緣 同理
```

每次重跑都進新 timestamped sub-dir，前次保留供並排比對。

### 2.3 iron_beam_prompt 8 張選圖理由

詳細選圖理由見 `data/phase_0_test/iron_beam_prompt/README.md`：

| 編號 | 來源 batch | 類型 | 重點 |
|---|---|---|---|
| prompt_01 | mr1205 | 純鐵 only | 高解析、邊界極銳利（baseline） |
| prompt_02 | mr1203 | 高解析 + label | 600×450、B41 大字 label、pallet 直放 |
| prompt_03 | mr1203 | 高解析 + 橘+綠 | 600×450、B45 大字 label、橘+綠完整 |
| prompt_04 | mr1204 | 純綠擋板 | 純綠 + label 偏左 |
| prompt_05 | mr1202 | 純橘擋板 | 純橘 + 鐵 |
| prompt_06 | mr1203 | 空棧板 | 鐵梁下緣 100% 暴露 |
| prompt_07 | mr1202 | 強反光 | light bar 直射 hard case |
| prompt_08 | mr1205 | 三色齊全 | 三色 + 警示燈 |

多樣性盤點：5 種彩擋組合（純鐵 / 純綠 / 純橘 / 橘+綠 / 三色）+ 高低解析度 + 多 label 位置 + 1 反光 hard case + 1 空棧板。滿足主流程 §3.1「多樣性 > 數量」。

### 2.4 Target — `data/phase_0_test/target/`

3548 張 raw 圖（`target/` 整包餵 driver；無 GT 所以全跑沒成本問題）。`target/` 也可以是 symlink，未來想跑特定 subset 時用 `ln -sfn small_target target` 即可。每次完整 sweep（3548 × N=3）約 45–60 分鐘（消費級 GPU + ViT-Large）。

### 2.5 N sweep 子集

driver 的 prompt 子集挑選邏輯（`scripts/phase0_driver.py` 內 `_N_SUBSETS`）：

| N | 索引 | 邏輯 |
|---|---|---|
| 1 | [1] | prompt_01（symlinked set 的第一張） |
| 2 | [1, 2] | 前 2 張（最小 ensemble baseline） |
| 4 | [1, 2, 3, 4] | 前 4 張 |
| 8 | [1, 2, 3, 4, 5, 6, 7, 8] | 全部 |

> 取「前 N 張」是為了配合 symlink workflow：你 curate prompt set 時自己決定 prompt_01..08 的順序，driver 不再有自己的 diversity-ordered 二次挑選。要改子集 → 編輯 `_N_SUBSETS`，理由寫在 PR description。

## 3. 執行步驟

### 3.1 進 container（host 端）

```bash
cd /home/yunchien/workspace/coreSAM_ws/coresam_ws/src/seggpt/docker
./run.sh
```

容器啟動時 entrypoint 自動跑 `pip install --no-deps -e ~/work`（PR #2 起），不需手動 install。

### 3.2 路徑 sanity check（容器內）

```bash
cd ~/work
ls -L data/phase_0_test/prompt/prompt_*.png | wc -l    # 期望 16（8 image + 8 mask；-L follow symlink）
ls data/phase_0_test/target/*.png | wc -l             # 期望 3548（或子集數量）
ls model/seggpt_vit_large.pth                          # 期望 1.48 GB
```

### 3.3 跑 driver（一行）

```bash
python scripts/phase0_driver.py
```

driver 預設（取自 `config/phase0_driver.yaml` + 內建 fallback）：

- `--prompts-dir data/phase_0_test/prompt`（symlink）
- `--targets-dir data/phase_0_test/target`
- `--n-values 1 2 4 8`
- `--mode instance`
- `--overlay-color 0,0,0` / `--overlay-alpha 0.5`
- YAML `no_gt: true`（GT 不可得，不算 mIoU）
- `--run-name <YYYY-MM-DD_HHMM>`
- 結果寫入 `output/<run-name>/`

driver 流程（model 只 load 一次）：

1. 對 3548 張 target × N ∈ {1, 2, 4, 8} 各跑一次 `SegGPTBackend.infer()` = 14192 次 inference
2. 每次寫 `output/<run-name>/N<n>/<target_stem>.png`（原圖）+ `<target_stem>_overlay.png`（target + 染色 mask）
3. 收 per-N stats（latency / GPU mem，no_gt=true 時不收 mIoU）
4. 寫 `SUMMARY.md`（latency / GPU mem 通過判讀，沒 mIoU gate）

### 3.4 visual inspection（host 端）

```bash
# host
ls output/<run-name>/N1/        # 每張 target 兩檔（原圖 + overlay）
ls output/<run-name>/N8/        # 同上
cat output/<run-name>/SUMMARY.md
```

並排比對 N=1 vs N=8 看 prompt 多樣性是否拉高了 overlay 品質：

```bash
diff output/<run-name>/N1 output/<run-name>/N8
# 或視覺工具開兩邊資料夾並排
```

## 4. 結果格式（driver 輸出）

```
output/<run-name>/
├── meta.json                                       # commit / model_path / prompts_dir / N / mode / no_gt / overlay
├── per_image_N1.csv                                # latency_ms, gpu_mem_mb, mask_positive_pixels
├── per_image_N2.csv
├── per_image_N4.csv
├── per_image_N8.csv
├── stats.json                                      # 每個 N 的 latency / GPU mem 統計
├── n_sweep.csv                                     # N=1/2/4/8 各自的 latency / GPU mem
├── N1/<target_stem>.png + <stem>_overlay.png × <n_targets>  # 原圖 + 染色 overlay
├── N2/<target_stem>.png + <stem>_overlay.png × <n_targets>
├── N4/<target_stem>.png + <stem>_overlay.png × <n_targets>
├── N8/<target_stem>.png + <stem>_overlay.png × <n_targets>
└── SUMMARY.md                                      # 自動生成的 markdown 通過判讀表
```

格式對齊主流程 `phase0-test-flow.md` §6。

## 5. 通過判讀

由 driver 自動寫進 `SUMMARY.md`（依主流程 §7）。**沒有 mIoU gate**，因為 GT 不可得；只看 latency / GPU mem + 視覺：

| 條件（N=8） | 預期 | 通過依據 |
|---|---|---|
| `latency median` ≤ 500 ms | 消費級 GPU | `stats.json N=8.latency_ms.median` |
| `peak gpu_mem` ≤ 12000 MB | RTX 3090/4080 | `stats.json N=8.gpu_mem_mb_peak` |
| visual: overlay 涵蓋鐵梁 | 目視 | `output/<run>/N8/<stem>_overlay.png` |
| visual: N=8 邊界比 N=1 乾淨 | 目視 | 並排比對 N1 vs N8 overlay |

額外 visual inspection（不是 hard gate，是 root cause 線索）：

| 觀察 | 警訊 |
|---|---|
| 邊界誤差大、漏標部分鐵梁 | prompt 多樣性不足 / 解析度太低 |
| 串到橘 / 綠擋板（彩色被當鐵）| prompt mask 漏標彩色 |
| 框住影子 / 縫隙 | mask 邊界訓練不夠嚴 |
| 全部 target 同一種失敗模式 | 場域跟 prompt 系統性失配 → 換 backend |

### 5.1 N sweep 解讀（`n_sweep.csv` + 視覺）

| 觀察 | 解讀 |
|---|---|
| N=4 / N=8 overlay 跟 N=1 看不出差別 | Feature Ensemble 沒發揮 → prompt pool 多樣性不夠 |
| N=8 邊界明顯比 N=4 乾淨 | DAVIS sweet spot 在這場域成立，留 N=8 |
| N=4 ≈ N=8（hard to tell）| 用 N=4（推論成本低，省 ~50% latency） |
| N=8 latency >> N=4 latency 但視覺差不多 | prompt pool 多樣性不夠（同質性高），補新光照 / 角度才有意義 |

## 6. 失敗應對

### 6.1 視覺都過不了（系統性失敗）

- 大量 target 全黑或大量錯框：先檢 `mask_positive_pixels` 在 csv 是否 > 0。若全 0 → prompt mask 讀進來可能是 RGBA-alpha 沒解出（已修，但 GIMP/Photoshop 新格式可能再爆）
- 大量 target 漏標一致：prompt 多樣性嚴重不夠；ln 換到一組更分散的 prompt 再跑
- 大量 target 串色一致：prompt mask 漏標彩色（確認 prompt_03 / prompt_05 / prompt_08 的 mask 沒把橘 / 綠標進去；見 `2_iron_beam_prompt/README.md`「標註提醒」）

### 6.2 Latency / GPU mem 不過

- `median > 500 ms` → 降 input resolution、評估 N=4 替代 N=8、考慮 VRP-SAM
- `peak > 12 GB` → 同上，或 model parallel

### 6.3 視覺串色嚴重

- 主因：prompt mask 對彩色擋板區域處理不一致 — 有幾張把彩色當鐵 cover 進去
- 解法：對 prompt_03 / prompt_05 / prompt_08 的 mask 重檢，確認只標白 / 銀金屬本體（見 `2_iron_beam_prompt/README.md`「標註提醒」）

## 7. 待補（不阻擋這次 sub-flow）

- **其他 3 類**：綠擋板 / 藍擋板 / 棧板下緣的 prompt set。鐵色打通再做；建立各自的 `<color>_beam_prompt/` 子目錄，跑時 `ln -sfn <set> data/phase_0_test/prompt`。
- **視覺化工具**：把同一 target 在 N=1/2/4/8 三張 overlay 並排成 grid PNG，方便人眼掃。driver 目前只存獨立檔案。
- **N sweep 自動 plot**：用 matplotlib 把 `n_sweep.csv` 畫成 latency vs N 折線圖。
- **SIM 端 GT 流**（C-Phase）：Isaac SIM 場景能自動生成 GT mask，未來 SIM Phase 0 可重啟 mIoU pass 準則。
- **3D 距離換算**：sub-flow 後 → mask × depth → 3D 是 B-Phase 0..2 的事，跟 SegGPT backend 無關（CoreSAM 介面邊界 §5.1）。

---

## 附 A：跟主 Phase 0 流程的關係

| 主流程 §X | 本 sub-flow 對應 | 狀態 |
|---|---|---|
| §3.1 prompt pool | iron_beam 8 對 | OK |
| §3.2 test set | target 3548 張（無 GT） | OK |
| §3.3 baseline 鎖定 | hmbb smoke（走主流程 `pytest test/integration/runtime/test_seggpt_backend_e2e.py`） | 走主流程 |
| §4.1 pipeline 正確性 | hmbb mIoU > 0.9（走主流程）| 走主流程 |
| §4.2 多場景目視 | driver overlay + visual inspection | OK |
| §4.3 N sweep | driver 跑 N=1/2/4/8 三輪 | OK |
| §4.4 latency / GPU mem | driver 收 per-image + per-N stats | OK |

## 附 B：mIoU 數字哪裡來

本地不算（GT 不可得）。要引用 SegGPT 在 in-context segmentation 任務的代表數字：

- **DAVIS**（video object segmentation）：mIoU ~0.85，N=8 sweet spot
- **COCO-20i**（few-shot semantic）：mIoU ~0.59
- 倉儲場域**沒有量化基準**（GT 不切實際）；A-Phase 1 prompt tuning 才是自家 quantitative bar 的來源（小 train set + 自家 GT）
