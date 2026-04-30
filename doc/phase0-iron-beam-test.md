# Phase 0 子流程：iron_beam_prompt × small_target

> Notion 同步版本：[Phase 0 子流程：iron_beam_prompt × small_target](https://app.notion.com/p/3526414c373e81bc9e7ec7e8ce020555)（在 `Phase 0 測試流程` 之下）
>
> 配套：`doc/phase0-runbook.md`（環境建置）/ `doc/phase0-test-flow.md`（主流程方法論）。

---

## 1. 為什麼跑這組（縮小子集 vs 完整 Phase 0）

完整 Phase 0（見 `phase0-test-flow.md` §3）需要 4 個分割對象（鐵 / 綠 / 藍擋板 + 棧板下緣），每個對象 ≥ 20 張帶 GT mask 的 test set。

本子流程只先跑「**鐵色擋板**」一類：

- prompt pool：N=8（已備齊）
- target：32 張小目標（鐵梁佔畫面比例小，邊界精度最 hard）
- GT mask：32 張（與 target 同檔名）

通過這組 → 才擴大到綠 / 藍擋板 + 棧板下緣。沒過 → 調 prompt pool / 評估換 backend，不要先擴大。

## 2. 資料盤點

### 2.1 資料位置（host 端）

```
coresam_ws/src/seggpt/data/                      # .gitignore'd
└── phase_0_test/
    ├── iron_beam_prompt/   # 8 對 prompt + mask
    ├── small_target/       # 32 張 target
    ├── ground_truth/       # 32 張 GT mask（與 small_target 同檔名）
    └── target/             # 3548 張 raw（保留作未來擴大來源）
```

`data/` 整個 `.gitignore`，理由：

1. **Backend repo 通用化**（CLAUDE.md「SegGPT Backend Repo 職責邊界」）：iron_beam / small_target / ground_truth 都是 CoreSAM 後推式貨架 application 的概念。`ycpss91255-docker/seggpt` 必須能被其他應用 reuse，不能綁死在這個應用。
2. **資料量**：3548 + 32 張 raw 圖會把 git history 灌爆。

容器內可見路徑：`~/work/data/phase_0_test/...`（整個 repo 已透過 `setup.conf.local` `mount_1` 掛在 `~/work`，driver 預設指向這裡）。

### 2.2 Prompt pool — `data/phase_0_test/iron_beam_prompt/`

8 對 image + mask（詳細選圖理由見 `data/phase_0_test/iron_beam_prompt/README.md`）：

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

### 2.3 Target / GT — `data/phase_0_test/{small_target,ground_truth}/`

32 對：`target/<name>.png` ↔ `ground_truth/<name>.png`（**同檔名**，driver 用 filename 一對一 lookup）。

選擇理由：從 `data/phase_0_test/target/` 3548 張 raw 中挑出的「小目標」case — 鐵梁佔畫面比例小，是邊界精度（≤ 3 pixel）最 hard 的場景。先打通這組再擴大其他三類。

### 2.4 N sweep 子集

driver 的 prompt 子集挑選邏輯（`scripts/phase0_driver.py` 內 `_N_SUBSETS`）：

| N | 索引 | 邏輯 |
|---|---|---|
| 1 | [1] | 純鐵 baseline（最強 single prompt） |
| 4 | [1, 4, 5, 7] | 鐵 + 純綠 + 純橘 + 強反光（多樣性導向，4 張覆蓋 4 種視覺場景） |
| 8 | [1..8] | 全部（DAVIS sweet spot） |

要改子集 → 編輯 `scripts/phase0_driver.py` 的 `_N_SUBSETS` dict，理由寫在 PR description。

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
ls data/phase_0_test/iron_beam_prompt/prompt_*.png | wc -l   # 期望 16（8 image + 8 mask）
ls data/phase_0_test/small_target/*.png | wc -l              # 期望 32
ls data/phase_0_test/ground_truth/*.png | wc -l              # 期望 32
ls model/seggpt_vit_large.pth                                # 期望 1.48 GB
```

### 3.3 跑 driver（一行）

```bash
python scripts/phase0_driver.py
```

driver 預設：

- `--prompts-dir data/phase_0_test/iron_beam_prompt`
- `--targets-dir data/phase_0_test/small_target`
- `--gt-dir     data/phase_0_test/ground_truth`
- `--n-values 1 4 8`
- `--mode instance`
- `--failure-threshold 0.5`
- `--run-name <YYYY-MM-DD_HHMM>_iron_beam_small_target`
- 結果寫入 `phase0_runs/<run-name>/`

driver 流程（model 只 load 一次）：

1. 對 32 張 target × N ∈ {1, 4, 8} 各跑一次 `SegGPTBackend.infer()` = 96 次 inference
2. 每次算 mIoU vs 對應檔名的 GT mask
3. mIoU < 0.5 自動拷貝 target / pred / gt 到 `failures/<stem>_N<n>/`
4. 收 per-N stats（mean / median / p10 / p90 / std mIoU + latency）
5. 寫 `SUMMARY.md` 含通過判讀

### 3.4 visual inspection（host 端）

```bash
# host
ls phase0_runs/<run-name>/pred_masks/   # 96 張 (32 × 3)
ls phase0_runs/<run-name>/failures/     # mIoU < 0.5 的 case
cat phase0_runs/<run-name>/SUMMARY.md
```

## 4. 結果格式（driver 輸出）

```
phase0_runs/<run-name>/
├── meta.json          # commit / model_path / n_subsets / device
├── per_image.csv      # target, N, latency_ms, gpu_mem_mb, mask_positive_pixels, miou
├── stats.json         # per N: latency_ms{median,p10,p90,max}, miou{mean,median,p10,p90,std}
├── n_sweep.csv        # N, median_latency_ms, peak_gpu_mem_mb, mean_miou, median_miou, p10_miou, p90_miou
├── pred_masks/        # <stem>_N<n>.png × 96
├── failures/          # mIoU < 0.5: target.png + pred.png + gt.png
└── SUMMARY.md         # 自動生成的 markdown 通過判讀表
```

格式對齊主流程 `phase0-test-flow.md` §6。

## 5. 通過判讀

由 driver 自動寫進 `SUMMARY.md`（依主流程 §7）：

| 條件（N=8） | 預期 | 通過依據 |
|---|---|---|
| `mean(mIoU)` ≥ 0.7 | 倉儲場域寬鬆值 | `stats.json N=8.miou.mean` |
| `p10(mIoU)` ≥ 0.5 | worst case | `stats.json N=8.miou.p10` |
| `latency median` ≤ 500 ms | 消費級 GPU | `stats.json N=8.latency_ms.median` |
| `peak gpu_mem` ≤ 12000 MB | RTX 3090/4080 | `stats.json N=8.gpu_mem_mb_peak` |

額外 visual inspection（不是 hard gate，是 root cause 線索）：

| 觀察 | 警訊 |
|---|---|
| 邊界誤差 > 3 px | prompt 多樣性不足 / 解析度太低 |
| 串到橘 / 綠擋板 | prompt mask 漏標彩色 |
| 框住影子 / 縫隙 | mask 邊界訓練不夠嚴 |

### 5.1 N sweep 解讀（`n_sweep.csv`）

| 觀察 | 解讀 |
|---|---|
| N=4 顯著優於 N=1 | Feature Ensemble 有效，多樣 prompt 在拉高 mIoU |
| N=8 顯著優於 N=4 | DAVIS sweet spot 在這場域成立，留 N=8 |
| N=4 ≈ N=8 | 用 N=4（推論成本低，省 50% latency） |
| N=8 平台 / 下降 | prompt pool 多樣性不夠（同質性高），補新光照 / 角度才有意義 |

## 6. 失敗應對

### 6.1 mIoU 不過

- `mean < 0.7` → 看 `failures/`，找模式（特定光照 / 角度集中失敗？）→ 補 prompt pool / 換 backend
- `p10 < 0.5` 但 `mean OK` → 個別 hard case，看 worst 5 張的共同特徵
- `mean < 0.7` 且 prompt pool 已多樣 → 評估換 backend（VRP-SAM / PerSAM-F / SAM 2 Tiny）

### 6.2 Latency / GPU mem 不過

- `median > 500 ms` → 降 input resolution、評估 N=4 替代 N=8、考慮 VRP-SAM
- `peak > 12 GB` → 同上，或 model parallel

### 6.3 視覺串色嚴重

- 主因：prompt mask 對彩色擋板區域處理不一致 — 有幾張把彩色當鐵 cover 進去
- 解法：對 prompt_03 / prompt_05 / prompt_08 的 mask 重檢，確認只標白 / 銀金屬本體（見 `iron_beam_prompt/README.md`「標註提醒」）

## 7. 待補（不阻擋這次 sub-flow）

- **其他 3 類**：綠擋板 / 藍擋板 / 棧板下緣的 prompt + GT mask。鐵色打通再做。
- **視覺化工具**：把 target / pred / gt 並排的 grid 自動產 PNG，方便人眼掃。driver 目前只存獨立檔案。
- **N sweep 自動 plot**：用 matplotlib 把 `n_sweep.csv` 畫成 mIoU vs N 折線圖。
- **3D 距離換算**：sub-flow 後 → mask × depth → 3D 是 B-Phase 0..2 的事，跟 SegGPT backend 無關（CoreSAM 介面邊界 §5.1）。

---

## 附 A：跟主 Phase 0 流程的關係

| 主流程 §X | 本 sub-flow 對應 | 狀態 |
|---|---|---|
| §3.1 prompt pool | iron_beam 8 對 | OK |
| §3.2 test set | small_target 32 張 + GT 32 張 | OK |
| §3.3 baseline 鎖定 | hmbb smoke（走主流程 `pytest test/integration/runtime/test_seggpt_backend_e2e.py`） | 走主流程 |
| §4.1 pipeline 正確性 | hmbb mIoU > 0.9（走主流程） | 走主流程 |
| §4.2 多場景 mIoU | driver 算 mean / median / p10 / p90 / std | OK |
| §4.3 N sweep | driver 跑 N=1/4/8 三輪 | OK |
| §4.4 latency / GPU mem | driver 收 per-image + per-N stats | OK |

## 附 B：driver --no-gt 模式（visual-only）

GT mask 還沒備齊時臨時跑：

```bash
python scripts/phase0_driver.py --no-gt
```

跳過 mIoU、failures/，只產出 pred_masks/ 和 latency stats。給 prompt pool 早期 sanity check 用，不能當通過判讀。
