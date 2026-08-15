# weights/

集中存放项目所有大体积模型权重（预训练 / 转换权重）。

| 子目录 | 内容 |
|---|---|
| `openpi/checkpoints/` | pi0.5 基座及各微调权重（158GB） |
| `realtime_vla/pi05_base_converted.pkl` | pi0.5 基座权重（JAX→PyTorch 转换格式，6.4GB） |
| `RLinf-Pi05-LIBERO-SFT/` | Pi0.5 LIBERO SFT 预训练权重（7.1GB） |
| `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT/` | Pi0 LIBERO SFT 预训练权重（6.7GB） |

本目录内容被 `.gitignore` 排除，不入库（仅本 README 入库）；权重通过下载或 OSS 同步（见 `docs/weight-sync-guide.md`）获取。