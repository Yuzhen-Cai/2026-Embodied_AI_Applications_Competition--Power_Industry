# weights/

集中存放项目所有大体积模型权重（预训练 / 转换权重）。

| 子目录 | 内容 |
|---|---|
| `openpi/checkpoints/` | pi0.5 基座及各微调权重（158GB） |
| `realtime_vla/pi05_base_converted.pkl` | pi0.5 基座权重（JAX→PyTorch 转换格式，6.4GB） |
| `RLinf-Pi05-LIBERO-SFT/` | Pi0.5 LIBERO SFT 预训练权重（7.1GB） |
| `RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT/` | Pi0 LIBERO SFT 预训练权重（6.7GB） |

## 获取方式（SFTP）

大权重文件不随 git 仓库分发（本目录内容已被 `.gitignore` 排除，仅本 README 入库），统一托管在服务器，通过 **SFTP** 连接下载。

1. 向管理员获取服务器地址与账号；
2. 将整个 `weights/` 目录同步到项目根目录：

```bash
# 方式一：scp 递归下载整个目录
scp -r <用户名>@<服务器地址>:/<服务器上weights路径>/ ./weights/

# 方式二：sftp 交互式下载
sftp <用户名>@<服务器地址>
sftp> lcd ./weights            # 本地目标目录（先自行 mkdir weights）
sftp> cd <服务器上weights路径>  # 远端权重目录
sftp> get -r openpi            # 按需下载子目录
sftp> get -r realtime_vla
sftp> get -r RLinf-Pi05-LIBERO-SFT
sftp> get -r RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT
```

3. 下载后核对大小与文件完整性（总量约 177GB），再运行训练/推理脚本。

> 自训练产出的权重不走本目录，放 `outputs/<task>/<version>/`，经本地服务器 SFTP 同步，见 `docs/weight-sync-guide.md`。
