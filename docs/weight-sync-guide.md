# 自训练权重同步使用说明（阿里云 OSS）

本文档说明**团队之间如何分发/拉取「自训练权重」**。公开预训练权重（pi0.5 / GR00T）不走本流程，请用 `scripts/download_weights.py`（经 hf-mirror / ModelScope 下载）。

---

## 1. 方案一句话

自训练权重统一存到**阿里云 OSS**（S3 兼容对象存储），通过 `scripts/sync_weights.py` 一键上传/下载

---



### 第一步：装依赖（一次性）

```bash
pip install boto3 python-dotenv
```

### 第二步：配置凭据（一次性）

```bash
# 在项目根目录执行
cp .env.example .env
```

编辑 `.env`，填入真实值：

```
OSS_ACCESS_KEY_ID=AccessKeyId
OSS_ACCESS_KEY_SECRET=AccessKeySecret
OSS_BUCKET=团队的bucket名
OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
OSS_REGION=cn-hangzhou
```

> `.env` 已被 `.gitignore` 忽略，不会提交到 git，请勿把真实密钥写进 `.env.example`。

### 第三步：用命令

```bash
# 上传某个版本的权重
python scripts/sync_weights.py push gait ppo_v1_2026-08-13

# 下载某个版本的权重
python scripts/sync_weights.py pull gait ppo_v1_2026-08-13

# 查看云端某个任务下的所有版本
python scripts/sync_weights.py list gait

# 查看云端所有任务
python scripts/sync_weights.py list
```

---

## 3. 目录约定

| 位置 | 路径 |
|------|------|
| 本地训练产物 | `outputs/<task>/<version>/` |
| 云端对象前缀 | `weights/<task>/<version>/` |

- `task`：任务名，例如 `gait`（步态）、`vla`（上半身）；
- `version`：版本名，建议 `算法_版本_日期`，例如 `ppo_v1_2026-08-13`。

本地与云端路径一一对应，`push`/`pull` 自动映射，无需手动指定远端路径。

---

## 4. 每个版本必须附带元信息卡片

每个版本目录下放一个 `README.md`，队友拉下来就能知道它是什么、怎么用：

```markdown
# ppo_v1_2026-08-13

- **任务**：步态策略
- **算法**：PPO + AMP
- **训练配置**：对应 configs/gait/ppo_v1.yaml
- **指标**：成功率 92%，平滑度 0.83
- **环境**：Isaac Lab x.x，Python 3.10
- **训练人 / 日期**：张三 / 2026-08-13
- **备注**：抗扰动测试通过，可用于联调
```

`push` 时若缺少 `README.md`，脚本会给出提示。

---

## 5. 权限管理

非直接使用主账号密钥，建议：

1. 在阿里云 RAM 控制台创建**子账号**，授予 `AliyunOSSFullAccess`（或仅限某个 bucket 的读写策略）；
2. 给每位队友发放独立的 AccessKey；
3. 需要时可随时在 RAM 里吊销某个子账号，不影响其他人。

