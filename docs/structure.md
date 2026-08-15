# 项目结构总览

```
2026-Embodied_AI_Applications_Competition--Power_Industry/
├── README.md                  # 项目主说明（总体架构、VLA、步态、训练基础设施）
├── .gitignore                 # Git 忽略规则
│
├── vla/                       # 上半身 VLA 模块
│   ├── gr00t/                 #   GR00T N1.7（上游仓库整包引入）
│   ├── openpi/                #   pi0.5 训练与推理（上游仓库整包引入）
│   └── realtime_vla/          #   pi0.5 实时部署栈（推理服务端 + 真机客户端，Realtime-VLA V2）
│
├── gait/                      # 下半身步态模块
│   ├── amp/                   #   AMP 对抗运动先验
│   └── ppo/                   #   PPO 步态策略
│
├── infra/                     # 共享训练基础设施
│   └── RLinf/                 #   RLinf 分布式训练框架（上游仓库整包引入）
│       └── rlinf/algorithms/  #     通用算法库（PPO / GRPO / SAC 等实现）
│
├── weights/                   # 统一权重目录（不入 git，经本地服务器 SFTP 下载，见 weights/README.md）
│   ├── openpi/checkpoints/    #   pi0.5 基座及各微调权重
│   ├── realtime_vla/          #   pi0.5 JAX→PyTorch 转换权重
│   └── RLinf-Pi0*-LIBERO-SFT/ #   RLinf 官方 Pi0/Pi0.5 LIBERO SFT 预训练权重
│
├── envs/                      # 仿真环境封装
├── controller/                # 上下半身软耦合通信层
├── data/                      # 数据采集、处理、增强
├── configs/                   # 全局配置文件
├── scripts/                   # 训练/评估/部署入口脚本
├── tools/                     # 通用工具库
└── docs/                      # 文档
    ├── structure.md           #   本文档：项目结构总览
    └── weight-sync-guide.md   #   权重同步使用说明（本地服务器 + SFTP）
```

说明：

- `vla/gr00t`、`vla/openpi`、`infra/RLinf` 均为上游开源仓库整包引入；大权重、训练日志、虚拟环境等产物不入 git，由 `.gitignore` 排除。
- 所有大体积模型权重统一放 `weights/`，从团队本地服务器经 SFTP 获取，详见 `weights/README.md` 与 `docs/weight-sync-guide.md`。