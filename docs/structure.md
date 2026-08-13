# 项目结构总览

```
2026-Embodied_AI_Applications_Competition--Power_Industry/
├── README.md                  # 项目主说明（总体架构、VLA、步态、训练基础设施）
├── .gitignore                 # Git 忽略规则
├── .env.example               # 阿里云 OSS 凭据模板（复制为 .env 填入真实值）
│
├── vla/                       # 上半身 VLA 模块
│   └── base_models/           #   基座模型复现
│       ├── gr00t/             #     GR00T N1.7
│       └── openpi/            #     pi0.5
│
├── gait/                      # 下半身步态模块
│   ├── amp/                   #   AMP 对抗运动先验
│   └── ppo/                   #   PPO 步态策略
│
├── infra/                     # 共享训练基础设施
│   └── rlinf/                 #   RLinf 分布式训练框架封装
│       └── algorithms/        #     通用算法库（PPO / GRPO / SAC）
│
├── envs/                      # 仿真环境封装
├── controller/                # 上下半身软耦合通信层
├── data/                      # 数据采集、处理、增强
├── configs/                   # 全局配置文件
├── scripts/                   # 训练/评估/部署入口脚本
│   └── sync_weights.py        #   自训练权重同步工具（OSS push/pull/list）
├── tools/                     # 通用工具库
└── docs/                      # 文档
    ├── structure.md           #   本文档：项目结构总览
    └── weight-sync-guide.md   #   自训练权重同步使用说明
```
