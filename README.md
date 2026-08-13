# 2026-Embodied_AI_Applications_Competition--Power_Industry

## 1. 总体控制架构设计
* **分层协同控制架构**
  **上半身：** 低频视觉-语言-动作（VLA）大模型，负责任务理解、环境感知与端到端高层操作指令生成
  **下半身：** 高频运动控制策略（Gait Policy），负责动态平衡、步态生成与抗扰动控制
* **数据流与并发机制** 采用上下身解耦/软耦合通信机制


## 2. 上半身操作控制算法（VLA）
* **基座模型复现与性能评估**
  * **pi0.5** ：


  * **GR00T N1.7** ：


* **强化学习微调策略（Post-Training）**
  * 引入 RL 算法（PPO / GRPO / SAC），针对具身操作的成功率、动作平滑度及抗干扰能力进行仿真强化训练


## 3. 步态模型
* **核心目标** 实现机器人移动过程中的高稳定性与自然步态，抵抗上半身操作带来的质心偏移与反作用力
* **核心算法组合**
  * **PPO（Proximal Policy Optimization）** 

  * **AMP（Adversarial Motion Priors）** 


## 4. 强化学习基础设施与工具链（Training Infrastructure）
* **分布式训练框架：**
  * **RLinf** 
* **优化器与算法工具库**
  * 整合 PPO、GRPO（适用于无 Critic 的大参数 VLA 微调）、SAC（高样本效率连续控制）等算法库，作为通用的训练优化工具