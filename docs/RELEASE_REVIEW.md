# 发布候选版审阅说明

这份目录用于审阅和继续完善，还没有推送到 GitHub。原项目保持不变。

## 已完成

- 重写英文 README，区分无需 API 的结构化 demo、可选 LLM/OCR 流程与历史实验。
- 添加英文技术报告、复现指南、团队署名、来源哈希与修改记录。
- 修复原 demo 的旧 schema 字段、label 路由未初始化变量，并恢复已有 arithmetic solver 的注册。
- 提供可改输入的 demo、合成图、JSON fixture 与完整消息示例。
- 从保存的逐题预测核对 easy/hard 各 30 题的结果；保留样本规模和实验限制。
- 添加 9 个本地通过的测试及 GitHub Actions 配置。
- 排除 `.env`、原 Git 历史、数据集图像、模型/环境目录和包含学号的原 PDF。

## 展示口径

可以说：团队项目实现了结构化图表推理；历史小样本 hard strict EM 从 11/30 提升到 21/30；当前提供可运行的离线 reasoning demo。

不要说：离线 demo 本身调用了多个 LLM；70% 是完整 DVQA benchmark 分数；此次重新跑出了模型成绩；所有代码由个人独立完成；已经完成任意阶段的自动修复。

前次筛选对 repair 的判断需要细化：当前 `crew.py` 中存在 reasoning-only repair，旧 README 没有更新；blackboard demo 本身没有对应循环。原始 baseline 元数据确认了 GPT-4o-mini，但 agent 历史配置记录不完整，不能把当前默认配置直接当作当时配置。

## 发布前剩余信息

1. 确认 Xinying Liu 的具体模块/实验贡献，替换团队层面的概述；目前没有编造个人分工。
2. 核实团队/课程允许公开的范围，以及源码 MIT 声明对应的许可证文件；本包未擅自给团队代码重新授权。
3. 若要宣传完整图像输入体验，在隔离环境实际安装并运行 LLM/OCR 路线。此版本只确认离线路线和历史指标计算通过。

不必等完整 LLM 路线完善后才审阅这份作品；离线 demo、证据审计与技术报告已可阅读和试跑。对外仓库描述可先用：

> Grounded chart reasoning with typed representations, routed solvers, inspectable traces, and an auditable small-sample DVQA evaluation.
