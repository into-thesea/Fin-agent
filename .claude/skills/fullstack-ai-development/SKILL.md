---
name: fullstack-ai-development
description: Use when building or extending a full-stack project with AI assistance — new project scaffold, new backend module, new frontend page, multi-file change, or tech-stack selection.
---

# Fullstack AI Development (Vibe Coding)

## Overview

核心原则: **先定技术栈 → 骨架先行 → 分层递进 → 计划审批 → 联调闭环**。
AI 是按方法论执行开发的"工程师", 不是替你瞎想的协作者 — 你给结构, 它填实现。源自 Qoder 教程 PDF, 语言/框架无关。

## When to Use

- 新建全栈项目、新增后端模块、新增前端页面
- 涉及多文件改动、需要先规划影响面
- 技术选型决策(框架/库/存储对比)
- 用 AI 生成代码但担心跑偏 / 风格不统一

**不要用**: 单文件小改(直接用 Inline Chat); 纯咨询(用 Ask, 不进这个流程)。

## Core Pattern — 三个维度

### 维度① 前后端构建
1. **骨架优先**: `backend/`+`frontend/` 分离; 先搭依赖 + 多环境配置(dev/prod 分离) + dev 代理, 再填业务。
2. **分层流水线** (每层独立生成→审查→测试): Entity → DTO → 数据访问 → Service(业务下沉, Controller 只做参数/校验) → API → 前端 api 封装 → 页面 → store → 路由 → 安全收口。
3. **前端黄金构图**: 顶部搜索栏 + 数据表格 + 分页; 新增/编辑/删除确认全走弹窗。
4. **异步三态必须齐全**: loading / 空数据 / 错误。
5. **状态分层**: UI 状态本地 state; 跨页共享进 store; 服务端数据进 React Query 缓存 — 别都塞 store。
6. **认证前后端贯穿**: 后端开放 `/auth/**`、保护 `/api/**`; 前端 token 存储 + 路由守卫 + axios 拦截器。

### 维度② 运用 AI 开发
1. **计划审批制是默认**: 多文件改动前强制 AI 输出"影响文件+依赖+风险+验证"计划, 批准再执行。
2. **提示词四原则**: 明确角色 / 充足上下文 / 指定输出格式 / 分步引导(复杂任务拆步骤渐进)。
3. **一次一层**: 分层递进优于整包生成; 每层审查 diff → 应用 → 测试。
4. **规则即代码**: 把 Git/命名/目录/技术栈/DB/测试/安全规范固化进规则文件(CLAUDE.md/AGENTS.md), 让 AI 持续遵守。
5. **模式分层**: Ask=咨询(不动代码) / Agent=委托开发(计划审批) / Quest=多阶段统筹(慎用, 消耗大)。

### 维度③ 技术栈选择
1. **先定技术栈再动工**: 开工前明确 [项目形态 + 核心功能范围 + 技术栈], 骨架围绕它搭, 不边写边选。
2. **选型用 Ask 多维对比**: 性能 / 内存 / API 易用性 / 社区活跃 / 与现有栈契合度 / 踩坑风险, 基于**具体场景**给倾向性结论。
3. **固定栈进规则文件**: 新依赖需说明理由, 不随手引入项目没有的库。

## Quick Reference

- **复制粘贴模板**: 打开 `docs/可复用经验库.md` §四 (9 个模板: 骨架/计划审批/后端模块/CRUD页/通用组件/规则文件/冒烟测试/代码审查/技术栈选型)；方法论见 §一~§三，企业级 Agent 见 §五。
- **AI 命令**: `/explain` `/optimize` `/refactor` `/generate-test` `/document`; Inline Chat 选中代码局部提问。
- **开发顺序**: 技术栈(模板9) → 计划审批(模板2) → 骨架/分层(模板3/4) → 联调(模板7)。

## Common Mistakes

| 问题 | 修法 |
|---|---|
| 没出计划就动手 | 先计划审批, 批准再执行 |
| 一次让 AI 整包生成模块 | 分层递进, 每层审查 |
| 页面缺三态 | loading / 空数据 / 错误 一个不能少 |
| 边写边选技术栈 | 开工前定死, 锁进规则文件 |
| 状态全塞 store | 页面内本地, 跨页才进 store |
| 生成不满意就重开 | 在对话里追加明确修正 |
