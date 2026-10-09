# DevFlow AI｜GitHub 研发协作多 Agent 平台

**项目概况**：面向 GitHub 仓库研发协作的多 Agent AI 平台，统一支持 Issue 分诊、PR Review、CI 排障、RAG 检索与跨 Agent 决策，并通过证据引用、审计追踪和人工确认控制外部写操作风险。

**技术栈**：Python / FastAPI / LangGraph / LangChain / SQLAlchemy / PostgreSQL / Milvus / BGE-M3 / BGE Reranker / MCP / Ragas / Next.js / TypeScript / Docker Compose

1. **研发任务质量**：建立 64 例人工金标专项集，在真实 LLM 模式下完成 14 例 Issue 七分类（准确率/宏 F1 均为 100%）、15 例 PR 审查（P1/P2 阻塞召回率 90.9%，合并建议准确率 93.3%）及 10 例 CI 诊断（根因定位准确率 100%）；相比启发式基线，PR 阻塞召回率由 36.4% 提升至 90.9%。
2. **Agent 编排与安全闭环**：基于 LangGraph 构建 ReAct 与 Plan–Execute–Replan 双层编排，支持 12 步预算、重复调用熔断、瞬时故障重试、依赖任务并行调度与 Observer 冲突检测；8 类门禁/冲突金标场景最终决策准确率 100%，3 个独立任务、每任务固定 120ms 的微基准较串行执行墙钟耗时降低 66.4%，且质量门禁结果一致。
3. **可追溯 RAG 与记忆治理**：使用本地 BGE-M3、BGE Reranker 与 Milvus + BM25 构建混合检索、证据定位和引用返回；10 条正向证据问题的 Hit@5、Top-1、MRR 均为 100%，2 条已批准记忆召回率 100%，3 条 pending/rejected/stale 负例误用率为 0%；5 题 Ragas 固定集的 Context Precision 为 0.95、Context Recall 与 Faithfulness 均为 1.00、正向 Answer Relevancy 为 0.895。
4. **能力扩展与可靠性**：基于 stdio MCP Server 封装记忆检索、写入和审核能力，实现 `SKILL.md` 注册、allowlist 权限校验与执行 trace，6 个 Skills 的注册覆盖率和入口路由准确率均为 100%；通过结构化响应契约、单次重试与 fenced JSON 解析，将 39 次专项调用的结构化 JSON 成功率由 69.2% 提升至 97.4%。

## 数字使用边界

- 上述结果来自仓库内固定合成金标集，不代表生产 GitHub 流量或真实团队分布。
- PR 干净样本误报率仍为 25%，简历中不应写“低误报”或“高精度”；应同时给出召回和误报。
- `retrieval_precision@5` 为 0.22，说明 Top-5 中仍有噪声；可以强调 Hit@5、Top-1 和 Ragas Context Precision，但不要把两者混为同一个 precision 指标。
- “处理耗时降低 X%”目前只对 3 个独立 Agent 的调度微基准成立；若要写“相较人工”，必须使用 `human_baseline_template.csv` 收集同题人工耗时和质量结果。
- 小样本 100% 指标必须带样本数，避免写成泛化到生产的绝对结论。

## 复现标识

- 专项评测集 SHA-256：`786287b7c212eaa8a7ad3a79c1153a0df80eefbce71edc196beedeb69142887f`
- 固定 Ragas 集 SHA-256：`c7891ecd9ee92270d4f9b3eac0055ea081ecbde14d02a44baba5e01544462c31`
