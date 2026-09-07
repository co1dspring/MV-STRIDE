# llm

LLM 层的**数据标注生成**模块（数据流水线 Stage 2）：
将 Level III QA 组转化为长链式思维（CoT）监督，并用真值校验，产出 MCA / 改写变体，最后清洗成训练格式。

> 注：本层属于**数据构建的一部分**（生成 CoT/MCA 标注），因此并入 `mvstride` 包。

## 流程

1. **CoT 生成** `mvstride_cot_generation`：把多轮 QA 上下文发给 LLM（vision），要求对 Level III 主问题产出 `thinking ... <answer> ... </answer>` 格式的逐步 CoT。
2. **后处理**：`cot_jsonl2swift`（格式修复）、`match_input_output`（对齐原 QA 并注入真值）、`remove_system`（去除 system 提示的消融变体）。
3. **验证** `mvstride_cot_verification`：宽松评审 LLM 对照 GT 中间 QA 与最终答案，分类为正确 / 事实不一致 / 推理不忠实 / 最终答案不一致 / 幻觉。
4. **变体**：`choice_generation`（开放 QA → A/B/C/D）、`paraphrasing`（问题改写）、`rebalance_options`（选项字母均衡）、`mca_jsonl2swift`（去元数据）。
5. **报告** `analyse_cot_verification_result`：汇总验证标签。

## 目录内容

本目录收录 Stage 2 的全部 LLM 标注脚本（自原 `api/` 目录迁移）：

`api_interface.py`（LLM 推理入口；凭据与代理通过环境变量传入）、`mvstride_cot_generation.py`、
`mvstride_cot_verification.py`、`mvstride_cvd_test.py`、`choice_generation.py`、`paraphrasing.py`、
`cot_jsonl2swift.py`、`remove_system.py`、`match_input_output.py`、`rebalance_options.py`、
`mca_jsonl2swift.py`、`analyse_cot_verification_result.py`。

> 脚本以 `python mvstride/llm/<script>.py` 运行（同目录脚本通过 `api_interface` 相互导入）。
