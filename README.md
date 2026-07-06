### 初赛提交: result.jsonl (DeepSeek LLM纠错版)

最新提交 (2026-07-06):
- Whisper-small 零样本推理
- 粤→普短语映射 50条
- OpenCC 繁→简转换
- DeepSeek V4 Flash LLM纠错 (752/1035条修正)

仓库结构:
├── data/phrase_mapping.json  — 粤→普映射50条
├── src/inference.py          — Whisper推理脚本
├── src/llm_correct.py        — LLM批量纠错脚本
├── outputs/result.jsonl      — 提交结果
└── requirements.txt
