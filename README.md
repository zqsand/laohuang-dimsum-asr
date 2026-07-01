# 点心杯 — 粤语 ASR

队伍: [待填]

## 方法

基于 whisper-small + LoRA 微调 + 字符级 N-Gram 语言模型 Rescoring

### 技术栈
- **基座模型**: openai/whisper-small
- **微调**: LoRA (decoder self-attention Q/V, rank=8, α=16)
- **后处理**: 218 条字符映射 + 67 条短语映射
- **语言模型**: 字符级 3-gram (8839 条粤语文本训练)
- **推理**: whisper → 字符映射 → 短语替换 → LM rescoring

### 文件结构
- `src/inference.py` — 推理入口
- `src/lm_rescore.py` — LM rescoring 模块
- `data/char_mapping.json` — 字符映射 (218 条)
- `data/phrase_mapping.json` — 短语映射 (67 条)
- `lm/cantonese_charlm.pkl` — 字符级语言模型
- `checkpoints/` — LoRA 权重
- `outputs/` — 预测结果

### 本地 CER
- 基线 (whisper-small 零样本): 19.03%
- LoRA + 后处理: 18.88%
- LoRA + 后处理 + LM rescoring: 13.99% (忽略标点)

### 竞赛得分
- v1 (LoRA + 后处理): 74.22 分
- v2 (LM rescoring): 提交中
