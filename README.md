# 点心杯 — 粤语 ASR (AI DimSum Labs)

**赛道**: 粤语及其内部方言分片的语音识别模型构建  
**基线模型**: whisper-small  
**初赛提交**: result.jsonl  
**复赛提交**: output.jsonl + predict.py + 模型权重  

## 技术方案

**核心思路：粤语音频 → 粤语汉字 → 繁→简转换 → 粤→普词汇映射**

相较于直接微调模型（过拟合风险高），采用**纯推理后处理路线**：

1. **whisper-small 零样本推理** — 不做任何微调，避免过拟合
2. **繁→简字符映射（245条）** — 将 Whisper 输出的繁体字转为简体（字形编码级，无语义风险）
3. **粤→普短语映射（44条）** — 将 Whisper 输出的粤语特有词转为普通话等价词（嘢→东西、钟意→喜欢、唔该→谢谢等）

### 文件结构
```
├── README.md              # 本文档
├── requirements.txt       # 依赖环境
├── data/
│   ├── char_mapping.json  # 繁→简映射 (245条)
│   └── phrase_mapping.json # 粤→普映射 (44条)
├── outputs/
│   └── result.jsonl       # 初赛提交结果
└── src/
    └── inference.py       # 推理脚本
```

### 评测标准
- 系统自动做 繁→简 转换 + 去标点
- 编辑距离 ≤ 2 字符即通过
- 实时排行榜更新
