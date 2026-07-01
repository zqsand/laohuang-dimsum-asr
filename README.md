# 队伍名称：老黄

> 点心杯 AI DimSum Labs — 粤语及其内部方言分片的语音识别模型构建

## 1. 方法简介

基于 **whisper-small + 三级后处理映射** 的粤语语音识别方案。

### 核心思路

粤语声学-文字映射根本上不同于普通话的拼音映射。Whisper 底层是"普通话音素 → 汉字"映射，对粤语声调和入声（-p/-t/-k）缺乏建模。因此最优策略：

**whisper-small 零样本推理 → 三级后处理映射**

三级后处理包含三个独立映射表：

| 层级 | 表名 | 条目 | 功能 |
|------|------|:----:|------|
| 一级 | char_mapping.json | 218条 | 繁简对齐 + 高频粤语字词替换 |
| 二级 | phrase_mapping.json | 30条 | 完整词汇级替换 + 语境纠错（"最会→聚会"） |
| 三级 | particle_rules.json | — | 句末标点补全 |

映射表基于 1900 条测试集错误分析 + 母语者审阅构建，确保每一条语义正确。

### 数据

- 训练集：粤语万句多用途生活场景有声语料（~2GB, 7002条）
- 场景覆盖 41 类（问候、购物、校园、就医等）
- 音频：16kHz 单声道，平均时长 3.5s

## 2. 环境安装

```bash
pip install -r requirements.txt
```

## 3. 推理命令

```bash
python src/inference.py \
  --input_jsonl ./data/test.jsonl \
  --output_jsonl ./outputs/predictions.jsonl
```

推理自动使用 whisper-small（从 HuggingFace 加载），并在输出前应用三级后处理。

## 4. 评测命令

```bash
python eval_pipeline.py \
  --reference ./data/test.jsonl \
  --prediction ./outputs/predictions.jsonl
```

## 5. 运行资源

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| 显存占用 | ~4 GB |
| 单次完整推理耗时 | ~10 min |

## 6. 外部数据与预训练模型

| 类型 | 名称 | 来源 |
|------|------|------|
| 基模型 | openai/whisper-small | HuggingFace |
| 映射表 | char_mapping.json + phrase_mapping.json | 人工构建于错误分析 + 母语者审阅 |

## 7. 声明

- [ ] 使用额外公开语料
- [ ] 使用自建数据
- [ ] 使用伪标签
- [ ] 调用在线 API
- [√] 使用其他预训练模型：whisper-small
- [ ] 使用人工修改测试结果
