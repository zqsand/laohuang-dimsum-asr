#!/usr/bin/env python3
"""点心杯 ASR 推理 — whisper-small + 繁→简映射 + 粤→普短语映射

策略：不做模型微调，纯后处理路线
1. whisper-small 零样本推理
2. 短语替换（粤→普，长词优先）
3. 字符映射（繁→简）

用法:
    python3 inference.py -i template.jsonl -o result.jsonl
"""

import json, os, sys, argparse, time, warnings
warnings.filterwarnings('ignore')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_mapping(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def postprocess(text, char_map, phrase_map):
    """短语替换(粤→普,长词优先) → 字符映射(繁→简)"""
    for old in sorted(phrase_map.keys(), key=len, reverse=True):
        text = text.replace(old, phrase_map[old])
    text = ''.join([char_map.get(c, c) for c in text])
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', '-i', required=True,
                        help='输入模板 (audio_path + 可选 ref_text)')
    parser.add_argument('--output_jsonl', '-o', required=True,
                        help='输出预测结果')
    parser.add_argument('--model_id', default='openai/whisper-small',
                        help='Whisper 模型 (默认 whisper-small)')
    parser.add_argument('--data_dir',
                        default=os.path.join(os.path.dirname(__file__), '..'),
                        help='映射表所在目录')
    args = parser.parse_args()

    # 加载映射表
    char_map = load_mapping(os.path.join(args.data_dir, 'data', 'char_mapping.json'))
    phrase_map = load_mapping(os.path.join(args.data_dir, 'data', 'phrase_mapping.json'))
    print(f"映射: 繁→简 {len(char_map)}条, 粤→普 {len(phrase_map)}条", flush=True)

    # 加载 Whisper
    import torch
    import whisper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(args.model_id.split('/')[-1], device=device)
    print(f"模型: {args.model_id} ({device})", flush=True)

    # 加载模板
    rows = []
    with open(args.input_jsonl) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"模板: {len(rows)} 条", flush=True)

    # 推理
    t0 = time.time()
    audio_dir = os.path.dirname(args.input_jsonl) if not rows[0].get('audio_path').startswith('/') else ''
    results = []
    for idx, row in enumerate(rows):
        ap = row['audio_path']
        full = os.path.join(audio_dir, os.path.basename(ap)) if audio_dir else os.path.basename(ap)
        if not os.path.exists(full):
            full = ap
        result = model.transcribe(full, language='zh', task='transcribe',
                                  temperature=0.0, condition_on_previous_text=False)
        processed = postprocess(result['text'].strip(), char_map, phrase_map)
        results.append({"audio_path": ap, "pred_text": processed})

        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"  [{idx+1}/{len(rows)}] {rate:.1f}条/s", flush=True)

    # 保存
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - t0
    print(f"✓ 完成! {len(rows)} 条, {elapsed:.1f}s ({len(rows)/elapsed:.2f}条/s)", flush=True)


if __name__ == '__main__':
    main()
