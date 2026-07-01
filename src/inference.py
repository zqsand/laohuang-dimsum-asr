#!/usr/bin/env python3
"""点心杯 ASR 推理 — whisper-small + 三级后处理"""

import json, os, sys, argparse, time, warnings
warnings.filterwarnings('ignore')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def load_mapping(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def postprocess(text, char_map, phrase_map):
    # 一级: 字符映射
    text = ''.join([char_map.get(c, c) for c in text])
    # 二级: 短语替换
    for old in sorted(phrase_map.keys(), key=len, reverse=True):
        text = text.replace(old, phrase_map[old])
    # 三级: 句末标点
    stripped = text.strip()
    if stripped and stripped[-1] not in chr(12290)+chr(65281)+chr(65311)+chr(65292):
        text = stripped + chr(12290)
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', '-i', required=True)
    parser.add_argument('--output_jsonl', '-o', required=True)
    parser.add_argument('--model_id', default='openai/whisper-small')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    char_map = load_mapping(os.path.join(script_dir, '..', 'data', 'char_mapping.json'))
    phrase_map = load_mapping(os.path.join(script_dir, '..', 'data', 'phrase_mapping.json'))
    print(f"表1: {len(char_map)}条  表2: {len(phrase_map)}条")
    
    import whisper
    model = whisper.load_model("small")
    
    os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)
    rows = []
    with open(args.input_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    
    for i, row in enumerate(rows):
        audio_path = row.get('audio_path', '')
        if not os.path.exists(audio_path):
            row['pred_text'] = ''
            continue
        result = model.transcribe(audio_path, language='zh', temperature=0.0, condition_on_previous_text=False)
        row['pred_text'] = postprocess(result['text'].strip(), char_map, phrase_map)
    
    with open(args.output_jsonl, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"完成 {len(rows)} 条 -> {args.output_jsonl}")

if __name__ == '__main__':
    main()
