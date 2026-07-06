#!/usr/bin/env python3
"""推理 — whisper-small + 后处理管线

管线:
1. whisper-small 零样本推理
2. 粤→普短语替换 (phrase_mapping.json, 长词优先)
3. OpenCC 繁→简转换 (全量覆盖)
"""

import json, os, time, warnings
warnings.filterwarnings('ignore')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import whisper

from opencc import OpenCC
cc_t2s = OpenCC('t2s')  # 繁体转简体

MAPPING_DIR = os.path.expanduser("~/cantonese_asr/submission/data/")
TEMPLATE_PATH = os.path.expanduser("~/cantonese_asr/updated/template.jsonl")
OUTPUT_PATH = os.path.expanduser("~/cantonese_asr/updated/result.jsonl")

def load_mapping(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def postprocess(text, phrase_map):
    """粤→普短语替换 → OpenCC繁→简"""
    for old in sorted(phrase_map.keys(), key=len, reverse=True):
        text = text.replace(old, phrase_map[old])
    text = cc_t2s.convert(text)
    return text

def main():
    with open(TEMPLATE_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"模板: {len(entries)} 条", flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    print(f"模型: whisper-small ({device})", flush=True)

    phrase_map = load_mapping(os.path.join(MAPPING_DIR, "phrase_mapping.json"))
    print(f"粤→普短语: {len(phrase_map)}条  + OpenCC繁→简(全量)", flush=True)

    audio_dir = os.path.join(os.path.dirname(TEMPLATE_PATH), "test_audio")
    results = []
    t0 = time.time()

    for idx, entry in enumerate(entries):
        ap = entry["audio_path"]
        full = os.path.join(audio_dir, os.path.basename(ap))
        result = model.transcribe(full, language="zh", task="transcribe",
                                    temperature=0.0, condition_on_previous_text=False)
        raw = result["text"].strip()
        processed = postprocess(raw, phrase_map)
        results.append({"audio_path": ap, "pred_text": processed})

        if (idx + 1) % 100 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"  [{idx+1}/{len(entries)}] {rate:.1f}条/s", flush=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - t0
    print(f"✓ 完成! {len(results)} 条, 耗时 {elapsed:.1f}s ({len(results)/elapsed:.2f}条/s)", flush=True)
    print(f"  保存: {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
