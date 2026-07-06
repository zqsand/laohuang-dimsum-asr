#!/usr/bin/env python3
"""批量LLM纠错 — 用DeepSeek V4 Flash纠正Whisper输出"""
import json, time, requests, os, sys

API_KEY = "sk-e1f8c3ec7a7e4c5390f70940a487b6b6"
API_URL = "https://api.deepseek.com/chat/completions"
INPUT = os.path.expanduser("~/cantonese_asr/updated/result.jsonl")
OUTPUT = os.path.expanduser("~/cantonese_asr/updated/result_llm.jsonl")

SYSTEM_PROMPT = """你是粤语ASR输出的纠错专家。用户给你的是一段自动语音识别输出的文本，可能有以下问题：
1. 粤语词汇（嘢、嘅、唔、系、咗等）需要转为标准普通话
2. 繁体字需要转为简体字
3. 听错的字需要修正
4. 多余的标点需要清理

输出要求：
- 输出标准简体普通话版本
- 只输出修正后的文本，不要解释
- 不要修改标点外的字符顺序
- 保持原句的结构

示例：
输入: 打工制最大的梦想就是没有表现照出粮
输出: 打工制最大的梦想就是没有加班能吃饱
"""

def fix_text(text):
    """用DeepSeek纠正一条ASR输出"""
    for attempt in range(2):
        try:
            resp = requests.post(API_URL, json={
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text}
                ],
                'max_tokens': 150,
                'temperature': 0,
            }, headers={'Authorization': f'Bearer {API_KEY}'}, timeout=15)
            
            if resp.status_code == 200:
                result = resp.json()
                corrected = result['choices'][0]['message']['content'].strip()
                if corrected:
                    return corrected
            
            time.sleep(1)
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
            else:
                return text
    return text

def main():
    # 加载现有结果
    with open(INPUT) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    
    print(f"共 {len(entries)} 条", flush=True)
    
    results = []
    t0 = time.time()
    errors = 0
    
    for idx, entry in enumerate(entries):
        original = entry['pred_text']
        corrected = fix_text(original)
        results.append({"audio_path": entry['audio_path'], "pred_text": corrected})
        
        if original != corrected:
            errors += 1
        
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(entries) - idx - 1) / rate
            print(f"  [{idx+1}/{len(entries)}] {rate:.1f}条/s ETA:{eta:.0f}s 改:{errors}", flush=True)
    
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    elapsed = time.time() - t0
    print(f"✓ 完成! {len(results)}条, {elapsed:.1f}s, 修正{errors}条", flush=True)
    
    # 也复制到桌面
    import shutil
    shutil.copy(OUTPUT, os.path.expanduser("~/桌面/result.jsonl"))
    print("已复制到桌面", flush=True)

if __name__ == "__main__":
    main()
