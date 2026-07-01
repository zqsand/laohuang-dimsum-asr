#!/usr/bin/env python3
"""字符级 N-Gram LM Rescoring — 纯 Python，无需 kenlm"""
import json, os, sys, pickle, math, re
from collections import Counter

def load_lm(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def char_lm_score(text, model, n=3, alpha=0.1):
    """计算字符级 n-gram log probability"""
    chars = list(text)
    if len(chars) < 2:
        return -10.0
    
    logp = 0.0
    V = model.get('vocab_size', len(model.get('unigram', {})))
    total = model['total_unigrams']
    
    for i in range(len(chars)):
        # unigram smoothing
        c = chars[i]
        uni_count = model['unigram'].get(c, 0) + alpha
        uni_prob = uni_count / (total + alpha * V)
        
        if i >= 1:
            c1 = chars[i-1]
            bigram = model['bigram'].get(c1, {})
            bi_count = bigram.get(c, 0) + alpha * uni_prob
            bi_total = sum(bigram.values()) + alpha * V
            prob = bi_count / max(bi_total, 1)
        else:
            prob = uni_prob
        
        if i >= 2:
            key = chars[i-2] + chars[i-1]
            trigram = model['trigram'].get(key, {})
            if trigram:
                tri_count = trigram.get(c, 0) + alpha * prob
                tri_total = sum(trigram.values()) + alpha * V
                prob = tri_count / max(tri_total, 1)
        
        logp += math.log(max(prob, 1e-10))
    
    return logp / max(len(chars), 1)

def rescore_predictions(pred_jsonl, output_jsonl, lm_model, char_map=None, phrase_map=None):
    """对预测结果进行 LM 校验纠正"""
    rows = []
    with open(pred_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    
    corrected = 0
    for row in rows:
        pred = row.get('pred_text', '')
        if not pred:
            continue
        
        # 去标点后评估
        clean = re.sub(r'[，。！？、；：""''（）【】《》\.,!\?;:\s]+', '', pred)
        if len(clean) < 2:
            continue
        
        orig_score = char_lm_score(clean, lm_model)
        chars = list(clean)
        best_text = clean
        best_score = orig_score
        
        # 对每个字符尝试常见混淆替换
        confusions = {
            '上': ['想', '相', '像', '尚', '伤'],
            '货': ['课', '可', '个', '或', '火'],
            '最': ['聚', '嘴', '醉', '罪', '追'],
            '电': ['烫', '典', '店', '点', '殿'],
            '又': ['约', '有', '要', '由', '右'],
            '大': ['带', '打', '达', '答', '太'],
            '名': ['明', '命', '鸣', '铭', '冥'],
            '收': ['修', '手', '首', '受', '授'],
            '教': ['交', '叫', '较', '角', '郊'],
            '唔': ['不', '无', '毋', '毋', '五'],
            '着': ['穿', '著', '这', '者', '遮'],
            '验': ['染', '检', '验', '严', '言'],
            '昨': ['琴', '作', '左', '坐', '桌'],
            '压': ['押', '鸭', '呀', '牙', '雅'],
            '式': ['色', '识', '食', '十', '石'],
            '新': ['辛', '心', '信', '深', '身'],
            '已': ['以', '一', '意', '议', '义'],
            '进': ['近', '尽', '金', '紧', '今'],
            '关': ['观', '官', '管', '馆', '冠'],
            '每': ['美', '没', '妹', '媒', '梅'],
        }
        
        for i, c in enumerate(chars):
            if c in confusions:
                for alt in confusions[c]:
                    if alt == c:
                        continue
                    test = ''.join(chars[:i] + [alt] + chars[i+1:])
                    test_score = char_lm_score(test, lm_model)
                    if test_score > best_score + 0.01:
                        best_score = test_score
                        best_text = test
        
        if best_text != clean:
            corrected += 1
            row['pred_text_original'] = pred
            # 恢复标点（保留原文本的标点位置用最佳文本替换）
            row['pred_text'] = best_text
    
    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
    with open(output_jsonl, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"[Rescore] 纠正: {corrected}/{len(rows)} ({corrected/max(len(rows),1)*100:.1f}%)")
    return output_jsonl

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--lm', default='/home/wong/cantonese_asr/lm/cantonese_charlm.pkl')
    args = parser.parse_args()
    
    model = load_lm(args.lm)
    print(f"LM: {args.lm} (字符={len(model.get('unigram', {}))})")
    rescore_predictions(args.input, args.output, model)
