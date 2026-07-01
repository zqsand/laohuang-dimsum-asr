#!/usr/bin/env python3
"""
点心杯本地评分容器 — 一键出分 + 对比基线

用法:
  python scripts/eval_pipeline.py --predictions ./outputs/predictions.jsonl

输出:
  ./eval_output/metrics.json          — 完整评分报告
  ./eval_output/comparison.json       — 与基线对比
  ./eval_output/delta_report.txt      — 文本摘要
"""

import json, os, sys, argparse, subprocess, tempfile
from datetime import datetime


# 路径常量
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))

EVALUATOR = os.path.join(PROJECT_DIR, 'evaluator.py')
SEC_EVALUATOR = os.path.join(PROJECT_DIR, 'sec_evaluator.py')
BASELINE_METRICS = os.path.join(PROJECT_DIR, 'baseline_output', 'metrics.json')
TEST_JSONL = os.path.join(PROJECT_DIR, 'test.jsonl')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'eval_output')

# 基线指标 (2026-06-01 baseline whisper-small)
BASELINE = {
    'cer': 0.19027382816244368,
    'char_accuracy_approx': 0.8097261718375564,
    'sentence_accuracy': 0.1792186870720902,
}


def parse_args():
    parser = argparse.ArgumentParser(description='点心杯本地评分容器')
    parser.add_argument('--predictions', '-p', type=str, required=True,
                        help='模型预测 JSONL (含 pred_text 字段)')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help='评分输出目录')
    parser.add_argument('--name', '-n', type=str, default='unnamed',
                        help='模型名称（用于对比报告）')
    parser.add_argument('--sec_test', '-s', type=str, default=None,
                        help='可选的 sec_test.jsonl 路径')
    return parser.parse_args()


def run_evaluator(pred_jsonl: str, output_dir: str) -> dict:
    """运行官方 evaluator.py 并解析结果"""
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [sys.executable, EVALUATOR,
         '--pred_jsonl', pred_jsonl,
         '--report_dir', output_dir,
         '--prediction_field', 'pred_text',
         '--reference_field', 'ref_text'],
        capture_output=True, text=True, cwd=PROJECT_DIR
    )

    metrics_path = os.path.join(output_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        print(f"⚠️  evaluator 输出:\n{result.stdout}\n{result.stderr}")
        metrics = {}

    return metrics


def run_sec_evaluator(sec_test: str, output_dir: str) -> float:
    """运行官方 sec_evaluator.py 并解析分数"""
    score_file = os.path.join(output_dir, 'sec_score.txt')

    result = subprocess.run(
        [sys.executable, SEC_EVALUATOR, sec_test, score_file],
        capture_output=True, text=True, cwd=PROJECT_DIR
    )

    if os.path.exists(score_file):
        with open(score_file) as f:
            return float(f.read().strip())
    return None


def compare_to_baseline(metrics: dict, model_name: str) -> dict:
    """对比基线并生成 delta 报告"""
    cer = metrics.get('cer', 0)
    sent_acc = metrics.get('sentence_accuracy', 0)
    cer_delta = BASELINE['cer'] - cer          # 负 = 更差, 正 = 更好
    sent_delta = sent_acc - BASELINE['sentence_accuracy']

    comparison = {
        'model_name': model_name,
        'eval_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_samples': metrics.get('num_samples', 0),
        'baseline': {
            'cer': BASELINE['cer'],
            'sentence_accuracy': BASELINE['sentence_accuracy'],
        },
        'current': {
            'cer': cer,
            'char_accuracy_approx': metrics.get('char_accuracy_approx', 0),
            'sentence_accuracy': sent_acc,
        },
        'delta': {
            'cer': cer_delta,                    # 正 = CER改善
            'cer_percent': f'{cer_delta * 100:+.2f}%',
            'sentence_accuracy': sent_delta,
            'sent_acc_percent': f'{sent_delta * 100:+.2f}%',
        },
        'verdict': '',
    }

    if cer_delta < 0:
        comparison['verdict'] = '❌ 比基线更差'
    elif cer_delta < 0.01:
        comparison['verdict'] = '➖ 与基线持平 (±1%)'
    elif cer_delta < 0.05:
        comparison['verdict'] = '✅ 优于基线 (CER改善 1-5%)'
    else:
        comparison['verdict'] = '🎉 大幅优于基线 (CER改善 >5%)'

    return comparison


def generate_text_report(comparison: dict, sec_score: float = None) -> str:
    """生成可读的文本报告"""
    lines = []
    lines.append('=' * 50)
    lines.append(f'  点心杯 ASR 评分报告')
    lines.append(f'  模型: {comparison["model_name"]}')
    lines.append(f'  日期: {comparison["eval_date"]}')
    lines.append('=' * 50)
    lines.append('')
    lines.append(f'📊 测试集: {comparison["num_samples"]} 条')
    lines.append('')
    lines.append(f'  指标         基线        当前        Δ')
    lines.append(f'  ──────────────────────────────────────')
    lines.append(f'  CER          {comparison["baseline"]["cer"]:.4f}    {comparison["current"]["cer"]:.4f}    {comparison["delta"]["cer"]:+.4f} ({comparison["delta"]["cer_percent"]})')
    lines.append(f'  句精度       {comparison["baseline"]["sentence_accuracy"]:.4f}    {comparison["current"]["sentence_accuracy"]:.4f}    {comparison["delta"]["sentence_accuracy"]:+.4f} ({comparison["delta"]["sent_acc_percent"]})')
    lines.append(f'  字准确率     —          {comparison["current"]["char_accuracy_approx"]:.4f}    —')
    lines.append('')
    lines.append(f'  结论: {comparison["verdict"]}')

    if sec_score is not None:
        lines.append('')
        lines.append(f'🔍 语料ID匹配准确率: {sec_score:.1f}%')

    lines.append('')
    lines.append('=' * 50)
    return '\n'.join(lines)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.predictions):
        print(f"❌ 预测文件不存在: {args.predictions}")
        sys.exit(1)

    print(f'{"=" * 50}')
    print(f'  点心杯本地评分容器')
    print(f'  模型: {args.name}')
    print(f'  预测: {args.predictions}')
    print(f'  输出: {args.output}')
    print(f'{"=" * 50}\n')

    # Step 1: 主评分
    print('📊 [1/3] 运行官方 evaluator...')
    metrics = run_evaluator(args.predictions, args.output)
    if not metrics:
        print('❌ 评分失败')
        sys.exit(1)
    print(f'   CER: {metrics.get("cer", "?"):.4f}')
    print(f'   句精度: {metrics.get("sentence_accuracy", "?"):.4f}')

    # Step 2: 对比基线
    print(f'\n📈 [2/3] 对比基线...')
    comparison = compare_to_baseline(metrics, args.name)

    comp_path = os.path.join(args.output, 'comparison.json')
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f'   对比结果: {comp_path}')

    # Step 3: sec 评分（可选）
    sec_score = None
    if args.sec_test and os.path.exists(args.sec_test):
        print(f'\n🔍 [3/3] 运行 sec_evaluator...')
        sec_score = run_sec_evaluator(args.sec_test, args.output)
        if sec_score is not None:
            print(f'   语料ID匹配: {sec_score:.1f}%')
    else:
        print(f'\n🔍 [3/3] 跳过 sec_evaluator (未提供 --sec_test)')

    # 输出报告
    report = generate_text_report(comparison, sec_score)
    report_path = os.path.join(args.output, 'delta_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f'\n📋 评分报告:')
    print(report)
    print(f'\n✅ 评分完成 → {args.output}')


if __name__ == '__main__':
    main()
