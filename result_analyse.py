#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取整合后的 smiles_output.smi（CSV 头：receptor,experiment,best_docking_score,qed,sa,smiles,generation,file），
按受体分组并在组内按 docking_score 升序排序，截取每个受体前 N 条（默认 300）
  - 前20% / 10% / 5% 的均值（向上取整，至少1个）

用法示例：
  python result_analyse.py \
      --input smiles_output.smi \
      --selected_csv smiles_output_300ranked_by_receptor.csv \
      --stats_csv smiles_output_300ranked_stats.csv \
      --per_receptor_limit 300
"""
from pathlib import Path
import argparse
import math
import statistics
import csv


def load_records(input_path: Path):    
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    records = []
    with input_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)        
        # 期望字段：receptor, experiment, best_docking_score, qed, sa, smiles, generation, file
        for row in reader:
            if not row:
                continue
            try:
                receptor = (row.get('receptor') or row.get('Receptor') or '').strip()
                experiment = (row.get('experiment') or row.get('Experiment') or '').strip()                
                score_str = (
                    row.get('best_docking_score')
                    or row.get('docking_score')
                    or row.get('score')
                    or row.get('DockingScore')
                    or row.get('Docking_Score')
                    or ''
                )
                score_str = score_str.strip()
                if score_str == '' or receptor == '':
                    continue
                score = float(score_str)               
                qed = row.get('qed'); sa = row.get('sa')
                qed = None if (qed is None or qed == '' or qed == 'NA') else float(qed)
                sa = None if (sa is None or sa == '' or sa == 'NA') else float(sa)
                smiles = (row.get('smiles') or '').strip()
                generation = (row.get('generation') or '').strip()
                file_path = (row.get('file') or row.get('path') or '').strip()
                records.append({
                    'receptor': receptor,
                    'experiment': experiment,
                    'score': score,
                    'qed': qed,
                    'sa': sa,
                    'smiles': smiles,
                    'generation': generation,
                    'file': file_path,
                })
            except Exception:                
                continue
    return records


def group_sort_and_limit(records, limit_per_receptor: int):
    receptors = sorted({r['receptor'] for r in records})
    selected = []
    for rec in receptors:
        group = [r for r in records if r['receptor'] == rec]
        group.sort(key=lambda x: x['score'])
        if limit_per_receptor > 0:
            group = group[:limit_per_receptor]
        selected.extend(group)
    return selected, receptors


def write_selected(selected, output_path: Path):   
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['receptor', 'experiment', 'best_docking_score', 'qed', 'sa', 'smiles', 'generation', 'file'])
        for r in selected:
            w.writerow([
                r['receptor'],
                r['experiment'],
                f"{r['score']:.6f}",
                'NA' if r['qed'] is None else f"{r['qed']:.6f}",
                'NA' if r['sa'] is None else f"{r['sa']:.6f}",
                r['smiles'],
                r['generation'],
                r['file'],
            ])


def compute_stats_per_receptor(selected, receptors, fractions=(0.50, 0.20, 0.10, 0.05, 0.03)):    
    rows = []
    print("按受体统计(基于各自前N条)：")
    for rec in receptors:
        scores = [r['score'] for r in selected if r['receptor'] == rec]
        n = len(scores)
        if n == 0:
            continue
        scores.sort()
        mean_all = statistics.mean(scores)
        var_all = statistics.pvariance(scores) if n > 1 else 0.0
        print(f"- 受体: {rec} | 计入样本 n={n} | 300条均值: {mean_all:.6f} | 方差: {var_all:.6f}")        
        frac_stats = []
        for frac in fractions:
            k = max(1, math.ceil(n * frac))
            sub = scores[:k]
            m = statistics.mean(sub)            
            v = statistics.pvariance(sub) if k > 1 else 0.0
            frac_stats.append((frac, k, m, v))
            pct = int(frac * 100)
            print(f"  · 前{pct:>2}% | n={k:>3} | 均值: {m:.6f} | 方差: {v:.6f}")     
        
        rows.append([
            rec,
            n,
            f"{mean_all:.6f}",
            # 50%
            frac_stats[0][1], f"{frac_stats[0][2]:.6f}",
            # 20%
            frac_stats[1][1], f"{frac_stats[1][2]:.6f}", f"{frac_stats[1][3]:.6f}",
            # 10%
            frac_stats[2][1], f"{frac_stats[2][2]:.6f}", f"{frac_stats[2][3]:.6f}",
            # 5%
            frac_stats[3][1], f"{frac_stats[3][2]:.6f}", f"{frac_stats[3][3]:.6f}",
            # 3%
            frac_stats[4][1], f"{frac_stats[4][2]:.6f}",
        ])
    return rows


def write_stats(rows, stats_path: Path):
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'receptor', 'count_selected', 'mean_300',
            'top50_count', 'top50_mean',
            'top20_count', 'top20_mean', 'top20_var',
            'top10_count', 'top10_mean', 'top10_var',
            'top05_count', 'top05_mean', 'top05_var',
            'top03_count', 'top03_mean',
        ])
        for row in rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description='按受体聚合smiles_output.smi')
    ap.add_argument('--input', type=str, default='smiles_output.smi', help='输入CSV(整合smi)')
    ap.add_argument('--selected_csv', type=str, default='smiles_output_top300_by_receptor.csv', help='输出明细CSV')
    ap.add_argument('--stats_csv', type=str, default='smiles_output_top300_stats.csv', help='输出统计CSV')
    ap.add_argument('--per_receptor_limit', type=int, default=300, help='每个受体选取的最大数量')
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    selected_path = Path(args.selected_csv).resolve()
    stats_path = Path(args.stats_csv).resolve()

    records = load_records(input_path)
    if not records:
        print(f"未从 {input_path} 读取到有效记录。")
        return

    selected, receptors = group_sort_and_limit(records, args.per_receptor_limit)
    write_selected(selected, selected_path)
    print(f"已写出筛选明细: {selected_path}")

    rows = compute_stats_per_receptor(selected, receptors)
    write_stats(rows, stats_path)
    print(f"已写出统计结果: {stats_path}")


if __name__ == '__main__':
    main()
