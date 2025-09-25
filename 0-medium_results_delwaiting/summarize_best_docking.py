#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 output_runs 下所有实验与受体，在每一代的
  generation_*/docking_results/final_scored.smi
中读取第一行（已按对接分数升序排好），取其对接分数（第二列），
并统计每个受体在迭代过程中达到的最佳（最小）对接分数。

用法：
  python summarize_best_docking.py

输出：
  按受体打印最佳分数及其来源(experiment / generation / 路径），
  同时在 stdout 给出总扫描统计。
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
import re
from rdkit import Chem  # type: ignore
from rdkit.Chem import QED  # type: ignore
# 统计模块(带兜底)
try:
    import statistics  # type: ignore
    _HAS_STATISTICS = True
except Exception:
    statistics = None  # type: ignore
    _HAS_STATISTICS = False
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore
import statistics

# SA scorer 回退导入（与项目目录兼容）
import sys as _sys
_SCORING_DIR = str((Path(__file__).resolve().parent / 'operations' / 'scoring').resolve())
if _SCORING_DIR not in _sys.path:
    _sys.path.insert(0, _SCORING_DIR)
_calc_sa = None
try:
    from sascorer import calculateScore as _calc_sa  # type: ignore
except Exception:
    _ALT_SCORING_DIR = str((Path(__file__).resolve().parent / 'fragment_GPT' / 'utils').resolve())
    if _ALT_SCORING_DIR not in _sys.path:
        _sys.path.insert(0, _ALT_SCORING_DIR)
    try:
        from sascorer import calculateScore as _calc_sa  # type: ignore
    except Exception:
        _calc_sa = None


def read_top1_score_from_final_scored(file_path: Path) -> Optional[Tuple[str, float, Optional[float], Optional[float]]]:
    """读取 final_scored.smi 第一条记录，返回 (smiles, docking_score)。
    若文件不存在或格式不符，返回 None。
    """
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return None
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.replace('\t', ' ').split()
                if len(parts) < 2:
                    continue
                smiles = parts[0]
                try:
                    score = float(parts[1])
                except ValueError:
                    continue
                # 解析可选 QED/SA 列
                qed = None
                sa = None
                try:
                    if len(parts) >= 3 and parts[2] != 'NA':
                        qed = float(parts[2])
                    if len(parts) >= 4 and parts[3] != 'NA':
                        sa = float(parts[3])
                except Exception:
                    pass
                return smiles, score, qed, sa
        return None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description='统计各受体在迭代过程中的最佳(最小)对接分数（仅纳入完成设定迭代次数的实验）')
    ap.add_argument('--output_root', type=str, default='output_runs', help='输出根目录(相对项目根目录)')
    ap.add_argument('--output_file', type=str, default='best_docking_summary.csv', help='汇总CSV输出文件(相对项目根目录)')
    #ap.add_argument('--output_txt', type=str, default=None, help='可选: 额外写入可读文本摘要文件(相对项目根目录)')
    #ap.add_argument('--stats_file', type=str, default='best_docking_stats.csv', help='按受体聚合(均值/方差)CSV(相对项目根目录)')
    # 新的完整性判定：显式指定起止代号（含端点）。按你的实验默认 0..10。
    ap.add_argument('--complete_start', type=int, default=0, help='完整性判定的起始代编号（含）')
    ap.add_argument('--complete_end', type=int, default=10, help='完整性判定的结束代编号（含）')
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root = project_root / args.output_root
    if not output_root.exists():
        print(f"未找到输出目录: {output_root}")
        return

    # best[receptor][exp] = { 'score': float, 'smiles': str, 'qed': Optional[float], 'sa': Optional[float],
    #                          'exp': str, 'generation': str, 'path': Path }
    best: Dict[str, Dict[str, Dict]] = {}
    scanned_files = 0
    found_records = 0

    # 遍历 exp_* 目录
    for exp_dir in sorted(output_root.glob('exp_*')):
        if not exp_dir.is_dir():
            continue

        # 遍历受体目录（exp_* 下直接子目录即为受体名）
        for receptor_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            receptor = receptor_dir.name

            # 过滤非受体的常见目录名（保险起见）
            if receptor in {'.git', 'logs', 'summaries'}:
                continue

            # 收集该受体的所有代目录及其编号
            gen_map: Dict[int, Path] = {}
            pat = re.compile(r'^generation_(\d+)$')
            for gen_dir in (p for p in receptor_dir.iterdir() if p.is_dir()):
                m = pat.match(gen_dir.name)
                if not m:
                    continue
                try:
                    idx = int(m.group(1))
                except Exception:
                    continue
                gen_map[idx] = gen_dir

            # 判定是否为“完整实验”：检查 complete_start..complete_end（含端点）是否全部存在且可读取
            start = int(args.complete_start)
            end = int(args.complete_end)
            ok = True
            for g in range(start, end + 1):
                gd = gen_map.get(g)
                if gd is None:
                    ok = False
                    break
                fp = gd / 'docking_results' / 'final_scored.smi'
                scanned_files += 1
                if read_top1_score_from_final_scored(fp) is None:
                    ok = False
                    break

            # 若未完整，则跳过该实验-受体
            if not ok:
                continue

            # 在完整的代序列范围内统计最佳
            per_exp_best: Optional[Dict] = None
            for g in range(start, end + 1):
                gd = gen_map[g]
                final_scored = gd / 'docking_results' / 'final_scored.smi'
                res = read_top1_score_from_final_scored(final_scored)
                if res is None:
                    # 理论上不会发生（已在完整性检查中验证），防御性跳过
                    continue
                smiles, score, qed, sa = res
                found_records += 1

                if per_exp_best is None or score < per_exp_best['score']:
                    per_exp_best = {
                        'score': score,
                        'smiles': smiles,
                        'qed': qed,
                        'sa': sa,
                        'exp': exp_dir.name,
                        'generation': f'generation_{g}',
                        'path': final_scored,
                    }

            if per_exp_best is not None:
                if receptor not in best:
                    best[receptor] = {}
                best[receptor][exp_dir.name] = per_exp_best

    # 输出汇总
    print("统计完成：")
    print(f"扫描文件数: {scanned_files}, 有效记录数: {found_records}")
    if not best:
        print("未找到任何 final_scored.smi 的有效记录。")
        return

    print("\n各受体在每个实验中的最佳(最小)对接分数：")
    # 若缺失QED/SA则尝试计算
    def _compute_qed_sa(smiles: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            qed_val = float(QED.qed(mol))
            sa_val = float(_calc_sa(mol)) if _calc_sa else None
            return qed_val, sa_val
        except Exception:
            return None, None

    for receptor in sorted(best.keys()):
        print(f"受体: {receptor}")
        for exp_name in sorted(best[receptor].keys()):
            item = best[receptor][exp_name]
            if item.get('qed') is None or item.get('sa') is None:
                q, s = _compute_qed_sa(item['smiles'])
                if item.get('qed') is None:
                    item['qed'] = q
                if item.get('sa') is None:
                    item['sa'] = s
            print("  实验: {} | 最佳DS: {:.6f} | QED: {} | SA: {} | SMILES: {} | 代: {} | 文件: {}".format(
                exp_name,
                item['score'],
                'NA' if item['qed'] is None else f"{item['qed']:.6f}",
                'NA' if item['sa'] is None else f"{item['sa']:.6f}",
                item['smiles'],
                item['generation'],
                str(item['path'].relative_to(project_root))
            ))

    # 写入逐实验明细CSV（按受体分组，并在组内按 docking_score 升序排序）
    out_csv = project_root / args.output_file
    try:
        import csv
        # 收集所有记录（后续既用于分组写出，也可被Top统计复用）
        records = []
        for receptor in best.keys():
            for exp_name, item in best[receptor].items():
                records.append({
                    'receptor': receptor,
                    'experiment': exp_name,
                    'score': float(item['score']),
                    'qed': item.get('qed'),
                    'sa': item.get('sa'),
                    'smiles': item['smiles'],
                    'generation': item['generation'],
                    'file': str(item['path'].relative_to(project_root))
                })

        with out_csv.open('w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['receptor', 'experiment', 'best_docking_score', 'qed', 'sa', 'smiles', 'generation', 'file'])
            # 受体有序遍历；在每个受体内部按 score 升序
            for receptor in sorted(best.keys()):
                # 组内排序
                group = [r for r in records if r['receptor'] == receptor]
                group.sort(key=lambda r: r['score'])
                for r in group:
                    w.writerow([
                        r['receptor'],
                        r['experiment'],
                        f"{r['score']:.6f}",
                        'NA' if r['qed'] is None else f"{r['qed']:.6f}",
                        'NA' if r['sa'] is None else f"{r['sa']:.6f}",
                        r['smiles'],
                        r['generation'],
                        r['file']
                    ])
        print(f"\n已写入CSV(按受体分组且组内按docking_score排序): {out_csv}")
    except Exception as e:
        print(f"写入CSV失败: {e}")

    # 已移除：按受体聚合(均值/方差)的 stats CSV 写入逻辑

    # 全局 Docking Score 统计与写入 CSV 已移除

    # 计算分受体的 top 20% / 10% / 5% 的 docking_score 均值
    try:
        import math as _math
        fracs = [(0.20, 'TOP_20'), (0.10, 'TOP_10'), (0.05, 'TOP_05')]
        rows = []
        print("\n按受体Docking Score Top统计：")
        for receptor in sorted(best.keys()):
            scores = sorted(float(best[receptor][exp]['score']) for exp in best[receptor].keys())
            n = len(scores)
            if n == 0:
                continue
            mean_all = statistics.mean(scores)
            print(f"受体: {receptor} | 全部 n={n} | 均值: {mean_all:.6f}")
            rows.append([receptor, 'ALL', '', n, f"{mean_all:.6f}"])
            for frac, name in fracs:
                k = max(1, _math.ceil(n * frac))
                top_scores = scores[:k]
                top_mean = statistics.mean(top_scores)
                print(f"  - {name}: 前{int(frac*100)}% | n={k} | 均值: {top_mean:.6f}")
                rows.append([receptor, name, f"{frac:.2f}", k, f"{top_mean:.6f}"])

        if rows:
            per_rec_csv = project_root / 'best_docking_top_stats_by_receptor.csv'
            with per_rec_csv.open('w', newline='', encoding='utf-8') as f:
                import csv as _csv
                w = _csv.writer(f)
                w.writerow(['receptor', 'metric', 'fraction', 'count', 'mean_best_docking_score'])
                for row in rows:
                    w.writerow(row)
            print(f"已写入分受体Top统计CSV: {per_rec_csv}")
    except Exception as e:
        print(f"计算/写入分受体Top统计失败: {e}")

    # 可选写入可读文本摘要（鲁棒处理：参数可能被注释掉）
    output_txt = getattr(args, 'output_txt', None)
    if output_txt:
        out_txt = project_root / output_txt
        try:
            lines = []
            lines.append(f"统计完成\n扫描文件数: {scanned_files}, 有效记录数: {found_records}\n")
            lines.append("\n各受体最佳（最小）对接分数：\n")
            for receptor in sorted(best.keys()):
                item = best[receptor]
                lines.append(
                    f"- 受体: {receptor} | 最佳DS: {item['score']:.6f} | SMILES: {item['smiles']}\n"
                    f"  来源: {item['exp']} / {item['generation']}\n"
                    f"  文件: {item['path'].relative_to(project_root)}\n"
                )
            out_txt.write_text(''.join(lines), encoding='utf-8')
            print(f"已写入文本摘要: {out_txt}")
        except Exception as e:
            print(f"写入文本摘要失败: {e}")


if __name__ == '__main__':
    main()
