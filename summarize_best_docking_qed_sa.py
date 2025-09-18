#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 summarize_best_docking.py 的基础上，增加 QED/SA 过滤：
  - 仅在满足 QED >= qed_min 且 SA <= sa_max 的个体集合中，
  - 按对接分数(DS)选择最佳个体（可选最小/最大）。

与原脚本保持其余逻辑一致：
  - 扫描 output_runs/exp_*/<receptor>/generation_*/docking_results/final_scored.smi
  - 仅纳入“完整”的实验（complete_start..complete_end 各代均存在对应文件且可读）
  - 在每个实验-受体范围内，跨世代选择一个“最佳”个体并输出

注意：若 final_scored.smi 缺失 QED/SA 列，将尝试用 RDKit/SA scorer 计算；
若仍无法得到 SA（sascorer 不可用），则该分子视为不满足 SA 过滤条件。

用法示例：
  python summarize_best_docking_qed_sa.py \
      --qed_min 0.5 --sa_max 5 \
      --score_order min \
      --output_root output_runs \
      --output_file best_docking_summary_qed_sa.csv
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import argparse
import re

from rdkit import Chem  # type: ignore
from rdkit.Chem import QED  # type: ignore

# 统计模块(带兜底)
try:
    import statistics  # type: ignore
except Exception:
    statistics = None  # type: ignore

# SA scorer 导入（与项目目录兼容）
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


def _parse_line(line: str) -> Optional[Tuple[str, float, Optional[float], Optional[float]]]:
    s = line.strip()
    if not s:
        return None
    parts = s.replace('\t', ' ').split()
    if len(parts) < 2:
        return None
    smiles = parts[0]
    try:
        score = float(parts[1])
    except ValueError:
        return None
    # 可选 QED / SA 列
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


def read_all_candidates(final_scored: Path) -> List[Dict]:
    """读取 final_scored.smi 的所有行，解析为候选列表。
    返回元素形如：{'smiles': str, 'score': float, 'qed': Optional[float], 'sa': Optional[float]}
    """
    candidates: List[Dict] = []
    if not final_scored.exists() or final_scored.stat().st_size == 0:
        return candidates
    with final_scored.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = _parse_line(line)
            if parsed is None:
                continue
            smiles, score, qed, sa = parsed
            candidates.append({'smiles': smiles, 'score': score, 'qed': qed, 'sa': sa})
    return candidates


def select_best_with_constraints(cands: List[Dict], qed_min: float, sa_max: float, score_order: str) -> Optional[Dict]:
    """在候选集中应用 QED/SA 过滤，返回按 score 排序后的最佳候选。
    score_order: 'min' 或 'max'
    若 SA 缺失且无法计算，则视为不满足 SA 过滤条件。
    """
    valid: List[Dict] = []
    for c in cands:
        q = c.get('qed')
        s = c.get('sa')
        if q is None or s is None:
            # 计算缺失的 QED/SA
            cq, cs = _compute_qed_sa(c['smiles'])
            if q is None:
                q = cq
            if s is None:
                s = cs
        # 过滤条件
        if q is None or s is None:
            continue
        if q >= qed_min and s <= sa_max:
            c2 = dict(c)
            c2['qed'] = q
            c2['sa'] = s
            valid.append(c2)

    if not valid:
        return None

    reverse = (score_order.lower() == 'max')
    valid.sort(key=lambda x: x['score'], reverse=reverse)
    return valid[0]


def _gather_rescore_stats(gen_dir: Path, chosen_smiles: str, rescore_glob: Optional[str]) -> Dict[str, Optional[float]]:
    """在给定 generation 目录中，根据 rescore_glob 搜索额外结果文件，
    收集与 chosen_smiles 匹配的 DS 值，返回简单统计。
    若 rescore_glob 为空或未命中/未匹配到该 SMILES，则返回 NA。
    """
    out = {
        'rescore_n': None,
        'rescore_mean': None,
        'rescore_std': None,
        'rescore_min': None,
        'rescore_max': None,
    }
    if not rescore_glob:
        return out
    paths = list(gen_dir.glob(rescore_glob))
    if not paths:
        return out
    vals: List[float] = []
    for p in paths:
        try:
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parsed = _parse_line(line)
                    if not parsed:
                        continue
                    smi, sc, _, _ = parsed
                    if smi == chosen_smiles:
                        vals.append(float(sc))
        except Exception:
            continue
    if not vals:
        return out
    n = len(vals)
    mean_v = sum(vals) / n
    if n >= 2:
        mu = mean_v
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        std_v = var ** 0.5
    else:
        std_v = 0.0
    out.update({
        'rescore_n': float(n),
        'rescore_mean': float(mean_v),
        'rescore_std': float(std_v),
        'rescore_min': float(min(vals)),
        'rescore_max': float(max(vals)),
    })
    return out


def main():
    ap = argparse.ArgumentParser(description='按 QED/SA 条件过滤后，统计各受体在迭代过程中的最佳对接分数（仅纳入完成设定迭代次数的实验）')
    ap.add_argument('--output_root', type=str, default='output_runs', help='输出根目录(相对项目根目录)')
    ap.add_argument('--output_file', type=str, default='best_docking_summary_qed_sa.csv', help='汇总CSV输出文件(相对项目根目录)')
    ap.add_argument('--complete_start', type=int, default=0, help='完整性判定的起始代编号（含）')
    ap.add_argument('--complete_end', type=int, default=10, help='完整性判定的结束代编号（含）')
    ap.add_argument('--qed_min', type=float, default=0.5, help='QED 最小阈值（含）')
    ap.add_argument('--sa_max', type=float, default=5.0, help='SA 最大阈值（含）')
    ap.add_argument('--score_order', type=str, choices=['min', 'max'], default='min', help='在有效集合内按 DS 选择最小或最大')
    ap.add_argument('--extra_stats_file', type=str, default='best_docking_distribution_stats_by_receptor_qed_sa.csv', help='分布统计与成功率CSV(相对项目根目录)')
    ap.add_argument('--topk', type=int, default=5, help='在分布统计中额外输出Top-K均值（按score_order方向）')
    # best-of-m 期望曲线
    ap.add_argument('--bestof_file', type=str, default='best_of_m_by_receptor_qed_sa.csv', help='best-of-m 曲线CSV(相对项目根目录)')
    ap.add_argument('--bestof_ms', type=str, default='1,2,5,10,20,50,100', help='m 列表，逗号分隔，例如 1,2,5,10')
    ap.add_argument('--bestof_bootstrap', type=int, default=1000, help='best-of-m 自助法重复次数')
    ap.add_argument('--bestof_seed', type=int, default=1337, help='best-of-m 随机种子')
    ap.add_argument('--bestof_plot', action='store_true', default=False, help='是否输出 best-of-m 曲线图（每个受体一张PNG）')
    ap.add_argument('--bestof_plot_dir', type=str, default='best_of_m_plots', help='best-of-m 曲线图输出目录(相对项目根目录)')
    # 多种子复评分占位：在 generation_* 目录内根据 glob 匹配文件，
    # 从中提取与所选 SMILES 相同的 DS 值，做均值/方差等统计
    ap.add_argument('--rescore_glob', type=str, default=None, help='相对于每个 generation_* 目录的glob，例如 "docking_results/final_scored_seed*.smi"；若不提供则跳过复评分统计')
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root = project_root / args.output_root
    if not output_root.exists():
        print(f"未找到输出目录: {output_root}")
        return

    # best[receptor][exp] = { 'score': float, 'smiles': str, 'qed': float, 'sa': float,
    #                          'exp': str, 'generation': str, 'path': Path }
    best: Dict[str, Dict[str, Dict]] = {}
    # 记录每个受体的“完整实验”总数，用于成功率统计
    total_complete_by_rec: Dict[str, int] = {}
    scanned_files = 0
    found_candidates = 0

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
                if not (fp.exists() and fp.stat().st_size > 0):
                    ok = False
                    break

            if not ok:
                continue
            # 计入该受体的完整实验数
            total_complete_by_rec[receptor] = total_complete_by_rec.get(receptor, 0) + 1

            # 在完整的代序列范围内，先按每代筛出一个符合条件且按 DS 排序的候选，再在跨代中选最佳
            per_exp_best: Optional[Dict] = None
            for g in range(start, end + 1):
                gd = gen_map[g]
                final_scored = gd / 'docking_results' / 'final_scored.smi'
                cands = read_all_candidates(final_scored)
                chosen = select_best_with_constraints(
                    cands,
                    qed_min=float(args.qed_min),
                    sa_max=float(args.sa_max),
                    score_order=str(args.score_order),
                )
                if chosen is None:
                    continue
                found_candidates += 1
                # 复评分统计（若指定）
                rescore_stats = _gather_rescore_stats(gd, chosen['smiles'], args.rescore_glob)
                chosen_full = {
                    'score': float(chosen['score']),
                    'smiles': chosen['smiles'],
                    'qed': float(chosen['qed']),
                    'sa': float(chosen['sa']),
                    'exp': exp_dir.name,
                    'generation': f'generation_{g}',
                    'path': final_scored,
                    **rescore_stats,
                }
                # 跨代聚合：在每个实验-受体范围内，按 score_order 选择全局最佳
                if per_exp_best is None:
                    per_exp_best = chosen_full
                else:
                    if args.score_order == 'min':
                        if chosen_full['score'] < per_exp_best['score']:
                            per_exp_best = chosen_full
                    else:  # max
                        if chosen_full['score'] > per_exp_best['score']:
                            per_exp_best = chosen_full

            if per_exp_best is not None:
                if receptor not in best:
                    best[receptor] = {}
                best[receptor][exp_dir.name] = per_exp_best

    # 输出汇总
    print("统计完成：")
    print(f"扫描文件数: {scanned_files}, 有效候选条目数(满足QED/SA): {found_candidates}")
    if not best:
        print("未找到任何满足 QED/SA 条件的记录。")
        return

    print("\n各受体在每个实验中的最佳对接分数（已过滤QED/SA）：")
    for receptor in sorted(best.keys()):
        print(f"受体: {receptor}")
        for exp_name in sorted(best[receptor].keys()):
            item = best[receptor][exp_name]
            print("  实验: {} | DS: {:.6f} | QED: {:.6f} | SA: {:.6f} | SMILES: {} | 代: {} | 文件: {}".format(
                exp_name,
                item['score'],
                item['qed'],
                item['sa'],
                item['smiles'],
                item['generation'],
                str(item['path'].relative_to(project_root))
            ))

    # 写入逐实验明细CSV（按受体分组，并在组内按 docking_score 升/降序）
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
                    'qed': float(item['qed']),
                    'sa': float(item['sa']),
                    'smiles': item['smiles'],
                    'generation': item['generation'],
                    'file': str(item['path'].relative_to(project_root))
                })

        with out_csv.open('w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            # 仅写核心字段，不写入末尾 rescore_* 指标
            w.writerow(['receptor', 'experiment', 'docking_score', 'qed', 'sa', 'smiles', 'generation', 'file'])
            # 受体有序遍历；在每个受体内部按 score 升/降序
            for receptor in sorted(best.keys()):
                group = [r for r in records if r['receptor'] == receptor]
                group.sort(key=lambda r: r['score'], reverse=(args.score_order == 'max'))
                for r in group:
                    w.writerow([
                        r['receptor'],
                        r['experiment'],
                        f"{r['score']:.6f}",
                        f"{r['qed']:.6f}",
                        f"{r['sa']:.6f}",
                        r['smiles'],
                        r['generation'],
                        r['file']
                    ])
        print(f"\n已写入CSV(按受体分组，组内按docking_score {'降序' if args.score_order == 'max' else '升序'}): {out_csv}")
    except Exception as e:
        print(f"写入CSV失败: {e}")

    # 可选：简单的按受体 Top 统计（仍按 score_order 方向进行前 k 的均值）
    try:
        import math as _math
        fracs = [(0.20, 'TOP_20'), (0.10, 'TOP_10'), (0.05, 'TOP_05')]
        rows = []
        print("\n按受体Docking Score Top统计（已过滤QED/SA）:")
        for receptor in sorted(best.keys()):
            scores = [float(best[receptor][exp]['score']) for exp in best[receptor].keys()]
            # 按方向排序
            scores.sort(reverse=(args.score_order == 'max'))
            n = len(scores)
            if n == 0:
                continue
            # 这里的“ALL 均值”不分方向
            mean_all = statistics.mean(scores) if statistics else sum(scores) / max(1, len(scores))
            print(f"受体: {receptor} | 全部 n={n} | 均值: {mean_all:.6f}")
            rows.append([receptor, 'ALL', '', n, f"{mean_all:.6f}"])
            for frac, name in fracs:
                k = max(1, _math.ceil(n * frac))
                top_scores = scores[:k]
                top_mean = (statistics.mean(top_scores) if statistics else sum(top_scores) / k)
                print(f"  - {name}: 前{int(frac*100)}% | n={k} | 均值: {top_mean:.6f}")
                rows.append([receptor, name, f"{frac:.2f}", k, f"{top_mean:.6f}"])

        if rows:
            per_rec_csv = project_root / 'best_docking_top_stats_by_receptor_qed_sa.csv'
            with per_rec_csv.open('w', newline='', encoding='utf-8') as f:
                import csv as _csv
                w = _csv.writer(f)
                w.writerow(['receptor', 'metric', 'fraction', 'count', 'mean_docking_score'])
                for row in rows:
                    w.writerow(row)
            print(f"已写入分受体Top统计CSV: {per_rec_csv}")
    except Exception as e:
        print(f"计算/写入分受体Top统计失败: {e}")

    # 新增：按受体输出分布统计与成功率
    try:
        def _percentile_nearest_rank(sorted_vals, p):
            # p: 0-100
            if not sorted_vals:
                return None
            import math
            n = len(sorted_vals)
            r = max(1, min(n, math.ceil((p/100.0) * n)))
            return float(sorted_vals[r-1])

        # 组装并写出
        extra_csv = project_root / args.extra_stats_file
        import csv as _csv
        with extra_csv.open('w', newline='', encoding='utf-8') as f:
            w = _csv.writer(f)
            w.writerow([
                'receptor', 'n_success', 'n_total_complete', 'success_rate',
                'mean', 'std', 'cv', 'iqr', 'mad', 'skew', 'long_tail_ratio',
                'p10', 'p25', 'p50', 'p75', 'p90', 'min', 'max',
                f"top{int(args.topk)}_mean"
            ])
            for receptor in sorted(set(list(best.keys()) + list(total_complete_by_rec.keys()))):
                # 成功数、总完整数、成功率
                n_success = len(best.get(receptor, {}))
                n_total = int(total_complete_by_rec.get(receptor, 0))
                success_rate = (n_success / n_total) if n_total > 0 else 0.0

                # 分布统计基于成功实验的 score 列表
                scores = [float(item['score']) for item in best.get(receptor, {}).values()]
                scores_sorted = sorted(scores, reverse=(args.score_order == 'max'))
                if scores:
                    mean_val = (statistics.mean(scores) if statistics else sum(scores) / len(scores))
                    if len(scores) >= 2:
                        try:
                            std_val = statistics.stdev(scores)  # type: ignore
                        except Exception:
                            mu = mean_val
                            std_val = (sum((x-mu)**2 for x in scores) / (len(scores)-1)) ** 0.5
                    else:
                        std_val = None
                    cv_val = (std_val / abs(mean_val)) if (std_val is not None and mean_val != 0.0) else None
                    srt = sorted(scores)
                    p10 = _percentile_nearest_rank(srt, 10)
                    p25 = _percentile_nearest_rank(srt, 25)
                    p50 = _percentile_nearest_rank(srt, 50)
                    p75 = _percentile_nearest_rank(srt, 75)
                    p90 = _percentile_nearest_rank(srt, 90)
                    # 形状指标
                    iqr = (None if (p25 is None or p75 is None) else (p75 - p25))
                    import math as _math
                    median = p50
                    mad = None
                    skew = None
                    ltr = None
                    if median is not None:
                        mad = (statistics.median([abs(x - median) for x in scores]) if statistics else None)
                    if std_val is not None and std_val > 0:
                        mu = mean_val
                        m3 = sum((x - mu) ** 3 for x in scores) / len(scores)
                        skew = m3 / (_math.pow(std_val, 3))
                    if median is not None and p10 is not None and p90 is not None:
                        denom = (median - p10)
                        ltr = ((p90 - median) / denom) if denom not in (0, None) else None
                    min_v = min(scores)
                    max_v = max(scores)
                    # top-k 均值（按方向）
                    k = max(1, min(len(scores_sorted), int(args.topk)))
                    topk_mean = (sum(scores_sorted[:k]) / k)
                else:
                    mean_val = std_val = cv_val = iqr = mad = skew = ltr = p10 = p25 = p50 = p75 = p90 = min_v = max_v = None
                    topk_mean = None

                def _fmt(x):
                    return 'NA' if x is None else f"{x:.6f}"

                w.writerow([
                    receptor,
                    n_success,
                    n_total,
                    _fmt(success_rate),
                    _fmt(mean_val),
                    _fmt(std_val),
                    _fmt(cv_val),
                    _fmt(iqr),
                    _fmt(mad),
                    _fmt(skew),
                    _fmt(ltr),
                    _fmt(p10), _fmt(p25), _fmt(p50), _fmt(p75), _fmt(p90),
                    _fmt(min_v), _fmt(max_v),
                    _fmt(topk_mean),
                ])
        print(f"已写入分受体分布与成功率CSV: {extra_csv}")
    except Exception as e:
        print(f"计算/写入分布与成功率失败: {e}")

    # best-of-m 期望曲线（自助法）
    try:
        import random as _random
        _random.seed(int(args.bestof_seed))
        # 解析 m 列表
        ms: List[int] = []
        for tok in str(args.bestof_ms).split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                mv = int(tok)
                if mv >= 1:
                    ms.append(mv)
            except Exception:
                pass
        ms = sorted(set(ms))
        if ms:
            bestof_csv = project_root / args.bestof_file
            import csv as _csv
            bestof_data = {}
            with bestof_csv.open('w', newline='', encoding='utf-8') as f:
                w = _csv.writer(f)
                w.writerow(['receptor', 'm', 'expected_best', 'std_best', 'n_bootstrap'])
                for receptor in sorted(best.keys()):
                    vals = [float(item['score']) for item in best[receptor].values()]
                    if not vals:
                        continue
                    nboot = int(args.bestof_bootstrap)
                    series = []
                    for m in ms:
                        bs = []
                        for _ in range(nboot):
                            sample = [_random.choice(vals) for __ in range(m)]
                            agg = (min(sample) if args.score_order == 'min' else max(sample))
                            bs.append(agg)
                        mu = sum(bs) / len(bs)
                        if len(bs) >= 2:
                            mu0 = mu
                            var = sum((x - mu0) ** 2 for x in bs) / (len(bs) - 1)
                            sd = var ** 0.5
                        else:
                            sd = 0.0
                        w.writerow([receptor, m, f"{mu:.6f}", f"{sd:.6f}", nboot])
                        series.append((m, mu, sd))
                    bestof_data[receptor] = series
            print(f"已写入 best-of-m 曲线CSV: {bestof_csv}")

            # 可选：绘图输出
            if args.bestof_plot:
                try:
                    import matplotlib.pyplot as plt
                    plot_dir = (project_root / args.bestof_plot_dir)
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    for receptor, series in bestof_data.items():
                        if not series:
                            continue
                        series = sorted(series, key=lambda t: t[0])
                        xs = [t[0] for t in series]
                        ys = [t[1] for t in series]
                        es = [t[2] for t in series]
                        fig, ax = plt.subplots(figsize=(5.5, 4.0))
                        ax.errorbar(xs, ys, yerr=es, fmt='-o', capsize=3)
                        # 若 m 跨度较大则用对数尺度
                        if (max(xs) / max(1, min(xs))) >= 10:
                            ax.set_xscale('log')
                        ax.set_xlabel('m (repeats)')
                        ax.set_ylabel('Expected best docking score')
                        ax.set_title(f'{receptor} (score_order={args.score_order})')
                        ax.grid(True, linestyle='--', alpha=0.4)
                        out_png = plot_dir / f'best_of_m_{receptor}.png'
                        fig.tight_layout()
                        fig.savefig(out_png, dpi=150)
                        plt.close(fig)
                    print(f"已输出 best-of-m 曲线图至目录: {plot_dir}")
                except Exception as e:
                    print(f"绘图失败(可能缺少 matplotlib)：{e}")
    except Exception as e:
        print(f"best-of-m 计算/写入失败: {e}")


if __name__ == '__main__':
    main()
