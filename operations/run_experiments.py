#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量实验调度器
===============
按稳妥方案执行 1000 次实验：
- 最大并发实验数: 默认 6
- 每实验：受体并行上限 3，每受体 CPU 并行 4（对接阶段）
- 按可见 GPU 列表为不同实验分配 gpt.device（轮转）
- 每实验完成后汇总 5 个受体的最佳分子（y 最大行的首行）到各自 CSV

可断点重入：若检测到目标实验的 5 个受体最终文件齐全且首行存在，则跳过。
"""
import os
import sys
import json
import time
import argparse
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_base_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_receptor_names(cfg: dict) -> list:
    return list(cfg.get('receptors', {}).get('target_list', {}).keys())


def build_exp_config(base_cfg: dict, exp_id: int, *, per_exp_max_workers: int, per_receptor_cpus: int, device_idx: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))  # 深拷贝
    # 性能参数
    perf = cfg.setdefault('performance', {})
    perf['parallel_processing'] = True
    perf['max_workers'] = int(per_exp_max_workers)
    perf['number_of_processors'] = int(per_receptor_cpus)
    perf['cleanup_intermediate_files'] = True
    # GPT 参数
    gpt = cfg.setdefault('gpt', {})
    gpt['seed'] = int(exp_id)
    gpt['device'] = str(device_idx)
    # 保留 visible_devices（若无则不动）
    return cfg


def exp_output_dir(root: Path, exp_id: int) -> Path:
    return root / f"exp_{exp_id:04d}"


def is_experiment_complete(exp_dir: Path, receptors: list) -> bool:
    for r in receptors:
        final_file = exp_dir / r / 'generation_10' / 'initial_population_docked.smi'
        if not final_file.exists():
            return False
        try:
            with open(final_file, 'r', encoding='utf-8') as f:
                first = None
                for line in f:
                    s = line.strip()
                    if s:
                        first = s
                        break
                if not first:
                    return False
        except Exception:
            return False
    return True


def parse_first_line(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.replace('\t', ' ').split()
            if len(parts) >= 5:
                smiles = parts[0]
                try:
                    ds = float(parts[1])
                except Exception:
                    ds = None
                try:
                    qed = float(parts[2]) if parts[2].upper() != 'NA' else None
                except Exception:
                    qed = None
                try:
                    sa = float(parts[3]) if parts[3].upper() != 'NA' else None
                except Exception:
                    sa = None
                try:
                    rag_y = float(parts[4]) if parts[4].upper() != 'NA' else None
                except Exception:
                    rag_y = None
                return smiles, ds, qed, sa, rag_y
            # 兼容只有 SMILES+DS 两列
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    ds = float(parts[1])
                except Exception:
                    ds = None
                return smiles, ds, None, None, None
    return None


def append_summary(summary_root: Path, receptor: str, row: dict):
    summary_root.mkdir(parents=True, exist_ok=True)
    out_csv = summary_root / f"{receptor}_best_across_experiments.csv"
    write_header = not out_csv.exists()
    with open(out_csv, 'a', encoding='utf-8') as f:
        if write_header:
            f.write('exp_id,receptor,smiles,docking_score,qed,sa,rag_y,output_dir\n')
        f.write('{exp_id},{receptor},{smiles},{ds},{qed},{sa},{rag_y},{outdir}\n'.format(
            exp_id=row.get('exp_id'),
            receptor=row.get('receptor'),
            smiles=row.get('smiles', ''),
            ds=('' if row.get('docking_score') is None else f"{row['docking_score']:.6f}"),
            qed=('' if row.get('qed') is None else f"{row['qed']:.6f}"),
            sa=('' if row.get('sa') is None else f"{row['sa']:.6f}"),
            rag_y=('' if row.get('rag_y') is None else f"{row['rag_y']:.6f}"),
            outdir=row.get('output_dir', '')
        ))


def run_single_experiment(exp_id: int, *, base_config_path: Path, output_root: Path, per_exp_max_workers: int, per_receptor_cpus: int, device_idx: int, receptors: list, log_dir: Path) -> bool:
    exp_dir = exp_output_dir(output_root, exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    # 构建专属配置
    base_cfg = load_base_config(base_config_path)
    cfg = build_exp_config(base_cfg, exp_id, per_exp_max_workers=per_exp_max_workers, per_receptor_cpus=per_receptor_cpus, device_idx=device_idx)
    exp_cfg_path = exp_dir / 'config_GA_gpt_rag.exp.json'
    save_json(cfg, exp_cfg_path)

    # 若已完成则跳过
    if is_experiment_complete(exp_dir, receptors):
        return True

    # 执行命令
    cmd = [
        sys.executable, str(PROJECT_ROOT / 'GA_gpt_rag.py'),
        '--config', str(exp_cfg_path),
        '--all_receptors',
        '--output_dir', str(exp_dir)
    ]
    log_file = log_dir / f"exp_{exp_id:04d}.log"
    with open(log_file, 'w', encoding='utf-8') as lf:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=lf, stderr=lf, env=os.environ.copy())
        ret = proc.wait()
    if ret != 0:
        return False

    # 汇总首行（最佳分子：已按 y 降序，首行即最佳）
    summary_root = output_root / 'summaries'
    for r in receptors:
        final_file = exp_dir / r / 'generation_10' / 'initial_population_docked.smi'
        if not final_file.exists():
            continue
        parsed = parse_first_line(final_file)
        if not parsed:
            continue
        smiles, ds, qed, sa, rag_y = parsed
        append_summary(summary_root, r, {
            'exp_id': exp_id,
            'receptor': r,
            'smiles': smiles,
            'docking_score': ds,
            'qed': qed,
            'sa': sa,
            'rag_y': rag_y,
            'output_dir': str(exp_dir)
        })
    return True


def main():
    ap = argparse.ArgumentParser(description='批量运行 GA-GPT-RAG 实验')
    ap.add_argument('--total', type=int, default=1000, help='总实验数')
    ap.add_argument('--start_id', type=int, default=1, help='起始实验ID（含）')
    ap.add_argument('--concurrency', type=int, default=6, help='同时运行的最大实验数')
    ap.add_argument('--per_exp_max_workers', type=int, default=3, help='每实验内最大受体并行数')
    ap.add_argument('--per_receptor_cpus', type=int, default=4, help='每受体CPU并行数（对接）')
    ap.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'GA_gpt' / 'config_GA_gpt_rag.json'))
    ap.add_argument('--output_root', type=str, default=str(PROJECT_ROOT / 'output_runs'))
    args = ap.parse_args()

    base_config_path = Path(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(base_config_path)
    receptors = get_receptor_names(base_cfg)
    if not receptors:
        print('配置中未找到受体列表，终止。')
        return 1

    # 读取可见 GPU 列表
    gpt_cfg = base_cfg.get('gpt', {})
    visible_devices = gpt_cfg.get('visible_devices', None)
    if isinstance(visible_devices, list) and len(visible_devices) > 0:
        # 在父进程设置一次，子进程继承
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in visible_devices)
        num_visible = len(visible_devices)
    else:
        num_visible = 0

    start_id = int(args.start_id)
    end_id = start_id + int(args.total) - 1

    futures = []
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as ex:
        for exp_id in range(start_id, end_id + 1):
            # 轮转分配设备索引（相对于可见列表）
            if num_visible > 0:
                device_idx = exp_id % num_visible
            else:
                device_idx = 0
            futures.append(ex.submit(
                run_single_experiment,
                exp_id,
                base_config_path=base_config_path,
                output_root=output_root,
                per_exp_max_workers=int(args.per_exp_max_workers),
                per_receptor_cpus=int(args.per_receptor_cpus),
                device_idx=int(device_idx),
                receptors=receptors,
                log_dir=log_dir
            ))

        for fut in as_completed(futures):
            ok = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
            if (success + failed) % 5 == 0:
                print(f"进度: 完成 {success} 个, 失败 {failed} 个")

    print(f"全部任务结束: 成功 {success}, 失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())


