#我需要对比的的指标计算
import argparse
import os
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tdc import Oracle, Evaluator

# 使用TDC进行所有指标评估
qed_evaluator = Oracle('qed')
sa_evaluator = Oracle('sa')
diversity_evaluator = Evaluator(name='Diversity')
# 注意: TDC的Novelty评估器需要一个初始SMILES列表进行初始化
# 我们将在主函数中根据参数动态创建它

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

def calculate_sa_scores(smiles_list: list) -> list:
    """使用TDC Oracle批量计算SA分数。"""
    if not smiles_list:
        return []
    print(f"使用TDC批量计算 {len(smiles_list)} 个分子的SA分数...")
    return sa_evaluator(smiles_list)

def calculate_qed_scores(smiles_list: list) -> list:
    """使用TDC Oracle批量计算QED分数。"""
    if not smiles_list:
        return []
    print(f"使用TDC批量计算 {len(smiles_list)} 个分子的QED分数...")
    return qed_evaluator(smiles_list)

def load_smiles_from_file(filepath):   #加载smile
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            smiles = line.strip().split()[0] 
            if smiles:
                smiles_list.append(smiles)    
    return smiles_list

def load_smiles_and_scores_from_file(filepath):   #加载smile和score：对接之后输出文件（带分数）
    molecules = []
    scores = []
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    score = float(parts[1])
                    molecules.append(smiles)
                    scores.append(score)
                    smiles_list.append(smiles)
                except ValueError:
                    print(f"Warning: Could not parse score for SMILES: {smiles}")
            elif len(parts) == 1: # If only SMILES is present, no score
                smiles_list.append(parts[0])    
    return smiles_list, molecules, scores

def get_rdkit_mols(smiles_list): #smiles-----mol
    mols = []
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            valid_smiles.append(s)
        else:
            print(f"Warning: Could not parse SMILES: {s}")
    return mols, valid_smiles

def calculate_docking_stats(scores):
    """Calculates Top-1, Top-10 mean, Top-100 mean docking scores."""    
    sorted_scores = sorted(scores) # Docking scores, lower is better
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else np.nan    #top1
    top10_scores = sorted_scores[:10]
    top10_mean = np.mean(top10_scores) if top10_scores else np.nan           #top10
    top100_scores = sorted_scores[:100]
    top100_mean = np.mean(top100_scores) if top100_scores else np.nan        #top100
    return top1_score, top10_mean, top100_mean

def calculate_novelty(current_smiles: list, initial_smiles_list: list) -> float:
    """使用TDC Evaluator计算新颖性。"""
    if not current_smiles:
        return 0.0
    # 正确用法: 直接按位置传入参数
    novelty_evaluator = Evaluator(name='Novelty')
    return novelty_evaluator(current_smiles, initial_smiles_list)

def calculate_top100_diversity(smiles_list: list) -> float:
    """使用TDC Evaluator计算Top-100的多样性。"""
    top_smiles = smiles_list[:min(100, len(smiles_list))]
    if not top_smiles:
        return 0.0
    return diversity_evaluator(top_smiles)

def print_calculation_results(results):    
    print("Calculation Results:")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generation of molecules.")
    parser.add_argument("--current_population_docked_file", type=str, required=True,
                        help="Path to the SMILES file of the current population with docking scores (SMILES score per line).")
    parser.add_argument("--initial_population_file", type=str, required=True,
                        help="Path to the SMILES file of the initial population (for novelty calculation).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file to save calculated metrics (e.g., results.txt or results.csv).")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Optional JSON config to read RAG normalization (selection.rag_score_settings.normalization).")
    
    args = parser.parse_args()
    print(f"Processing population file: {args.current_population_docked_file}")
    print(f"Using initial population for novelty: {args.initial_population_file}")
    print(f"Saving results to: {args.output_file}")

    # 加载当前SMILES和对接分数
    current_smiles_list, scored_molecules_smiles, docking_scores = load_smiles_and_scores_from_file(args.current_population_docked_file)
    
    # 对接分数排序
    if scored_molecules_smiles and docking_scores:
        molecules_with_scores = sorted(zip(scored_molecules_smiles, docking_scores), key=lambda x: x[1])
        sorted_smiles = [s for s, _ in molecules_with_scores]
    else:
        sorted_smiles = current_smiles_list # 如果没有分数，则使用原始顺序
        
    # 0. 可选：读取 RAG 归一化配置
    norm = { 'ds_clip_min': -20.0, 'ds_clip_max': 0.0, 'sa_max_value': 10.0, 'sa_denominator': 9.0 }
    if args.config_file and os.path.exists(args.config_file):
        try:
            import json
            with open(args.config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            _s = cfg.get('selection', {}).get('rag_score_settings', {}).get('normalization', {})
            norm.update({
                'ds_clip_min': _s.get('ds_clip_min', norm['ds_clip_min']),
                'ds_clip_max': _s.get('ds_clip_max', norm['ds_clip_max']),
                'sa_max_value': _s.get('sa_max_value', norm['sa_max_value']),
                'sa_denominator': _s.get('sa_denominator', norm['sa_denominator']),
            })
        except Exception:
            pass

    # 1. Docking Score Metrics
    top1_score, top10_mean_score, top100_mean_score = calculate_docking_stats(docking_scores)
    
    # 定义用于计算所有属性指标的精英分子群体 (Top 100)
    smiles_for_scoring = sorted_smiles[:min(100, len(sorted_smiles))]
    score_description = f"Top {len(smiles_for_scoring)}"

    # 2. Novelty (基于Top 100精英种群)
    initial_smiles = load_smiles_from_file(args.initial_population_file)
    novelty = calculate_novelty(smiles_for_scoring, initial_smiles)
    
    # 3. Diversity (Top 100 molecules)
    diversity = calculate_top100_diversity(smiles_for_scoring)
    
    # 4. QED & SA Scores (for Top 100)
    qed_scores = calculate_qed_scores(smiles_for_scoring)
    sa_scores = calculate_sa_scores(smiles_for_scoring)
    
    mean_qed = np.mean(qed_scores) if qed_scores else np.nan
    mean_sa = np.mean(sa_scores) if sa_scores else np.nan

    # 5. 计算 y = DS_hat * QED * SA_hat 统计（针对“当前群体”文件）
    # 5.1 先为当前群体的所有分子构建 SMILES->DS 字典（若缺 DS 则 y 无法计算）
    smiles_to_ds = {}
    with open(args.current_population_docked_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    smiles_to_ds[parts[0]] = float(parts[1])
                except Exception:
                    continue

    # 5.2 定义归一化函数
    def _ds_hat(x: float) -> float:
        x = max(float(norm['ds_clip_min']), min(float(norm['ds_clip_max']), x))
        return -x / 20.0
    def _sa_hat(x: float) -> float:
        return max(0.0, min(1.0, (float(norm['sa_max_value']) - x) / float(norm['sa_denominator'])))

    # 5.3 计算“当前群体全体”的 y 统计
    # 为避免再次批量算性质，这里仅在对接输出含 QED/SA 时可直接读取；否则针对 Top-100 计算一个子集统计
    y_all = []
    ds_all = []
    # 如果当前文件已包含 QED/SA（我们刚在对接阶段新增了列），可直接解析列 3、4、5
    try:
        with open(args.current_population_docked_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        smi = parts[0]
                        ds = float(parts[1])
                        qed_v = float(parts[2]) if parts[2] != 'NA' else None
                        sa_v = float(parts[3]) if parts[3] != 'NA' else None
                        # 第五列可能是 RAG_Y，若没有则计算
                        if parts[4] != 'NA':
                            y_val = float(parts[4])
                        else:
                            y_val = None
                        if y_val is None and (qed_v is not None and sa_v is not None):
                            y_val = _ds_hat(ds) * float(qed_v) * _sa_hat(float(sa_v))
                        if y_val is not None:
                            y_all.append(y_val)
                        ds_all.append(ds)
                    except Exception:
                        continue
    except Exception:
        pass

    # 若当前文件没有附带 QED/SA，则退化为针对 Top-100 估计 y（使用上面已计算的 qed_scores/sa_scores 和对应的 DS）
    y_top = []
    if not y_all and smiles_for_scoring:
        for smi, qed_v, sa_v in zip(smiles_for_scoring, qed_scores, sa_scores):
            ds = smiles_to_ds.get(smi, None)
            if ds is None:
                continue
            try:
                y_top.append(_ds_hat(ds) * float(qed_v) * _sa_hat(float(sa_v)))
            except Exception:
                continue

    # 安全地处理可能包含特殊字符的文件名
    population_filename = os.path.basename(args.current_population_docked_file)
    initial_population_filename = os.path.basename(args.initial_population_file)    
    # 为了避免f-string格式化问题，使用传统的字符串格式化
    results = "Metrics for Population: {}\n".format(population_filename)
    results += "--------------------------------------------------\n"
    results += "Total molecules processed: {}\n".format(len(current_smiles_list))
    results += "Valid RDKit molecules for properties: {}\n".format(len(sorted_smiles))
    results += "Molecules with docking scores: {}\n".format(len(docking_scores))
    results += "--------------------------------------------------\n"    
    # y 分数统计输出
    if y_all:
        results += "RAG-Y (All) count={}: mean={:.6f}, max={:.6f}, min={:.6f}\n".format(
            len(y_all), float(np.mean(y_all)), float(np.max(y_all)), float(np.min(y_all))
        )
    if y_top:
        results += "RAG-Y (Top 100 subset) count={}: mean={:.6f}, max={:.6f}, min={:.6f}\n".format(
            len(y_top), float(np.mean(y_top)), float(np.max(y_top)), float(np.min(y_top))
        )
    if ds_all:
        results += "DS (All) mean={:.6f}, best(min)={:.6f}\n".format(float(np.mean(ds_all)), float(np.min(ds_all)))
    results += "--------------------------------------------------\n"
    # 处理浮点数格式化，注意处理NaN情况
    if np.isnan(top1_score): #top1
        results += "Docking Score - Top 1: N/A\n"
    else:
        results += "Docking Score - Top 1: {:.4f}\n".format(top1_score)
        
    if np.isnan(top10_mean_score): #top10
        results += "Docking Score - Top 10 Mean: N/A\n"
    else:
        results += "Docking Score - Top 10 Mean: {:.4f}\n".format(top10_mean_score)    

    if np.isnan(top100_mean_score): #top100
        results += "Docking Score - Top 100 Mean: N/A\n"
    else:
        results += "Docking Score - Top 100 Mean: {:.4f}\n".format(top100_mean_score)    
    results += "--------------------------------------------------\n"
    results += "Novelty (vs {}): {:.4f}\n".format(initial_population_filename, novelty)
    results += "Diversity (Top 100): {:.4f}\n".format(diversity)
    results += "--------------------------------------------------\n"    
    if np.isnan(mean_qed):
        results += "QED - {} Mean: N/A\n".format(score_description)
    else:
        results += "QED - {} Mean: {:.4f}\n".format(score_description, mean_qed)        
    if np.isnan(mean_sa):
        results += "SA Score - {} Mean: N/A\n".format(score_description)
    else:
        results += "SA Score - {} Mean: {:.4f}\n".format(score_description, mean_sa)    
    results += "--------------------------------------------------\n"    
    print_calculation_results(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")
if __name__ == "__main__":
    main()
