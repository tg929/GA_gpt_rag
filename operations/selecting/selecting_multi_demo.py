#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于NSGA-II帕累托算法的多目标分子选择脚本
同时优化对接分数、QED分数和SA分数
目标优化方向：
- 对接分数：最小化（越小越好）
- QED分数:最大化(转换为最小化 -QED)
- SA分数:最小化(越小越好)
"""
import argparse
import os
import sys
import numpy as np
from rdkit import Chem
import logging
from typing import List, Dict
from tdc import Oracle
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 使用TDC Oracle进行分子属性评估
qed_evaluator = Oracle('qed')
sa_evaluator = Oracle('sa')
logger.info("已初始化TDC Oracle用于QED和SA分数计算。")

def non_dominated_sort(objectives):
    """
    执行非支配排序    
    Args:
        objectives (np.array): 一个 (n_molecules, n_objectives) 的数组    
    Returns:
        list of lists: 每个子列表包含属于同一帕累托前沿的分子索引
    """
    n_points, n_obj = objectives.shape
    fronts = []
    domination_counts = np.zeros(n_points, dtype=int)
    dominated_solutions = [[] for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # 检查i是否支配j
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                domination_counts[j] += 1
                dominated_solutions[i].append(j)
            # 检查j是否支配i
            elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                domination_counts[i] += 1
                dominated_solutions[j].append(i)
    front_1 = np.where(domination_counts == 0)[0].tolist()
    fronts.append(front_1)
    k = 0
    while len(fronts[k]) > 0:
        next_front = []
        for i in fronts[k]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    return fronts[:-1] # 最后一个是空列表
def crowding_distance(objectives, front):
    """
    计算一个前沿内的拥挤度距离
    """
    n_points = len(front)
    if n_points == 0:
        return np.array([])      
    distances = np.zeros(n_points)
    obj_array = objectives[front, :]
    n_obj = objectives.shape[1]
    for m in range(n_obj):
        sorted_indices = np.argsort(obj_array[:, m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
        if n_points > 2:
            obj_min = obj_array[sorted_indices[0], m]
            obj_max = obj_array[sorted_indices[-1], m]            
            if obj_max == obj_min:
                continue
            for i in range(1, n_points - 1):
                distances[sorted_indices[i]] += (obj_array[sorted_indices[i+1], m] - obj_array[sorted_indices[i-1], m]) / (obj_max - obj_min)
    return distances

def select_molecules_nsga2(molecules_data: List[Dict], n_select: int) -> List[Dict]:
    """
    使用非支配排序和拥挤度距离进行多目标分子选择 (NSGA-II的核心思想)    
    Args:
        molecules_data: 包含所有候选分子及其分数的列表
        n_select: 要选择的分子数量        
    Returns:
        选中的分子列表
    """
    if not molecules_data:
        logger.warning("分子数据为空，无法执行选择。")
        return []
    # 目标：1.对接分数(min), 2.-QED(min), 3.SA分数(min)
    objectives = np.array([
        [m['docking_score'], -m.get('qed_score', 0), m.get('sa_score', 10)] 
        for m in molecules_data
    ])    
    # 执行非支配排序
    fronts = non_dominated_sort(objectives)    
    selected_molecules = []
    selected_indices = []    
    for front in fronts:
        if len(selected_molecules) + len(front) <= n_select:
            selected_indices.extend(front)
            selected_molecules.extend([molecules_data[i] for i in front])
        else:
            # 如果当前前沿无法完全加入，则使用拥挤度距离选择
            distances = crowding_distance(objectives, front)
            sorted_by_crowding = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)            
            remaining_needed = n_select - len(selected_molecules)
            for i in range(remaining_needed):
                idx = sorted_by_crowding[i][0]
                selected_indices.append(idx)
                selected_molecules.append(molecules_data[idx])
            break # 已选够数量            
    logger.info(f"多目标选择完成：从 {len(molecules_data)} 个候选中选出 {len(selected_molecules)} 个分子。")
    return selected_molecules
def load_molecules_with_scores(docked_file):
    """
    从对接结果文件中加载分子及其分数    
    Args:
        docked_file: 对接结果文件路径，格式为 "SMILES score" 或 "SMILES\tscore"
    Returns:
        list: 分子数据列表,每个元素包含SMILES和对接分数
    """
    molecules = []    
    try:
        with open(docked_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    # 支持空格或制表符分隔
                    parts = line.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        smiles = parts[0]
                        try:
                            docking_score = float(parts[1])
                            molecules.append({
                                'smiles': smiles,
                                'docking_score': docking_score
                            })
                        except ValueError:
                            logger.warning(f"第{line_num}行: 无法解析分数 '{parts[1]}' for SMILES {smiles}")
                    else:
                        logger.warning(f"第{line_num}行: 格式不正确，跳过: {line}")
    except FileNotFoundError:
        logger.error(f"错误: 找不到文件 {docked_file}")
        return []    
    return molecules

def load_molecules_from_combined_files(parent_file: str, docked_file: str):
    """
    合并父代和子代分子数据进行统一选择，自动去重
    
    Args:
        parent_file: 父代文件路径 (格式: SMILES score)
        docked_file: 子代对接结果文件路径 (格式: SMILES score)
        
    Returns:
        list: 合并并去重后的分子数据列表
    """
    # 加载父代分子
    parent_molecules = load_molecules_with_scores(parent_file)
    logger.info(f"从父代文件加载了 {len(parent_molecules)} 个分子")
    
    # 加载子代分子  
    offspring_molecules = load_molecules_with_scores(docked_file)
    logger.info(f"从子代文件加载了 {len(offspring_molecules)} 个分子")
    
    # 合并并去重：使用字典以SMILES为key，保留分数更好的分子
    molecules_dict = {}
    total_before_dedup = len(parent_molecules) + len(offspring_molecules)
    
    # 先添加父代分子
    for mol in parent_molecules:
        smiles = mol['smiles']
        score = mol['docking_score']
        if smiles not in molecules_dict or score < molecules_dict[smiles]['docking_score']:
            molecules_dict[smiles] = mol
    
    # 然后添加子代分子（保留更好分数的）
    for mol in offspring_molecules:
        smiles = mol['smiles']
        score = mol['docking_score']
        if smiles not in molecules_dict or score < molecules_dict[smiles]['docking_score']:
            molecules_dict[smiles] = mol
    
    all_molecules = list(molecules_dict.values())
    duplicates_removed = total_before_dedup - len(all_molecules)
    
    logger.info(f"合并前总数: {total_before_dedup}, 去重后: {len(all_molecules)}, 删除重复: {duplicates_removed}")
    logger.info(f"去重率: {duplicates_removed/total_before_dedup*100:.1f}%")
    
    return all_molecules

def add_additional_scores(molecules: List[Dict]) -> List[Dict]:
    """
    使用TDC Oracle为分子批量添加QED和SA分数。

    Args:
        molecules: 包含分子SMILES的字典列表。

    Returns:
        更新了QED和SA分数的分子列表。
    """
    if not molecules:
        return []

    logger.info(f"开始为 {len(molecules)} 个分子批量计算QED和SA分数...")
    
    # 提取所有SMILES用于批量处理
    smiles_list = [m['smiles'] for m in molecules]
    
    # 使用TDC进行批量计算
    qed_scores = qed_evaluator(smiles_list)
    sa_scores = sa_evaluator(smiles_list)
    
    # 将计算出的分数分配回每个分子字典
    for i, mol_data in enumerate(molecules):
        mol_data['qed_score'] = qed_scores[i]
        mol_data['sa_score'] = sa_scores[i]
        
    logger.info(f"完成所有 {len(molecules)} 个分子的分数计算。")
    return molecules

def save_selected_molecules(selected_molecules, output_file):
    """保存选中的分子到文件(仅SMILES格式)"""
    with open(output_file, 'w') as f:
        for mol_data in selected_molecules:
            f.write(f"{mol_data['smiles']}\n")
    
    logger.info(f"已保存 {len(selected_molecules)} 个选中的分子到 {output_file}")

def save_selected_molecules_with_scores(selected_molecules, output_file):
    """
    保存选中的分子及其完整分数信息到文件，按对接分数排序（分数越低越好排在前面）
    
    Args:
        selected_molecules: 选中的分子数据列表
        output_file: 输出文件路径
        
    输出格式: SMILES\tdocking_score\tqed_score\tsa_score
    """
    # 按对接分数排序（分数越低越好，升序排列）
    sorted_molecules = sorted(selected_molecules, key=lambda mol: mol['docking_score'])
    
    with open(output_file, 'w') as f:
        for mol_data in sorted_molecules:
            smiles = mol_data['smiles']
            docking = mol_data['docking_score']
            qed = mol_data.get('qed_score', 0.0)
            sa = mol_data.get('sa_score', 5.0)
            f.write(f"{smiles}\t{docking:.4f}\t{qed:.4f}\t{sa:.4f}\n")
    
    logger.info(f"已保存 {len(sorted_molecules)} 个选中的分子及分数(已按对接分数排序)到 {output_file}")

def print_selection_statistics(selected_molecules):
    """打印选择统计信息"""
    if not selected_molecules:
        print("没有选择任何分子")
        return    
    docking_scores = [mol['docking_score'] for mol in selected_molecules]
    qed_scores = [mol['qed_score'] for mol in selected_molecules]
    sa_scores = [mol['sa_score'] for mol in selected_molecules]
    
    print("\n========== 选择统计信息 ==========")
    print(f"选中分子数量: {len(selected_molecules)}")
    print(f"对接分数 - 最优: {min(docking_scores):.4f}, 平均: {np.mean(docking_scores):.4f}")
    print(f"QED分数 - 最优: {max(qed_scores):.4f}, 平均: {np.mean(qed_scores):.4f}")
    print(f"SA分数 - 最优: {min(sa_scores):.4f}, 平均: {np.mean(sa_scores):.4f}")
    print("="*40)
def main():
    parser = argparse.ArgumentParser(description='基于NSGA-II的多目标分子选择')    
    # 输入输出参数
    parser.add_argument('--docked_file', type=str, required=True,
                       help='子代对接结果文件路径（格式: SMILES score)')
    parser.add_argument('--parent_file', type=str, required=False,
                       help='父代文件路径（格式: SMILES score)')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出的下一代父代文件路径')    
    # 选择参数
    parser.add_argument('--n_select', type=int, required=True,
                       help='要选择的分子数量')    
    # 输出格式选择
    parser.add_argument('--output_format', type=str, choices=['smiles_only', 'with_scores'], 
                       default='with_scores',
                       help='输出格式:仅SMILES或包含分数 (默认: with_scores)')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='显示详细信息')    
    args = parser.parse_args()    
    
    logger.info("开始基于NSGA-II的多目标分子选择...")
    logger.info(f"子代文件: {args.docked_file}")
    if args.parent_file:
        logger.info(f"父代文件: {args.parent_file}")
    logger.info(f"输出文件: {args.output_file}")    
    
    # 1. 加载分子数据
    if args.parent_file:
        # 如果提供了父代文件，合并父代和子代
        molecules = load_molecules_from_combined_files(args.parent_file, args.docked_file)
    else:
        # 仅使用子代文件
        molecules = load_molecules_with_scores(args.docked_file)
        logger.info(f"仅使用子代文件，加载了 {len(molecules)} 个分子")
    
    if not molecules:
        logger.error("错误: 无法加载分子数据")
        return    
    
    # 2. 计算QED和SA分数
    molecules = add_additional_scores(molecules)    
    
    # 3. 使用NSGA-II进行多目标选择
    selected_molecules = select_molecules_nsga2(molecules, args.n_select)    
    
    # 4. 保存结果
    if selected_molecules:
        if args.output_format == 'smiles_only':
            save_selected_molecules(selected_molecules, args.output_file)
        else:
            save_selected_molecules_with_scores(selected_molecules, args.output_file)
        print_selection_statistics(selected_molecules)
    else:
        logger.error("错误: 未选择任何分子")

if __name__ == "__main__":
    main()
