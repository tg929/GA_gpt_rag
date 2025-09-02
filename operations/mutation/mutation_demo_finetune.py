#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import argparse
import random
import logging
from pathlib import Path
from typing import Dict, List
from contextlib import contextmanager
from tqdm import tqdm
from rdkit import Chem
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from autogrow.operators.mutation.smiles_click_chem.smiles_click_chem import SmilesClickChem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class MutationExecutor:
    """执行分子变异操作的类"""
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger
        self.mutation_config = self.config.get('mutation_finetune', {})
        
        # 从mutation_finetune配置块中直接读取路径
        rxn_library = self.mutation_config.get('rxn_library', 'all_rxns')
        rxn_library_file = self.mutation_config.get('rxn_library_file', '')
        function_group_library = self.mutation_config.get('function_group_library', '')
        complementary_mol_directory = self.mutation_config.get('complementary_mol_directory', '')       
        
        # 解析为绝对路径
        rxn_library_vars = [
            rxn_library,
            str(PROJECT_ROOT / rxn_library_file) if rxn_library_file else '',
            str(PROJECT_ROOT / function_group_library) if function_group_library else '',
            str(PROJECT_ROOT / complementary_mol_directory) if complementary_mol_directory else ''
        ]        
        
        self.filter_object_dict = {
            'Structure_check': lambda mol: mol is not None
        }        
        self.mutator = SmilesClickChem(rxn_library_vars, [], self.filter_object_dict)
    def run_mutation_generation(self, parent_smiles: List[str], additional_smiles: List[str] = None) -> List[str]:
        if additional_smiles is None:
            additional_smiles = []            
        # 从配置文件读取参数
        num_mutations = self.mutation_config.get('mutation_attempts', 40)
        max_attempts_multiplier = self.mutation_config.get('max_attempts_multiplier', 50)
        max_consecutive_failures_multiplier = self.mutation_config.get('max_consecutive_failures_multiplier', 2)
        enable_progress_bar = self.mutation_config.get('enable_progress_bar', True)        
        total_population = list(set(parent_smiles + additional_smiles))        
        if not total_population:
            self.logger.warning("种群为空，无法执行变异操作。")
            return []            
        mutation_results = []
        attempts = 0
        failed_molecules = set()
        max_attempts = num_mutations * max_attempts_multiplier
        consecutive_failures = 0
        max_consecutive_failures = len(total_population) * max_consecutive_failures_multiplier
        self.logger.info(f"开始变异操作: 目标生成 {num_mutations} 个新分子，种群大小 {len(total_population)}")        
        progress_bar = None
        if enable_progress_bar:
            progress_bar = tqdm(
                total=num_mutations, 
                desc="Generating mutations",
                unit="molecules",
                leave=True
            )
        try:
            while len(mutation_results) < num_mutations and attempts < max_attempts:
                attempts += 1                
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.warning(f"连续失败 {consecutive_failures} 次，可能种群中大部分分子无法变异，提前退出")
                    break                    
                available_molecules = [mol for mol in total_population if mol not in failed_molecules]
                if not available_molecules:
                    self.logger.warning("所有分子都尝试过且失败，无法继续生成新的变异分子")
                    break                    
                parent = random.choice(available_molecules)                
                try:
                    # 使用suppress_stdout_stderr来避免输出干扰
                    with suppress_stdout_stderr():
                        new_smiles_list = self.mutator.run_smiles_click2(parent)                    
                    if not new_smiles_list:
                        failed_molecules.add(parent)
                        consecutive_failures += 1
                        continue

                    consecutive_failures = 0
                    
                    for new_smi in new_smiles_list:
                        is_valid = all(check(new_smi) for check in self.filter_object_dict.values())
                        if (new_smi and is_valid and
                            new_smi not in mutation_results and 
                            new_smi not in total_population):
                            
                            mutation_results.append(new_smi)
                            self.logger.debug(f"成功生成新分子: {new_smi}")                            
                            # 更新进度条
                            if progress_bar:
                                progress_bar.update(1)
                                progress_bar.set_postfix({
                                    'success_rate': f"{len(mutation_results)}/{attempts}",
                                    'failed_mols': len(failed_molecules)
                                })
                            
                            if len(mutation_results) >= num_mutations:
                                break
                                
                    if len(mutation_results) >= num_mutations:
                        break
                        
                except Exception as e:
                    self.logger.debug(f"分子 {parent} 变异失败: {str(e)}")
                    failed_molecules.add(parent)
                    consecutive_failures += 1
                    continue
        
        finally:
            # 确保进度条被正确关闭
            if progress_bar:
                progress_bar.close()        
        unique_results = list(set(mutation_results))
        success_rate = len(unique_results) / attempts * 100 if attempts > 0 else 0
        
        self.logger.info(f"变异完成: 目标 {num_mutations}, 实际生成 {len(unique_results)} 个独特分子")
        self.logger.info(f"总尝试次数: {attempts}, 成功率: {success_rate:.1f}%, 无法变异的分子数: {len(failed_molecules)}")
        
        if len(unique_results) == 0:
            self.logger.warning("未能生成任何新分子，请检查输入分子是否包含可变异的官能团")
            
        return unique_results

def run_mutation_simple(config: Dict, parent_smiles: List[str], additional_smiles: List[str] = None) -> List[str]:
    """
    简化的变异操作函数
    """
    try:
        executor = MutationExecutor(config)
        return executor.run_mutation_generation(parent_smiles, additional_smiles or [])
    except Exception as e:
        logger.error(f"变异操作主函数出错: {e}", exc_info=True)
        return []

def main():
    parser = argparse.ArgumentParser(description='分子变异操作')
    parser.add_argument('--smiles_file', type=str, required=True, help='输入SMILES文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出SMILES文件路径')
    parser.add_argument('--config_file', type=str, default='GA_gpt/config_example.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"无法加载配置文件 {args.config_file}: {e}")
        return
    
    try:
        with open(args.smiles_file, 'r') as f:
            parent_smiles = [line.strip().split()[0] for line in f if line.strip()]
    except Exception as e:
        logger.error(f"无法读取SMILES文件 {args.smiles_file}: {e}")
        return
    
    if not parent_smiles:
        logger.warning("输入SMILES文件为空，无法执行变异操作")
        with open(args.output_file, 'w') as f: pass
        return
    
    parent_smiles = list(set(parent_smiles))
    logger.info(f"开始变异操作: {len(parent_smiles)} 个父代分子（已去重）")
    
    mutated_smiles = run_mutation_simple(config, parent_smiles)
    
    try:
        with open(args.output_file, 'w') as f:
            for smiles in mutated_smiles: f.write(f"{smiles}\n")
        logger.info(f"变异结果已保存到: {args.output_file} ({len(mutated_smiles)} 个分子)")
    except Exception as e:
        logger.error(f"无法保存结果到 {args.output_file}: {e}")

if __name__ == "__main__":
    main()

