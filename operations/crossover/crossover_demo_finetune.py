import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import random
import json
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge 
import autogrow.operators.crossover.execute_crossover as execute_crossover
import autogrow.operators.filter.execute_filters as Filter

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("crossover")
def main():
    parser = argparse.ArgumentParser(description='改进的GA交叉参数')
    parser.add_argument("--smiles_file", "-s", type=str, required=True,
                      help="输入SMILES文件路径")
    parser.add_argument("--output_file", "-o", type=str, 
                      default=os.path.join(PROJECT_ROOT, "output/generation_crossover_0.smi"),
                      help="输出文件路径")
    parser.add_argument('--config_file', type=str, default='GA_gpt/config_example.json', 
                      help='配置文件路径')    
    args = parser.parse_args()    
    # 加载配置
    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        crossover_config = config['crossover_finetune']
    except (FileNotFoundError, KeyError) as e:
        print(f"错误：无法从 {args.config_file} 加载交叉配置: {e}")
        return
    # 设置日志
    logger = setup_logging()
    logger.info("开始交叉操作")    
    # 加载SMILES文件
    all_smiles = []
    with open(args.smiles_file, 'r') as f:
        all_smiles = [line.split()[0].strip() for line in f if line.strip()]
        logger.info(f"加载分子数量: {len(all_smiles)}")    
    initial_population = list(set(all_smiles))    
    # 从配置中读取交叉参数
    vars = {
        'min_atom_match_mcs': crossover_config.get('min_atom_match_mcs', 4),
        'max_time_mcs_prescreen': crossover_config.get('max_time_mcs_prescreen', 1),
        'max_time_mcs_thorough': crossover_config.get('max_time_mcs_thorough', 1),
        'protanate_step': crossover_config.get('protanate_step', True),
        'number_of_crossovers': crossover_config.get('crossover_attempts', 20),
        'filter_object_dict': {},  # 过滤器配置（如果需要）
        'max_variants_per_compound': crossover_config.get('max_variants_per_compound', 1),
        'debug_mode': crossover_config.get('debug_mode', False),
        'gypsum_timeout_limit': crossover_config.get('gypsum_timeout_limit', 120.0),
        'gypsum_thoroughness': crossover_config.get('gypsum_thoroughness', 3)
    }    
    crossover_attempts = vars['number_of_crossovers']
    max_attempts_multiplier = crossover_config.get('max_attempts_multiplier', 10)
    merge_attempts = crossover_config.get('merge_attempts', 3)
    
    logger.info(f"开始交叉操作，本轮目标生成 {crossover_attempts} 个新分子")
    crossed_population = []
    attempts = 0
    max_attempts = crossover_attempts * max_attempts_multiplier
    
    while len(crossed_population) < crossover_attempts and attempts < max_attempts:
        attempts += 1
        try:
            parent1, parent2 = random.sample(initial_population, 2)
            mol1 = execute_crossover.convert_mol_from_smiles(parent1)
            mol2 = execute_crossover.convert_mol_from_smiles(parent2)
            if mol1 is None or mol2 is None: continue
                
            mcs_result = execute_crossover.test_for_mcs(vars, mol1, mol2)
            if mcs_result is None: continue
                
            ligand_new_smiles = None
            for _ in range(merge_attempts):
                ligand_new_smiles = smiles_merge.run_main_smiles_merge(vars, parent1, parent2)
                if ligand_new_smiles is not None: break
                    
            if ligand_new_smiles is None: continue
                
            if Filter.run_filter_on_just_smiles(ligand_new_smiles, vars['filter_object_dict']):
                crossed_population.append(ligand_new_smiles)
                
        except Exception as e:
            logger.warning(f"交叉操作出错: {str(e)}")
            continue
            
    if attempts >= max_attempts:
        logger.warning(f"达到最大尝试次数 {max_attempts}，但只生成了 {len(crossed_population)} 个有效分子")
        
    logger.info(f"本轮交叉实际生成 {len(crossed_population)} 个新分子，尝试次数: {attempts}")
    
    # 保存最终结果 (只保存新交叉生成的分子)
    with open(args.output_file, 'w') as f:
        for smi in crossed_population:
            f.write(f"{smi}\n")
    logger.info(f"最终结果已保存至: {args.output_file} (仅包含新生成的分子)")

if __name__ == "__main__":
    main() 