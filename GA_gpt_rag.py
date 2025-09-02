#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_gpt_rag.py
==============
Entry point for running the GA-GPT workflow using the RAG-score based
selection strategy. All other stages and scripts are identical to the
finetune pipeline; only the selection step is swapped.
"""

import os
import sys
import argparse
import json
import multiprocessing
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from operations.operations_execute_GAgpt_rag import GAGPTRAGWorkflowExecutor
from utils.cpu_utils import get_available_cpu_cores, calculate_optimal_workers


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GA_GPT_RAG_MAIN")


def run_workflow_for_receptor(config_path: str, receptor_name: str, output_dir: str, num_processors: int):
    try:
        executor = GAGPTRAGWorkflowExecutor(
            config_path=config_path,
            receptor_name=receptor_name,
            output_dir_override=output_dir,
            num_processors_override=num_processors
        )
        return receptor_name or 'default', executor.run_complete_workflow()
    except Exception as e:
        logger.exception("子流程异常: %s", e)
        return receptor_name or 'default', False


def main():
    parser = argparse.ArgumentParser(description='GA-GPT with RAG-score selection')
    parser.add_argument('--config', type=str, default='GA_gpt/config_GA_gpt_rag.json')
    parser.add_argument('--receptor', type=str, default=None)
    parser.add_argument('--all_receptors', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    receptors_to_run = []
    if args.all_receptors:
        receptors_to_run = list(config.get('receptors', {}).get('target_list', {}).keys())
    else:
        receptors_to_run.append(args.receptor)

    perf = config.get('performance', {})
    parallel_enabled = perf.get('parallel_processing')
    max_workers_config = perf.get('max_workers')
    inner_processors_config = perf.get('number_of_processors')

    if not parallel_enabled or len(receptors_to_run) <= 1:
        if inner_processors_config == -1:
            available_cores, _ = get_available_cpu_cores()
            cores_per = available_cores
        else:
            cores_per = inner_processors_config
        results = [run_workflow_for_receptor(args.config, r, args.output_dir, cores_per) for r in receptors_to_run]
    else:
        available_cores, _ = get_available_cpu_cores()
        if max_workers_config == -1 and inner_processors_config == -1:
            max_workers, cores_per = calculate_optimal_workers(
                target_count=len(receptors_to_run), available_cores=available_cores, cores_per_worker=-1
            )
        elif max_workers_config == -1:
            max_workers = min(len(receptors_to_run), available_cores // inner_processors_config)
            cores_per = inner_processors_config
        elif inner_processors_config == -1:
            max_workers = min(max_workers_config, len(receptors_to_run))
            cores_per = max(1, available_cores // max_workers)
        else:
            max_workers = min(max_workers_config, len(receptors_to_run))
            cores_per = inner_processors_config

        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_workflow_for_receptor, args.config, r, args.output_dir, cores_per): r for r in receptors_to_run}
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as e:
                    logger.exception("并行任务异常: %s", e)
                    results.append((futs[fut], False))

    success = [r for r, ok in results if ok]
    failed = [r for r, ok in results if not ok]
    logger.info("RAG选择流程完成, 成功=%s, 失败=%s", success, failed)
    raise SystemExit(0 if not failed else 1)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()


