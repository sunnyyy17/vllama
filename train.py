"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import vllama.tasks as tasks
from vllama.common.config import Config
from vllama.common.dist_utils import get_rank, init_distributed_mode
from vllama.common.logger import setup_logger
from vllama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from vllama.common.registry import registry
from vllama.common.utils import now

# imports modules for registration
from vllama.datasets.builders import *
from vllama.models import *
from vllama.processors import *
from vllama.runners import *
from vllama.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    #os.environ['MASTER_ADDR'] = '127.0.0.2'
    #os.environ['MASTER_PORT'] = '39500'
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "0" 
    #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    print("datasets keys", datasets.keys())
    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
