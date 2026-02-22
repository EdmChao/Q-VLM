import logging
import os
import pickle
import traceback
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset


logger = logging.getLogger(__name__)

print("Loaded test_gen_traj_proposals.py")


# Hydra config used by the original runner
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    print("Starting test_gen_traj_proposals.main()")

    def setup_logger(cfg):
        log_level = getattr(cfg, "log_level", "INFO")
        log_format = "%(asctime)s %(levelname)s %(message)s"
        if hasattr(cfg, "log_file"):
            logging.basicConfig(filename=cfg.log_file, level=log_level, format=log_format)
        logging.basicConfig(level=log_level, format=log_format)

    # In your main function, call:
    setup_logger(cfg)
    print("Logger set up")
    combined = cfg.get('combined_inference', False)

    print(f'Combined inference: {combined}')
    print(f'cfg keys: {list(cfg.keys())}')
    dump_path = os.getenv('SUBSCORE_PATH')
    print(f'Subscore/Trajectories saved to {dump_path}')
    # gpu inference
    agent: AbstractAgent = instantiate(cfg.agent)
    print('Instantiated agent, initializing...')
    agent.initialize()
    print('Agent initialized')

    scene_loader_inference = SceneLoader(
    synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
    original_sensor_path=Path(cfg.original_sensor_path),
    data_path=Path(cfg.navsim_log_path),
    synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
    scene_filter=instantiate(cfg.train_test_split.scene_filter),
    sensor_config=agent.get_sensor_config(),
    )

    dataset = Dataset(
        scene_loader=scene_loader_inference,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False
    )

    #extract only single item from dataset for testing purposes
    single_index = os.getenv("SINGLE_INDEX")
    if single_index is not None:
        try:
            idx = int(single_index)
            from torch.utils.data import Subset
            dataset = Subset(dataset, [idx])
            logger.info(f"Using single dataset index {idx}")
        except Exception:
            logger.warning(f"Invalid SINGLE_INDEX={single_index}; running full dataset")
        
    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(AgentLightningModule(agent=agent, combined=False), dataloader, return_predictions=True)

    # merge and save
    merged = {}
    for proc_prediction in predictions:
        for d in proc_prediction:
            merged.update(d)
    pickle.dump(merged, open(os.environ['SUBSCORE_PATH'], 'wb'))
    try:
        size = os.path.getsize(os.environ['SUBSCORE_PATH'])
    except Exception:
        size = 'unknown'
    print(f"WROTE PICKLE: {os.environ.get('SUBSCORE_PATH')} len={len(merged)} size={size}")
    print("WROTE PICKLE:", os.environ['SUBSCORE_PATH'], "len:", len(merged), "size:", os.path.getsize(os.environ['SUBSCORE_PATH']))