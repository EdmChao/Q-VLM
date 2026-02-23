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
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from navsim.common.dataclasses import SceneFilter

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
CONFIG_NAME = "diffusion_inference"

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

    # Hardcoded relaxed scene filter: 1 history + 1 future frame (2 frames total)
    # This overrides the composed Hydra `train_test_split.scene_filter` for quick testing.
    scene_filter_override = SceneFilter(
        num_history_frames=2,
        num_future_frames=1,
        frame_interval=1,
        has_route=False,
        include_synthetic_scenes=True,
    )

    scene_loader_inference = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter_override,
        #scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=agent.get_sensor_config(),
    )

    # Diagnostic guard: ensure resolved paths look correct and there are log files to load.
    resolved_navsim_log_path = Path(cfg.navsim_log_path)
    resolved_original_sensor_path = Path(cfg.original_sensor_path)
    resolved_synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    resolved_synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    print("Resolved paths:")
    print(" - navsim_log_path:", resolved_navsim_log_path)
    print(" - original_sensor_path:", resolved_original_sensor_path)
    print(" - synthetic_sensor_path:", resolved_synthetic_sensor_path)
    print(" - synthetic_scenes_path:", resolved_synthetic_scenes_path)
    print(" - SUBSCORE_PATH env:", os.getenv('SUBSCORE_PATH'))
    print(" - DP_PREDS env:", os.getenv('DP_PREDS'))

    if not resolved_navsim_log_path.exists():
        print(f"ERROR: navsim_log_path does not exist: {resolved_navsim_log_path}")
        print("Tip: for NavsimHard set navsim_log_path to <NAVSIMHARD_PATH>/openscene_meta_datas")
        raise SystemExit(1)

    try:
        num_logs = len(list(resolved_navsim_log_path.iterdir()))
    except Exception:
        num_logs = 0

    if num_logs == 0:
        print(f"ERROR: no log files found in navsim_log_path ({resolved_navsim_log_path}); found 0 files.")
        print("Check that you're pointing to the directory that contains the original log .pkl files (e.g. openscene_meta_datas).")
        raise SystemExit(1)

    # DEBUG: Check scene loader tokens before creating dataset
    print(f"\n[DEBUG] SceneFilter override: {scene_filter_override}")
    print(f"[DEBUG] SceneLoader tokens count: {len(scene_loader_inference.tokens)}")
    if len(scene_loader_inference.tokens) > 0:
        print(f"[DEBUG] First 5 tokens: {scene_loader_inference.tokens[:5]}")
    else:
        print("[DEBUG] WARNING: SceneLoader has NO tokens!")

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

    fb = agent.get_feature_builders()[0]
    print("[DEBUG] required seq_len:", fb._config.seq_len)

    dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)

    # DEBUG: Check dataset length before trainer
    print(f"\n[DEBUG] Dataset length: {len(dataset)}")
    print(f"[DEBUG] Dataloader params: {cfg.dataloader.params}")

    # Adjust trainer params to avoid PyTorch Lightning DDP sampler assertion when
    # the dataset is smaller than the number of GPU processes. In that case some
    # ranks would receive zero samples which triggers an AssertionError.
    trainer_params = dict(cfg.trainer.params) if cfg.get('trainer') else {}
    try:
        requested_devices = trainer_params.get('devices', None)
        if requested_devices is None:
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            devices_to_check = available_gpus if available_gpus > 0 else 1
        else:
            devices_to_check = int(requested_devices)
    except Exception:
        devices_to_check = 1

    dataset_len = len(dataset)
    if (devices_to_check > 1) and (dataset_len < devices_to_check):
        logger.warning(
            f"Dataset size ({dataset_len}) < devices ({devices_to_check}); forcing single-device trainer to avoid DDP sampler issues."
        )
        print(f"[DEBUG] DDP FIX TRIGGERED: Setting devices=1, removing strategy")
        trainer_params['devices'] = 1
        trainer_params.pop('strategy', None)
    else:
        print(f"[DEBUG] DDP fix NOT triggered: devices_to_check={devices_to_check}, dataset_len={dataset_len}")
    
    print(f"[DEBUG] Final trainer_params: {trainer_params}")
    print(f"[DEBUG] Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())
    predictions = trainer.predict(AgentLightningModule(agent=agent, combined=False), dataloader, return_predictions=True)

    # merge and save
    merged = {}
    # Predictions from PL can come back in a few formats depending on agent:
    # - a dict mapping token->result
    # - an iterable (list) of dicts
    # Be defensive and support both shapes.
    for proc_prediction in predictions:
        if isinstance(proc_prediction, dict):
            merged.update(proc_prediction)
        else:
            for d in proc_prediction:
                if isinstance(d, dict):
                    merged.update(d)
                else:
                    logger.warning(f"Skipping unexpected prediction item of type {type(d)}")
    # Resolve SUBSCORE_PATH env variable safely. It must be a filepath, not a directory.
    subscore_env = os.environ.get('SUBSCORE_PATH')
    if subscore_env is None:
        # default to local file if not provided
        out_path = Path.cwd() / f"subscore_{uuid.uuid4().hex}.pkl"
        print(f"[WARNING] SUBSCORE_PATH not set; writing to default {out_path}")
    else:
        out_path = Path(subscore_env)
        if out_path.is_dir() or str(subscore_env).endswith(os.path.sep):
            # treat as directory: create a file inside it
            out_path = out_path / f"subscore_{uuid.uuid4().hex}.pkl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(merged, f)

    try:
        size = out_path.stat().st_size
    except Exception:
        size = 'unknown'
    print(f"WROTE PICKLE: {out_path} len={len(merged)} size={size}")
    print("WROTE PICKLE:", out_path, "len:", len(merged), "size:", size)


if __name__ == "__main__":
    main()