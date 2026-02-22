"""Lightweight, no-Hydra inference runner to generate DP trajectory proposals.

Usage: set environment variables and run directly:

export CKPT_PATH=/full/path/to/ckpt.ckpt
export NAVSIM_LOG_PATH=/path/to/navsim/logs
export SUBSCORE_PATH=/path/to/output.pkl  # optional
export SINGLE_INDEX=0  # optional, loads one dataset item
python navsim/planning/script/test_gen_traj_proposals_nohydra.py

This script mirrors the inference part of the Hydra script but avoids Hydra and config parsing.
"""
import os
import pickle
from pathlib import Path
import logging

from omegaconf import OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
# try importing DP agent and config for common case
try:
    from navsim.agents.dp.dp_agent import DPAgent
    from navsim.agents.dp.dp_config import DPConfig
except Exception:
    DPAgent = None
    DPConfig = None

from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nohydra_dp_infer")


def main():
    ckpt = os.environ.get("CKPT_PATH")
    if not ckpt:
        raise RuntimeError("CKPT_PATH not set; export CKPT_PATH=/path/to/checkpoint.ckpt")

    navsim_log = os.environ.get("NAVSIM_LOG_PATH") or os.environ.get("DATASET_ROOT")
    if not navsim_log:
        raise RuntimeError("NAVSIM_LOG_PATH or DATASET_ROOT must be set to point to navsim logs")

    out_path = os.environ.get("SUBSCORE_PATH")
    if not out_path:
        out_path = str(Path(os.environ.get("NAVSIM_EXP_ROOT", ".")) / "dp_proposals.pkl")

    single_index = os.environ.get("SINGLE_INDEX")
    batch_size = int(os.environ.get("DATALOADER_BATCH_SIZE", "1"))

    print("No-Hydra DP inference runner")
    print("CKPT", ckpt)
    print("NAVSIM LOG", navsim_log)
    print("OUT", out_path)
    print("SINGLE_INDEX", single_index, "BATCH_SIZE", batch_size)

    # Build agent
    agent = None
    if DPAgent is not None and DPConfig is not None:
        cfg = DPConfig()
        agent = DPAgent(config=cfg, lr=1e-4, checkpoint_path=ckpt)
        try:
            agent.initialize()
        except Exception as e:
            print("Agent.initialize() failed:", e)
            raise
    else:
        raise RuntimeError("DPAgent or DPConfig not importable; Hydrified agent not supported by this runner")

    # SceneLoader + Dataset
    scene_filter = SceneFilter()
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(navsim_log),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
        is_training=False,
    )

    if single_index is not None:
        try:
            idx = int(single_index)
            from torch.utils.data import Subset
            dataset = Subset(dataset, [idx])
            print(f"Using single dataset index {idx}")
        except Exception as e:
            logger.warning(f"Invalid SINGLE_INDEX={single_index}; running full dataset: {e}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Trainer: CPU by default; override via env if needed
    trainer_kwargs = {}
    use_gpu = os.environ.get("USE_GPU", "0") == "1"
    if use_gpu:
        trainer_kwargs.update({"accelerator": "gpu", "devices": 1})
    else:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

    trainer = pl.Trainer(**trainer_kwargs)

    predictions = trainer.predict(AgentLightningModule(agent=agent, combined=False), dataloader, return_predictions=True)

    # merge
    merged = {}
    for proc_prediction in predictions:
        for d in proc_prediction:
            merged.update(d)

    # ensure parent dir exists
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "wb") as f:
        pickle.dump(merged, f)

    print(f"WROTE PICKLE: {outp} len={len(merged)} size={outp.stat().st_size}")


if __name__ == "__main__":
    main()
