import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import pytorch_lightning as pl
import cv2
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.cluster import KMeans

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.visualization.camera import _transform_pcs_to_images

logger = logging.getLogger(__name__)


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "diffusion_to_projection"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        # instantiate agent and scene loader similar to the test runner
        agent = instantiate(cfg.agent)
        agent.initialize()

        scene_filter_override = SceneFilter(
            num_history_frames=2,
            num_future_frames=1,
            frame_interval=1,
            has_route=False,
            include_synthetic_scenes=True,
        )

        scene_loader = SceneLoader(
            synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
            original_sensor_path=Path(cfg.original_sensor_path),
            data_path=Path(cfg.navsim_log_path),
            synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
            scene_filter=scene_filter_override,
            sensor_config=agent.get_sensor_config(),
        )

        # Build dataset but restrict to a single sample (first token)
        dataset = Dataset(
            scene_loader=scene_loader,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            cache_path=None,
            force_cache_computation=False,
            append_token_to_batch=True,
            is_training=False,
        )

        if len(dataset) == 0:
            raise SystemExit("Dataset empty - nothing to run")

        # restrict to first item only for quick test (or use all, based on cfg.generate_count)
        from torch.utils.data import Subset, DataLoader
        fb = agent.get_feature_builders()[0]

        subset = None
        gen_count = str(cfg.get('generate_count', 'one')).lower()
        if gen_count == 'one':
            print("Extracting first element from dataset for testing!")
            subset = Subset(dataset, [0])
            dataloader = DataLoader(subset, batch_size=1, shuffle=False)
        elif gen_count == 'all':
            print("Processing all samples in dataset")
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        else:
            print(f"Unknown generate_count '{gen_count}'; defaulting to 'one'")
            subset = Subset(dataset, [0])
            dataloader = DataLoader(subset, batch_size=1, shuffle=False)

        # Ensure trainer devices configuration is compatible with available hardware and dataset size.
        trainer_params = dict(cfg.trainer.params) if cfg.get('trainer') else {}

        # debug: will print resolved trainer params after resolving devices/strategy
        try:
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            available_gpus = 0

        try:
            requested_devices = trainer_params.get('devices', None)
            if requested_devices is None:
                # default: use all GPUs if available else CPU (1)
                devices_to_check = available_gpus if available_gpus > 0 else 1
            elif isinstance(requested_devices, str) and requested_devices.lower() in ('auto', 'all'):
                devices_to_check = available_gpus if available_gpus > 0 else 1
            else:
                devices_to_check = int(requested_devices)
                if available_gpus > 0:
                    devices_to_check = min(devices_to_check, available_gpus)
                if devices_to_check <= 0:
                    devices_to_check = 1
            # record resolved device count back into trainer params so Trainer uses it
            trainer_params['devices'] = devices_to_check
        except Exception:
            devices_to_check = 1
            trainer_params['devices'] = 1

        subset_len = len(subset) if subset is not None else len(dataset)
        if (devices_to_check > 1) and (subset_len < devices_to_check):
            # force single-device trainer to avoid distributed sampler errors
            print(f"Dataset size ({subset_len}) < devices ({devices_to_check}); forcing devices=1 to avoid DDP sampler issues.")
            trainer_params['devices'] = 1
            trainer_params.pop('strategy', None)
        # If resolved devices == 1, ensure no distributed strategy is used
        if trainer_params.get('devices', 1) <= 1:
            trainer_params.pop('strategy', None)

        # create trainer and run prediction; guard against distributed failures
        print("Resolved trainer_params:", trainer_params)
        trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())

        try:
            predictions = trainer.predict(AgentLightningModule(agent=agent, combined=False), dataloader, return_predictions=True)
        except Exception:
            # try to clean up distributed state if something went wrong
            traceback.print_exc()
            try:
                if torch.distributed.is_initialized():
                    try:
                        torch.distributed.destroy_process_group()
                    except Exception:
                        pass
            except Exception:
                pass
            raise

        # merge predictions as in test script
        merged: Dict[str, Dict] = {}
        for proc_prediction in predictions:
            if isinstance(proc_prediction, dict):
                merged.update(proc_prediction)
            else:
                for d in proc_prediction:
                    if isinstance(d, dict):
                        merged.update(d)

        if len(merged) == 0:
            print("No predictions produced")
            return
        
        print("proposal predictions created")

        # decide which tokens to process based on generate_count
        gen_count = str(cfg.get('generate_count', 'one')).lower()
        if gen_count == 'all':
            tokens_to_process = list(merged.keys())
        else:
            tokens_to_process = [list(merged.keys())[0]]

        for token in tokens_to_process:
            result = merged[token]

            # extract dp proposals (look for 'dp_pred')
            dp_pred = None
            if isinstance(result, dict):
                dp_pred = result.get('dp_pred', None)
                if dp_pred is None:
                    # fallback: find first array-like
                    for v in result.values():
                        if hasattr(v, 'shape'):
                            dp_pred = v
                            break

            if dp_pred is None:
                print(f'No trajectory proposals found in prediction for token {token}')
                continue

            print(f"normalizing proposals for token {token}")

            # convert to numpy and normalize shape to (N, H, D)
            if hasattr(dp_pred, 'cpu'):
                dp_np = dp_pred.cpu().numpy()
            else:
                dp_np = np.array(dp_pred)
            if dp_np.ndim == 4:
                dp_np = dp_np[0]

            N, HORIZON, DIM = dp_np.shape
            print(f'Found {N} proposals for token {token}; selecting 5 exemplars')

            k = cfg.k
            sel_method = cfg.selection_method.lower()
            if k > N:
                k = N
            if sel_method == 'kmeans' and N >= k:
                traj_flat = dp_np.reshape(N, -1)
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(traj_flat)
                centers = kmeans.cluster_centers_.reshape(k, HORIZON, DIM)
            elif sel_method in ('ff', 'farthest_first'):
                # greedy farthest-first (k-center) selection on flattened trajectories
                print(f"Using farthest-first sampling for trajectories (method={sel_method})")
                traj_flat = dp_np.reshape(N, -1)
                rng = np.random.RandomState(0)
                if N <= k:
                    centers = dp_np.copy()
                else:
                    # start from a random seed index for determinism
                    first_idx = int(rng.randint(0, N))
                    selected = [first_idx]
                    for _ in range(1, k):
                        dists = np.linalg.norm(traj_flat[:, None, :] - traj_flat[selected][None, :, :], axis=2)
                        min_dists = np.min(dists, axis=1)
                        min_dists[selected] = -1.0
                        next_idx = int(np.argmax(min_dists))
                        selected.append(next_idx)
                    centers = dp_np[selected]
            else:
                # fallback to random sampling to encourage diverse exemplars
                print(f"Using random sampling for trajectories (method={sel_method})")
                rng = np.random.RandomState(0)
                if N >= k:
                    idxs = rng.choice(N, size=k, replace=False)
                else:
                    idxs = np.arange(N)
                centers = dp_np[idxs]

        # load the scene/frame and reconstruct front-stitched image using same crops as feature builder
        scene = scene_loader.get_scene_from_token(scene_loader.tokens[0])
        frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[frame_idx]

        cam_l0 = frame.cameras.cam_l0
        cam_f0 = frame.cameras.cam_f0
        cam_r0 = frame.cameras.cam_r0

        # cropping logic matches hydra_features._get_camera_feature
        def crop_cam(img, left_right_crop=False):
            if img is None:
                return None
            if left_right_crop:
                return img[28:-28, 416:-416]
            else:
                return img[28:-28]

        l0_crop = crop_cam(cam_l0.image, left_right_crop=True)
        f0_crop = crop_cam(cam_f0.image, left_right_crop=False)
        r0_crop = crop_cam(cam_r0.image, left_right_crop=True)

        if l0_crop is None or f0_crop is None or r0_crop is None:
            print('Missing camera images; cannot create stitched overlay')
            return

        stitched = np.concatenate([l0_crop, f0_crop, r0_crop], axis=1)
        cam_w = fb._config.camera_width
        cam_h = fb._config.camera_height
        stitched_resized = cv2.resize(stitched, (cam_w, cam_h))

        # pre-resize tile widths and offsets
        w_l, w_f, w_r = l0_crop.shape[1], f0_crop.shape[1], r0_crop.shape[1]
        stitched_w = w_l + w_f + w_r
        stitched_h = stitched.shape[0]
        scale_x = cam_w / stitched_w
        scale_y = cam_h / stitched_h
        offsets = [0, w_l, w_l + w_f]

        # helper to project a ground-plane ego (x,y) to stitched_resized pixel coords
        def project_to_stitched(xy):
            # Create a lidar_pc-like array expected by _transform_pcs_to_images: shape (6, n)
            lidar_pc = np.zeros((6, 1), dtype=np.float32)
            lidar_pc[0, 0] = xy[0]
            lidar_pc[1, 0] = xy[1]
            lidar_pc[2, 0] = 0.0

            # iterate front cameras with their crop offsets
            cams = [
                (cam_l0, offsets[0], True),
                (cam_f0, offsets[1], False),
                (cam_r0, offsets[2], True),
            ]

            for cam, x_off, lr_crop in cams:
                if cam.intrinsics is None or cam.sensor2lidar_rotation is None or cam.sensor2lidar_translation is None:
                    continue
                intr = np.array(cam.intrinsics)
                rot = np.array(cam.sensor2lidar_rotation)
                trans = np.array(cam.sensor2lidar_translation)
                # pass original full image shape so in_fov masking is correct
                img_h_full, img_w_full = cam.image.shape[:2]
                pc_img, in_fov = _transform_pcs_to_images(lidar_pc, rot, trans, intr, img_shape=(img_h_full, img_w_full))
                if in_fov[0]:
                    u_full, v_full = pc_img[0]
                    # apply crop offsets used in feature builder
                    u_crop = u_full - (416 if lr_crop else 0)
                    v_crop = v_full - 28
                    # stitched coordinates before resize
                    stitched_x = u_crop + x_off
                    stitched_y = v_crop
                    px = int(np.round(stitched_x * scale_x))
                    py = int(np.round(stitched_y * scale_y))
                    return px, py
            return None

        out_img = stitched_resized.copy()
        if out_img.dtype != np.uint8:
            out_img = out_img.astype(np.uint8)


        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128), (128, 128, 0), (64, 128, 192)]
        color_names = [
            "red",
            "green",
            "blue",
            "yellow",
            "magenta",
            "cyan",
            "purple",
            "teal",
            "olive",
            "steelblue",
        ]

        polyline_strings = []

        for i in range(k):
            pts = []
            for t in range(HORIZON):
                xy = centers[i, t][:2]
                p = project_to_stitched(xy)
                if p is not None:
                    pts.append(p)
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(out_img, [pts_arr], isClosed=False, color=colors[i % len(colors)], thickness=2)
            elif len(pts) == 1:
                cv2.circle(out_img, tuple(pts[0]), 3, colors[i % len(colors)], -1)

            # build string representation for this trajectory
            if len(pts) > 0:
                coord_str = ";".join([f"{int(x)},{int(y)}" for (x, y) in pts])
            else:
                coord_str = ""
            color_str = color_names[i % len(color_names)]
            polyline_strings.append(f"{color_str}: {coord_str}")

        # want to store in our /exp folder
        out_dir = os.getenv('NAVSIM_EXP_ROOT')
        sel_method = cfg.selection_method.lower()
        if out_dir is None:
            overlay_dir = Path.cwd() / f"{k}_proposals_{sel_method}"
        else:
            overlay_dir = Path(out_dir) / f"{k}_proposals_{sel_method}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = overlay_dir / f"traj_overlay_{k}_{sel_method}_{token}.png"
        cv2.imwrite(str(out_img_path), out_img)
        print(f'Wrote overlay image to {out_img_path}')

        # save polyline strings to a text file next to the image
        out_txt_path = overlay_dir / f"traj_overlay_{k}_{sel_method}_{token}.txt"
        with open(out_txt_path, 'w') as ftxt:
            for line in polyline_strings:
                ftxt.write(line + "\n")
        print(f'Wrote trajectory strings to {out_txt_path}')

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
