import logging
import os
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


def select_centorids(dp_np, k, sel_method, rng_seed=0, min_total_disp=0.0, dedup_tol=1e-2):
    # copied and slightly adapted from gen_traj_and_project
    N = dp_np.shape[0]
    if N == 0:
        return dp_np

    traj_xy = dp_np[..., :2]
    if traj_xy.shape[1] > 1:
        step_dists = np.linalg.norm(np.diff(traj_xy, axis=1), axis=2)
        total_disp = step_dists.sum(axis=1)
    else:
        total_disp = np.zeros((N,))

    keep_mask = total_disp >= float(min_total_disp)
    if keep_mask.sum() == 0:
        cand = dp_np
    else:
        cand = dp_np[keep_mask]

    def filter_empty_proposals(arr, min_total_disp_local=0.0):
        M_local = arr.shape[0]
        if M_local == 0:
            return arr
        traj_xy_local = arr[..., :2]
        L = traj_xy_local.shape[1]

        empty_mask = np.zeros((M_local,), dtype=bool)
        single_point_mask = np.zeros((M_local,), dtype=bool)
        low_disp_mask = np.zeros((M_local,), dtype=bool)

        for i in range(M_local):
            xy = traj_xy_local[i]
            valid_t = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
            n_valid = int(valid_t.sum())
            if n_valid == 0:
                empty_mask[i] = True
                continue
            if n_valid <= 1:
                single_point_mask[i] = True
                continue
            idx = np.where(valid_t)[0]
            if len(idx) > 1:
                seq = xy[idx]
                step_dists_local = np.linalg.norm(np.diff(seq, axis=0), axis=1)
                total_disp_local = float(step_dists_local.sum())
            else:
                total_disp_local = 0.0
            if total_disp_local < float(min_total_disp_local):
                low_disp_mask[i] = True

        flat = arr.reshape(M_local, -1)
        any_nan_mask = np.any(~np.isfinite(flat), axis=1)

        invalid_mask = empty_mask | single_point_mask | low_disp_mask | any_nan_mask

        if int(np.sum(invalid_mask)) == M_local:
            return np.empty((0,) + arr.shape[1:], dtype=arr.dtype)
        return arr[~invalid_mask]

    if dedup_tol is not None and cand.shape[0] > 1:
        flat = np.round(cand.reshape(cand.shape[0], -1) / float(dedup_tol)).astype(np.int64)
        _, unique_idx = np.unique(flat, axis=0, return_index=True)
        cand = cand[sorted(unique_idx)]

    cand = filter_empty_proposals(cand, min_total_disp)
    if cand.shape[0] == 0:
        return cand

    M = cand.shape[0]
    k_out = min(int(k), M)

    sel_method_l = str(sel_method).lower()
    if sel_method_l == 'kmeans' and M >= k_out:
        traj_flat = cand.reshape(M, -1)
        kmeans = KMeans(n_clusters=k_out, random_state=int(rng_seed), n_init=10).fit(traj_flat)
        centers = kmeans.cluster_centers_.reshape(k_out, cand.shape[1], cand.shape[2])
    elif sel_method_l in ('ff', 'farthest_first'):
        traj_flat = cand.reshape(M, -1)
        rng = np.random.RandomState(int(rng_seed))
        if M <= k_out:
            centers = cand.copy()
        else:
            first_idx = int(rng.randint(0, M))
            selected = [first_idx]
            for _ in range(1, k_out):
                dists = np.linalg.norm(traj_flat[:, None, :] - traj_flat[selected][None, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                min_dists[selected] = -1.0
                next_idx = int(np.argmax(min_dists))
                selected.append(next_idx)
            centers = cand[selected]
    else:
        rng = np.random.RandomState(int(rng_seed))
        if M >= k_out:
            idxs = rng.choice(M, size=k_out, replace=False)
        else:
            idxs = np.arange(M)
        centers = cand[idxs]

    return centers


def make_stitched_and_projector(scene, fb):
    # identical to original helper
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]

    cam_l0 = frame.cameras.cam_l0
    cam_f0 = frame.cameras.cam_f0
    cam_r0 = frame.cameras.cam_r0

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
        return None, None

    stitched = np.concatenate([l0_crop, f0_crop, r0_crop], axis=1)
    cam_w = fb._config.camera_width
    cam_h = fb._config.camera_height
    stitched_resized = cv2.resize(stitched, (cam_w, cam_h))

    w_l, w_f, w_r = l0_crop.shape[1], f0_crop.shape[1], r0_crop.shape[1]
    stitched_w = w_l + w_f + w_r
    stitched_h = stitched.shape[0]
    scale_x = cam_w / stitched_w
    scale_y = cam_h / stitched_h
    offsets = [0, w_l, w_l + w_f]

    def project_to_stitched(xy):
        lidar_pc = np.zeros((6, 1), dtype=np.float32)
        lidar_pc[0, 0] = xy[0]
        lidar_pc[1, 0] = xy[1]
        lidar_pc[2, 0] = 0.0

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
            img_h_full, img_w_full = cam.image.shape[:2]
            pc_img, in_fov = _transform_pcs_to_images(lidar_pc, rot, trans, intr, img_shape=(img_h_full, img_w_full))
            if in_fov[0]:
                u_full, v_full = pc_img[0]
                u_crop = u_full - (416 if lr_crop else 0)
                v_crop = v_full - 28
                stitched_x = u_crop + x_off
                stitched_y = v_crop
                px = int(np.round(stitched_x * scale_x))
                py = int(np.round(stitched_y * scale_y))
                return px, py
        return None

    out_img = stitched_resized.copy()
    if out_img.dtype != np.uint8:
        out_img = out_img.astype(np.uint8)

    return out_img, project_to_stitched


def draw_and_save_debug(out_img, project_fn, centers, token, cfg, k):
    # create BEV image first (non-projected trajectories)
    HORIZON = centers.shape[1] if centers is not None and centers.size else 0

    color_map = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (128, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (192, 128, 64),
    ]

    out_dir = os.getenv('NAVSIM_EXP_ROOT')
    sel_method = cfg.selection_method.lower()
    if out_dir is None:
        overlay_dir = Path.cwd() / f"{k}_proposals_{sel_method}_bev"
    else:
        overlay_dir = Path(out_dir) / f"{k}_proposals_{sel_method}_bev"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    bev_w = int(getattr(cfg, 'bev_width', 512))
    bev_h = int(getattr(cfg, 'bev_height', 512))

    bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8) + 255

    polyline_proj_strings = []
    polyline_bev_strings = []

    # compute world bounds from centers
    xs = []
    ys = []
    if centers is not None and centers.size:
        for i in range(centers.shape[0]):
            for t in range(HORIZON):
                xy = centers[i, t][:2]
                if np.isfinite(xy).all():
                    xs.append(float(xy[0]))
                    ys.append(float(xy[1]))

    if len(xs) == 0:
        min_x = -10.0
        max_x = 10.0
        min_y = -10.0
        max_y = 10.0
    else:
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

    # add margin
    margin = max(5.0, 0.05 * max(max_x - min_x + 1e-6, max_y - min_y + 1e-6))
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    span_x = max_x - min_x if (max_x - min_x) > 0 else 1.0
    span_y = max_y - min_y if (max_y - min_y) > 0 else 1.0
    scale = min((bev_w - 2) / span_x, (bev_h - 2) / span_y)

    def world_to_bev_px(xy):
        x, y = float(xy[0]), float(xy[1])
        u = int((x - min_x) * scale) + 1
        v = bev_h - (int((y - min_y) * scale) + 1)
        return u, v

    for i in range(k):
        color = color_map[i % len(color_map)]
        bev_pts = []
        for t in range(HORIZON):
            xy = centers[i, t][:2]
            if np.isfinite(xy).all():
                p = world_to_bev_px(xy)
                bev_pts.append(p)
        if len(bev_pts) >= 2:
            pts_arr = np.array(bev_pts, dtype=np.int32)
            cv2.polylines(bev_img, [pts_arr], isClosed=False, color=color, thickness=2)
        elif len(bev_pts) == 1:
            cv2.circle(bev_img, tuple(bev_pts[0]), 3, color, -1)

        if len(bev_pts) > 0:
            coord_str_bev = ";".join([f"{int(x)},{int(y)}" for (x, y) in bev_pts])
        else:
            coord_str_bev = ""
        polyline_bev_strings.append(f"traj_{i}: {coord_str_bev}")

    bev_path = overlay_dir / f"traj_bev_{k}_{sel_method}_{token}.png"
    cv2.imwrite(str(bev_path), bev_img)
    print(f'Wrote BEV image to {bev_path}')

    # now draw projected trajectories onto stitched image copy
    proj_img = out_img.copy()
    for i in range(k):
        color = color_map[i % len(color_map)]
        pts = []
        for t in range(HORIZON):
            xy = centers[i, t][:2]
            p = project_fn(xy)
            if p is not None:
                pts.append(p)
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(proj_img, [pts_arr], isClosed=False, color=color, thickness=2)
        elif len(pts) == 1:
            cv2.circle(proj_img, tuple(pts[0]), 3, color, -1)

        if len(pts) > 0:
            coord_str_proj = ";".join([f"{int(x)},{int(y)}" for (x, y) in pts])
        else:
            coord_str_proj = ""
        polyline_proj_strings.append(f"traj_{i}: {coord_str_proj}")

    proj_path = overlay_dir / f"traj_proj_{k}_{sel_method}_{token}.png"
    cv2.imwrite(str(proj_path), proj_img)
    print(f'Wrote projected overlay image to {proj_path}')

    # write a combined txt file containing BEV (non-projected) and Projected trajectories
    out_txt_path = overlay_dir / f"traj_debug_{k}_{sel_method}_{token}.txt"
    with open(out_txt_path, 'w') as ftxt:
        ftxt.write("BEV (non-projected) trajectories:\n")
        for line in polyline_bev_strings:
            ftxt.write(line + "\n")
        ftxt.write("\nProjected trajectories (on stitched image):\n")
        for line in polyline_proj_strings:
            ftxt.write(line + "\n")
        if k == 0:
            ftxt.write("ALL_NAN: True\n")
    print(f'Wrote debug trajectory strings to {out_txt_path}')


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    try:
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

        from torch.utils.data import Subset, DataLoader
        fb = agent.get_feature_builders()[0]

        subset = None
        dl_cfg = None
        if cfg.get('dataloader') and cfg.dataloader.get('params'):
            dl_cfg = cfg.dataloader.params
        batch_size = int(dl_cfg.get('batch_size', 1)) if dl_cfg is not None else 1
        num_workers = int(dl_cfg.get('num_workers', 0)) if dl_cfg is not None else 0
        pin_memory = bool(dl_cfg.get('pin_memory', False)) if dl_cfg is not None else False

        # parse generate_count to support 'one', 'all', or an integer top-N subset
        gen_count_cfg = cfg.get('generate_count', 'one')
        gen_count = None
        try:
            if isinstance(gen_count_cfg, int):
                gen_count = int(gen_count_cfg)
            else:
                gen_count_str = str(gen_count_cfg).lower()
                if gen_count_str == 'one':
                    gen_count = 'one'
                elif gen_count_str == 'all':
                    gen_count = 'all'
                else:
                    # try parse integer-like strings
                    gen_count = int(gen_count_cfg)
        except Exception:
            gen_count = 'one'

        if gen_count == 'one':
            print("Extracting first element from dataset for testing!")
            subset = Subset(dataset, [0])
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        elif gen_count == 'all':
            print("Processing all samples in dataset")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        elif isinstance(gen_count, int) and gen_count > 0:
            n = min(len(dataset), gen_count)
            print(f"Processing first {n} samples from dataset for debugging")
            subset = Subset(dataset, list(range(n)))
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        else:
            print(f"Unknown generate_count '{gen_count_cfg}'; defaulting to 'one'")
            subset = Subset(dataset, [0])
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        trainer_params = dict(cfg.trainer.params) if cfg.get('trainer') else {}

        try:
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            available_gpus = 0

        try:
            requested_devices = trainer_params.get('devices', None)
            if requested_devices is None:
                devices_to_check = available_gpus if available_gpus > 0 else 1
            elif isinstance(requested_devices, str) and requested_devices.lower() in ('auto', 'all'):
                devices_to_check = available_gpus if available_gpus > 0 else 1
            else:
                devices_to_check = int(requested_devices)
                if available_gpus > 0:
                    devices_to_check = min(devices_to_check, available_gpus)
                if devices_to_check <= 0:
                    devices_to_check = 1
            trainer_params['devices'] = devices_to_check
        except Exception:
            devices_to_check = 1
            trainer_params['devices'] = 1

        subset_len = len(subset) if subset is not None else len(dataset)
        if (devices_to_check > 1) and (subset_len < devices_to_check):
            print(f"Dataset size ({subset_len}) < devices ({devices_to_check}); forcing devices=1 to avoid DDP sampler issues.")
            trainer_params['devices'] = 1
            trainer_params.pop('strategy', None)
        if trainer_params.get('devices', 1) <= 1:
            trainer_params.pop('strategy', None)

        print("Resolved trainer_params:", trainer_params)
        trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())

        try:
            predictions = trainer.predict(AgentLightningModule(agent=agent, combined=False), dataloader, return_predictions=True)
        except Exception:
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

        # decide tokens to process based on generate_count config
        if gen_count == 'all':
            tokens_to_process = list(merged.keys())
        elif gen_count == 'one':
            tokens_to_process = [list(merged.keys())[0]]
        else:
            # numeric top-N selection from merged keys
            try:
                num = int(gen_count)
                tokens_to_process = list(merged.keys())[:num]
            except Exception:
                tokens_to_process = [list(merged.keys())[0]]

        for token in tokens_to_process:
            result = merged[token]

            dp_pred = None
            if isinstance(result, dict):
                dp_pred = result.get('dp_pred', None)
                if dp_pred is None:
                    for v in result.values():
                        if hasattr(v, 'shape'):
                            dp_pred = v
                            break

            if dp_pred is None:
                print(f'No trajectory proposals found in prediction for token {token}')
                continue

            print(f"normalizing proposals for token {token}")

            if hasattr(dp_pred, 'cpu'):
                dp_np = dp_pred.cpu().numpy()
            else:
                dp_np = np.array(dp_pred)
            if dp_np.ndim == 4:
                dp_np = dp_np[0]

            N, HORIZON, DIM = dp_np.shape
            print(f'Found {N} proposals for token {token}; selecting exemplars')

            k = int(cfg.k)
            sel_method = cfg.selection_method.lower()
            centers = select_centorids(dp_np, k=k, sel_method=sel_method, rng_seed=0)
            k = centers.shape[0] if (centers is not None and centers.shape[0] > 0) else 0

            scene = scene_loader.get_scene_from_token(token)
            out_img, project_fn = make_stitched_and_projector(scene, fb)
            if out_img is None or project_fn is None:
                print(f'Missing camera images for token {token}; cannot create stitched overlay')
                continue

            draw_and_save_debug(out_img, project_fn, centers, token, cfg, k)

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
