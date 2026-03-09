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

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import Dataset
from navsim.visualization.camera import _transform_pcs_to_images

logger = logging.getLogger(__name__)


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "diffusion_to_projection"


def make_stitched_and_projector(scene, fb):
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
        return None, None

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


def draw_trajectories_and_save(out_img, project_fn, centers, token, cfg, k):
    # Map color names to BGR tuples for OpenCV
    color_map = [
        ("red", (0, 0, 255)),
        ("green", (0, 255, 0)),
        ("blue", (255, 0, 0)),
        ("yellow", (0, 255, 255)),
        ("magenta", (255, 0, 255)),
        ("cyan", (255, 255, 0)),
        ("purple", (128, 0, 128)),
        ("teal", (128, 128, 0)),
        ("olive", (0, 128, 128)),
        ("steelblue", (192, 128, 64)),
    ]

    polyline_strings = []
    HORIZON = centers.shape[1]

    for i in range(k):
        color_str, color_bgr = color_map[i % len(color_map)]
        pts = []
        for t in range(HORIZON):
            xy = centers[i, t][:2]
            p = project_fn(xy)
            if p is not None:
                pts.append(p)
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(out_img, [pts_arr], isClosed=False, color=color_bgr, thickness=2)
        elif len(pts) == 1:
            cv2.circle(out_img, tuple(pts[0]), 3, color_bgr, -1)

        if len(pts) > 0:
            coord_str = ";".join([f"{int(x)},{int(y)}" for (x, y) in pts])
        else:
            coord_str = ""
        polyline_strings.append(f"{color_str}: {coord_str}")

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

    out_txt_path = overlay_dir / f"traj_overlay_{k}_{sel_method}_{token}.txt"
    with open(out_txt_path, 'w') as ftxt:
        for line in polyline_strings:
            ftxt.write(line + "\n")
        # mark cases where no valid proposals remained
        if k == 0:
            ftxt.write("ALL_NAN: True\n")
    print(f'Wrote trajectory strings to {out_txt_path}')


def score_and_select_trajectories_gtrs_dense(
    dp_np: np.ndarray,
    token: str,
    gtrs_agent,
    features: Dict[str, torch.Tensor],
    k: int,
) -> tuple:
    """
    Score trajectory proposals using GTRS-Dense neural network and select top-k by overall score.
    
    Args:
        dp_np: (N, H, D) numpy array of proposals in ego-frame (relative coordinates)
        token: scene token for logging
        gtrs_agent: GTRSAgent instance with evaluate_dp_proposals method
        features: dict with 'camera_feature' and 'status_feature' tensors
        k: number of top proposals to select
    
    Returns:
        tuple of (centers, scores) where centers is (k_out, H, D) and scores is (k_out,)
    """

    print(f"GTRS-Dense scoring {dp_np.shape[0]} proposals for token {token}")

    N = dp_np.shape[0]
    if N == 0:
        print(f"  No proposals to score")
        return np.empty((0, dp_np.shape[1], dp_np.shape[2])), np.array([])

    # Convert numpy proposals to torch tensor (keep in ego-frame format)
    dp_torch = torch.from_numpy(dp_np).float()  # (N, H, D)
    
    # Reshape proposals to (1, N, H*D) - add batch dimension and flatten trajectory dims
    N, H, D = dp_torch.shape
    # Before flattening, ensure proposals match model vocab horizon if possible
    try:
        vocab = gtrs_agent.model._trajectory_head.vocab.data
        _, V_H, V_D = vocab.shape
        expected_flat = V_H * V_D
        if (H != V_H) or (D != V_D):
            print(f"  Warning: proposal horizon/dim ({H},{D}) != model vocab ({V_H},{V_D}), interpolating proposals to match model.")
            # Interpolate each proposal to target horizon V_H
            dp_np_interp = np.zeros((N, V_H, V_D), dtype=dp_np.dtype)
            old_x = np.arange(H)
            new_x = np.linspace(0, H - 1, V_H)
            for i in range(N):
                for dim_i in range(D):
                    dp_np_interp[i, :, dim_i] = np.interp(new_x, old_x, dp_np[i, :, dim_i])
            # If D < V_D, pad zeros for missing dims; if D > V_D, truncate
            if D < V_D:
                if V_D > D:
                    pad = np.zeros((N, V_H, V_D - D), dtype=dp_np.dtype)
                    dp_np_interp = np.concatenate([dp_np_interp, pad], axis=2)
            elif D > V_D:
                dp_np_interp = dp_np_interp[:, :, :V_D]
            dp_np = dp_np_interp
            N, H, D = dp_np.shape
            dp_torch = torch.from_numpy(dp_np).float()
            dp_torch = dp_torch.reshape(1, N, H * D)
        else:
            dp_torch = dp_torch.reshape(1, N, H * D)  # (1, N, H*D)
    except Exception:
        dp_torch = dp_torch.reshape(1, N, H * D)  # (1, N, H*D)

    # Call GTRS-Dense scorer
    print(f"  Scoring {N} proposals with GTRS-Dense model")
    try:
        with torch.no_grad():
            # Move features to same device as model
            device = next(gtrs_agent.parameters()).device

            # Helper to normalize/move features to target device while preserving
            # the expected list/tensor structure used by HydraModel.
            def _move_and_normalize(feat_dict, device):
                out = {}
                for kf, fv in feat_dict.items():
                    # Camera features: may be list(history) or a single tensor/ndarray
                    if kf.startswith('camera_feature'):
                        if isinstance(fv, list):
                            new_list = []
                            for item in fv:
                                if item is None:
                                    new_list.append(None)
                                    continue
                                if isinstance(item, torch.Tensor):
                                    t = item
                                elif isinstance(item, np.ndarray):
                                    t = torch.from_numpy(item)
                                else:
                                    try:
                                        t = torch.tensor(item)
                                    except Exception:
                                        new_list.append(item)
                                        continue
                                if t.dim() == 3:
                                    t = t.unsqueeze(0)
                                new_list.append(t.to(device))
                            out[kf] = new_list
                        elif isinstance(fv, torch.Tensor):
                            t = fv
                            if t.dim() == 3:
                                t = t.unsqueeze(0)
                            out[kf] = t.to(device)
                        elif isinstance(fv, np.ndarray):
                            t = torch.from_numpy(fv)
                            if t.dim() == 3:
                                t = t.unsqueeze(0)
                            out[kf] = t.to(device)
                        else:
                            out[kf] = fv

                    # Status features: often a list where each element is a tensor/ndarray
                    elif kf == 'status_feature' or kf.endswith('status_feature'):
                        if isinstance(fv, list):
                            new_list = []
                            for item in fv:
                                if item is None:
                                    new_list.append(None)
                                    continue
                                if isinstance(item, torch.Tensor):
                                    t = item
                                elif isinstance(item, np.ndarray):
                                    t = torch.from_numpy(item)
                                else:
                                    try:
                                        t = torch.tensor(item)
                                    except Exception:
                                        new_list.append(item)
                                        continue
                                if t.dim() == 1:
                                    t = t.unsqueeze(0)
                                new_list.append(t.to(device))
                            out[kf] = new_list
                        elif isinstance(fv, torch.Tensor):
                            t = fv
                            if t.dim() == 1:
                                t = t.unsqueeze(0)
                            out[kf] = t.to(device)
                        elif isinstance(fv, np.ndarray):
                            t = torch.from_numpy(fv)
                            if t.dim() == 1:
                                t = t.unsqueeze(0)
                            out[kf] = t.to(device)
                        else:
                            out[kf] = fv

                    # Generic: move tensors/ndarrays, leave other types unchanged
                    else:
                        if isinstance(fv, torch.Tensor):
                            out[kf] = fv.to(device)
                        elif isinstance(fv, np.ndarray):
                            out[kf] = torch.from_numpy(fv).to(device)
                        else:
                            out[kf] = fv
                return out

            features_device = _move_and_normalize(features, device)
            dp_torch = dp_torch.to(device)

            def _shape_str(x):
                try:
                    if isinstance(x, list):
                        for item in reversed(x):
                            if item is None:
                                continue
                            return str(getattr(item, 'shape', None))
                        return 'list(all None)'
                    return str(getattr(x, 'shape', None))
                except Exception:
                    return 'N/A'

            print(f"  features['camera_feature'] shape: {_shape_str(features_device.get('camera_feature'))}")
            print(f"  features['status_feature'] shape: {_shape_str(features_device.get('status_feature'))}")
            print(f"  dp_torch shape: {dp_torch.shape}")

            # Call the GTRS scorer's evaluate_dp_proposals method
            result = gtrs_agent.evaluate_dp_proposals(
                features=features_device,
                dp_proposals=dp_torch
            )
    except Exception as e:
        print(f"  Error during GTRS scoring: {e}")
        traceback.print_exc()
        return np.empty((0, dp_np.shape[1], dp_np.shape[2])), np.array([])

    # Extract scores
    if 'overall_log_scores' in result:
        scores_arr = result['overall_log_scores'].cpu().numpy().flatten()
    elif 'overall_scores' in result:
        scores_arr = result['overall_scores'].cpu().numpy().flatten()
    else:
        print("  Warning: no overall_scores in result; using sum of sub-scores")
        # Fallback: combine sub-scores manually
        sub_scores = {}
        for key in ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'lane_keeping']:
            if key in result:
                sub_scores[key] = result[key].cpu().numpy().flatten()
        if sub_scores:
            scores_arr = np.sum(list(sub_scores.values()), axis=0)
        else:
            scores_arr = np.ones(N)

    # Select top-k by score
    k_out = min(k, len(scores_arr))
    top_k_indices = np.argsort(scores_arr)[-k_out:][::-1]  # descending

    centers = dp_np[top_k_indices]  # Return in original ego-frame format
    scores = scores_arr[top_k_indices]

    print(f"  Selected top {k_out} proposals by GTRS-Dense score")
    print(f"  GTRS scores: {scores}")

    return centers, scores


def collect_features_by_token(dataloader):
    """
    Iterate through dataloader and build a mapping of token -> features.
    
    The DataLoader may return batches as:
    - dict: features_dict directly
    - tuple: (features_dict, targets_dict, tokens) when append_token_to_batch=True
    
    Returns:
        dict mapping token -> {'camera_feature': list/tensor, 'status_feature': list/tensor}
    """
    features_by_token = {}
    with torch.no_grad():
        for batch in dataloader:
            features_dict = None
            tokens = None
            
            # Unpack batch depending on structure
            if isinstance(batch, dict):
                # Direct features dict
                features_dict = batch
                tokens = batch.get('token', None)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                # Tuple format: (features_dict, targets_dict, tokens, ...)
                features_dict = batch[0]
                if len(batch) >= 3:
                    tokens = batch[2]
                elif isinstance(batch[1], dict) and 'token' in batch[1]:
                    tokens = batch[1]['token']
            else:
                continue
            
            if not isinstance(features_dict, dict):
                continue
                
            camera_feat = features_dict.get('camera_feature', None)
            status_feat = features_dict.get('status_feature', None)
            
            if camera_feat is None or status_feat is None:
                continue
            
            # Extract tokens - could be list or string
            if tokens is None:
                continue
            
            if isinstance(tokens, (list, tuple)):
                # Multiple tokens in batch
                batch_size = len(tokens)
                for i in range(batch_size):
                    token = tokens[i]

                    # Handle camera_feature as list of images or batched tensor
                    cam = None
                    try:
                        # camera_feat could be:
                        # - list over history where each element is a list of per-sample tensors (history-major nested list)
                        # - list over history where each element is a batched tensor with shape [B, C, H, W]
                        # - list of per-history single-sample tensors (when DataLoader batch_size==1)
                        if isinstance(camera_feat, list):
                            # history-major
                            hist_list = camera_feat
                            per_hist_samples = []
                            for hist in hist_list:
                                if isinstance(hist, list):
                                    per_hist_samples.append(hist[i] if i < len(hist) else None)
                                elif hasattr(hist, 'shape') and len(hist.shape) >= 4:
                                    # batched tensor [B,...]
                                    per_hist_samples.append(hist[i:i+1])
                                else:
                                    per_hist_samples.append(hist)
                            # prefer returning a list of tensors (each with batch dim 1)
                            cam = []
                            for item in per_hist_samples:
                                if item is None:
                                    cam.append(None)
                                elif hasattr(item, 'shape') and len(item.shape) == 3:
                                    cam.append(item.unsqueeze(0))
                                else:
                                    cam.append(item)
                        elif hasattr(camera_feat, '__getitem__'):
                            # fallback: try indexing by sample
                            itm = camera_feat[i] if i < len(camera_feat) else camera_feat
                            cam = itm
                        else:
                            cam = camera_feat
                    except Exception:
                        cam = camera_feat
                    
                    # Handle status_feature as list or batched tensor
                    st = None
                    try:
                        if isinstance(status_feat, list):
                            # status_feat usually is a list over ego-status entries; each entry may be a batched tensor or list
                            per_status = []
                            for s in status_feat:
                                if isinstance(s, list):
                                    per_status.append(s[i] if i < len(s) else None)
                                elif hasattr(s, 'shape') and len(s.shape) >= 2 and s.shape[0] == batch_size:
                                    per_status.append(s[i:i+1])
                                else:
                                    per_status.append(s)
                            # ensure each per_status element has batch dim
                            st = []
                            for item in per_status:
                                if item is None:
                                    st.append(None)
                                elif hasattr(item, 'shape') and len(item.shape) == 1:
                                    st.append(item.unsqueeze(0))
                                else:
                                    st.append(item)
                        elif hasattr(status_feat, '__getitem__'):
                            itm = status_feat[i] if i < len(status_feat) else status_feat
                            st = itm
                        else:
                            st = status_feat
                    except Exception:
                        st = status_feat
                    
                    if token is not None and cam is not None and st is not None:
                        features_by_token[token] = {
                            'camera_feature': cam,
                            'status_feature': st,
                        }
            else:
                # Single token in batch
                token = tokens
                features_by_token[token] = {
                    'camera_feature': camera_feat,
                    'status_feature': status_feat,
                }

    return features_by_token



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        # instantiate main agent (diffusion/DP) and separate GTRS scorer agent
        agent = instantiate(cfg.agent)
        agent.initialize()

        scorer_agent = None
        if cfg.get('scorer_agent'):
            try:
                scorer_agent = instantiate(cfg.scorer_agent)
                # not all agents require initialize, but call if present
                if hasattr(scorer_agent, 'initialize'):
                    scorer_agent.initialize()
            except Exception:
                print('Warning: failed to instantiate or initialize scorer_agent')
                traceback.print_exc()

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

        # restrict to first item only for quick test (or use all, based on cfg.generate_count)
        from torch.utils.data import Subset, DataLoader
        fb = agent.get_feature_builders()[0]

        subset = None
        # resolve dataloader params from config
        dl_cfg = None
        if cfg.get('dataloader') and cfg.dataloader.get('params'):
            dl_cfg = cfg.dataloader.params
        batch_size = int(dl_cfg.get('batch_size', 1)) if dl_cfg is not None else 1
        num_workers = int(dl_cfg.get('num_workers', 0)) if dl_cfg is not None else 0
        pin_memory = bool(dl_cfg.get('pin_memory', False)) if dl_cfg is not None else False

        gen_count = str(cfg.get('generate_count', 'one'))
        # gen_count_lower = gen_count.lower()

        if gen_count.isdigit():
            num = int(gen_count)
            num = max(1, num)
            print(f"Extracting first {num} elements from dataset for testing!")
            subset = Subset(dataset, list(range(min(num, len(dataset)))))
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        elif gen_count == 'one':
            print("Extracting first element from dataset for testing!")
            subset = Subset(dataset, [0])
            print(f"Using DataLoader batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            # dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        elif gen_count == 'all':
            print("Processing all samples in dataset")
            print(f"Using DataLoader batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            # dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        else:
            print(f"Unknown generate_count '{gen_count}'; defaulting to 'one'")
            subset = Subset(dataset, [0])
            print(f"Using DataLoader batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            # dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
        
        #check keys
        # run in your script right after dataloader is created
        for batch in dataloader:
            print("batch type:", type(batch))
            if isinstance(batch, dict):
                for k,v in batch.items():
                    print(k, "->", type(v), getattr(v, 'shape', None))
            elif isinstance(batch, (list, tuple)):
                print("batch is list/tuple length", len(batch))
                first = batch[0] if len(batch)>0 else None
                if isinstance(first, dict):
                    for k,v in first.items():
                        print("item[0].", k, "->", type(v), getattr(v, 'shape', None))
            break

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
        # Build a separate Dataset/DataLoader for GTRS features (camera_feature/status_feature)
        features_by_token = {}
        if scorer_agent is not None:
            print("Building GTRS feature dataset and dataloader...")
            try:
                # Build a SceneLoader for the scorer agent using its expected sensor config.
                # The main scene_loader above was created with the proposal agent's sensor config,
                # which can differ from the scorer agent and lead to missing camera images.
                try:
                    sc_cfg = scorer_agent.get_sensor_config()
                    print(f"Scorer agent sensor_config: {sc_cfg}")
                    print("Scorer SceneLoader paths:", {
                        'synthetic_sensor_path': cfg.synthetic_sensor_path,
                        'original_sensor_path': cfg.original_sensor_path,
                        'navsim_log_path': cfg.navsim_log_path,
                        'synthetic_scenes_path': cfg.synthetic_scenes_path,
                    })
                    scorer_scene_loader = SceneLoader(
                        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
                        original_sensor_path=Path(cfg.original_sensor_path),
                        data_path=Path(cfg.navsim_log_path),
                        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
                        scene_filter=scene_filter_override,
                        sensor_config=scorer_agent.get_sensor_config(),
                    )
                    print("Scorer SceneLoader created successfully using scorer_agent.get_sensor_config()")
                except Exception as e:
                    # fallback to using the original scene_loader if scorer agent doesn't provide get_sensor_config
                    print("unable to load scorer_scene_loader, defaulting to scene_loader", e)
                    scorer_scene_loader = scene_loader

                gtrs_dataset = Dataset(
                    scene_loader=scorer_scene_loader,
                    feature_builders=scorer_agent.get_feature_builders(),
                    target_builders=scorer_agent.get_target_builders(),
                    cache_path=None,
                    force_cache_computation=False,
                    append_token_to_batch=True,
                    is_training=False,
                )

                print("GTRS dataset length:", len(gtrs_dataset))
                if len(gtrs_dataset) > 0:
                    # inspect first item to verify features are produced
                    try:
                        sample = gtrs_dataset[0]
                        if isinstance(sample, dict):
                            print("gtrs_dataset[0] keys:", list(sample.keys()))
                            for k, v in sample.items():
                                print(f"  {k}: type={type(v)}, shape={getattr(v,'shape', None)}")
                        else:
                            print("gtrs_dataset[0] returned non-dict type:", type(sample))
                    except Exception:
                        print("Failed to index gtrs_dataset[0]")

                gtrs_dataloader = DataLoader(gtrs_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
                print("GTRS dataloader created:", {
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'pin_memory': pin_memory,
                    'dataset_len': len(gtrs_dataset)
                })
                # Peek a single batch to inspect structure without disrupting iteration
                try:
                    batch_peek = next(iter(gtrs_dataloader))
                    print("Peek GTRS batch type:", type(batch_peek))
                    if isinstance(batch_peek, dict):
                        for k, v in batch_peek.items():
                            print(f"  peek {k} -> {type(v)}, shape={getattr(v,'shape', None)}")
                    elif isinstance(batch_peek, (list, tuple)):
                        print("  peek batch is list/tuple length", len(batch_peek))
                        first = batch_peek[0] if len(batch_peek) > 0 else None
                        if isinstance(first, dict):
                            for k, v in first.items():
                                print(f"    peek item[0].{k} -> {type(v)}, shape={getattr(v,'shape', None)}")
                except Exception as e:
                    print("  Failed to peek gtrs_dataloader batch:", e)

                print("Collecting features from GTRS dataloader...")
                features_by_token = collect_features_by_token(gtrs_dataloader)
                print(f"  Collected features for {len(features_by_token)} tokens")
                if len(features_by_token) > 0:
                    sample_keys = list(features_by_token.keys())[:5]
                    print(f"  Sample tokens with features: {sample_keys}")
            except Exception:
                print('Warning: failed to build GTRS dataset/dataloader or collect features')
                traceback.print_exc()
                features_by_token = {}
        else:
            print('No scorer_agent configured; skipping GTRS feature collection')

        gen_count = str(cfg.get('generate_count', 'one'))
        # allow numeric strings to request N examples
        if gen_count.isdigit():
            n = int(gen_count)
            all_tokens = list(merged.keys())
            tokens_to_process = all_tokens[:n]
        elif gen_count.lower() == 'all':
            tokens_to_process = list(merged.keys())
        else:
            tokens_to_process = [list(merged.keys())[0]]

        # Filter tokens to those for which we collected GTRS features
        if features_by_token:
            prior_count = len(tokens_to_process)
            available = set(features_by_token.keys())
            tokens_to_process = [t for t in tokens_to_process if t in available]
            removed = prior_count - len(tokens_to_process)
            print(f"Filtered tokens_to_process by available GTRS features: kept {len(tokens_to_process)} / {prior_count} (removed {removed})")
            if removed > 0:
                missing = [t for t in tokens_to_process if t not in available]
                print(f"  Note: some requested tokens had no features; sample missing tokens omitted from processing")

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
            print(f'Found {N} proposals for token {token}; selecting {cfg.k} exemplars')

            k = int(cfg.k)
            
            # Use GTRS-Dense scoring for proposal selection
            features = features_by_token.get(token)

            # DEBUG: print out feature container types and shapes for this token
            try:
                print(f"  Debug: features for token {token} -> type={type(features)}")
                if isinstance(features, dict):
                    for fk, fv in features.items():
                        try:
                            shape = getattr(fv, 'shape', None)
                            device = getattr(fv, 'device', None)
                            print(f"    {fk}: type={type(fv)}, shape={shape}, device={device}")
                        except Exception:
                            print(f"    {fk}: type={type(fv)} (failed to get shape)")
                else:
                    print("    features is not a dict; repr:", repr(features))
            except Exception:
                print("    Failed to print debug info for features")

            if features is None:
                print(f'  Warning: No features found for token {token}; cannot use GTRS-Dense scoring')
                print(f'  Skipping {token}')
                continue
            
            if scorer_agent is None:
                print('No scorer_agent configured; cannot run GTRS-Dense scoring')
                continue

            centers, gtrs_scores = score_and_select_trajectories_gtrs_dense(
                dp_np=dp_np,
                token=token,
                gtrs_agent=scorer_agent,
                features=features,
                k=k,
            )
            k = centers.shape[0] if (centers is not None and centers.shape[0] > 0) else 0

            scene = scene_loader.get_scene_from_token(token)
            out_img, project_fn = make_stitched_and_projector(scene, fb)
            if out_img is None or project_fn is None:
                print(f'Missing camera images for token {token}; cannot create stitched overlay')
                continue

            draw_trajectories_and_save(out_img, project_fn, centers, token, cfg, k)

        

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
