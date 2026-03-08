# # single_token_check.py
# from pathlib import Path
# import hydra
# from hydra.utils import instantiate
# from navsim.common.dataloader import SceneLoader
# from navsim.common.dataclasses import SensorConfig, SceneFilter


# def main():
#     # Compose the same Hydra config used by gen_traj_and_project
#     with hydra.initialize(config_path="config/pdm_scoring", version_base=None):
#         cfg = hydra.compose(config_name="diffusion_to_projection")

#     # Instantiate scorer agent if available in config
#     scorer_agent = None
#     try:
#         if cfg.get('scorer_agent'):
#             try:
#                 scorer_agent = instantiate(cfg.scorer_agent)
#                 if hasattr(scorer_agent, 'initialize'):
#                     scorer_agent.initialize()
#                 print('Instantiated scorer_agent from config')
#             except Exception as e:
#                 print('Failed to instantiate scorer_agent:', e)
#     except Exception:
#         # cfg may not have scorer_agent defined
#         pass

#     # Resolve sensor config
#     if scorer_agent is not None and hasattr(scorer_agent, 'get_sensor_config'):
#         sensor_config = scorer_agent.get_sensor_config()
#     else:
#         sensor_config = SensorConfig.build_all_sensors()

#     # Build same scene_filter as gen_traj_and_project
#     scene_filter_override = SceneFilter(
#         num_history_frames=2,
#         num_future_frames=1,
#         frame_interval=1,
#         has_route=False,
#         include_synthetic_scenes=True,
#     )

#     # Paths from config
#     synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
#     original_sensor_path = Path(cfg.original_sensor_path)
#     navsim_log_path = Path(cfg.navsim_log_path)
#     synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

#     loader = SceneLoader(
#         synthetic_sensor_path=synthetic_sensor_path,
#         original_sensor_path=original_sensor_path,
#         data_path=navsim_log_path,
#         synthetic_scenes_path=synthetic_scenes_path,
#         scene_filter=scene_filter_override,
#         sensor_config=sensor_config,
#     )

#     # Pick a token to inspect
#     token = list(loader.tokens)[0]
#     print("Inspecting token:", token)
#     scene = loader.get_scene_from_token(token)
#     frame_idx = scene.scene_metadata.num_history_frames - 1
#     print("frame_idx:", frame_idx)

#     frame = scene.frames[frame_idx]
#     cams = frame.cameras

#     cam_names = ["cam_l2", "cam_l1", "cam_l0", "cam_f0", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
#     for name in cam_names:
#         cam = getattr(cams, name)
#         print(f"{name}: camera_path={cam.camera_path}, image_is_None={cam.image is None}")
#         if cam.camera_path is not None:
#             full = (synthetic_sensor_path / cam.camera_path)
#             print("  resolved path (synthetic):", full, "exists:", full.exists())
#             full2 = (original_sensor_path / cam.camera_path)
#             print("  resolved path (original):", full2, "exists:", full2.exists())


# if __name__ == '__main__':
#     main()

# debug_feature_inspect.py
from pathlib import Path
import hydra
from hydra.utils import instantiate
import numpy as np

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter

# compose same config as gen_traj_and_project
with hydra.initialize(config_path="config/pdm_scoring", version_base=None):
    cfg = hydra.compose(config_name="diffusion_to_projection")

# instantiate scorer agent if present
scorer_agent = None
if cfg.get('scorer_agent'):
    try:
        scorer_agent = instantiate(cfg.scorer_agent)
        if hasattr(scorer_agent, 'initialize'):
            scorer_agent.initialize()
        print("scorer_agent ready")
    except Exception as e:
        print("failed to instantiate scorer_agent:", e)
        raise

scene_filter_override = SceneFilter(
    num_history_frames=2,
    num_future_frames=1,
    frame_interval=1,
    has_route=False,
    include_synthetic_scenes=True,
)

scorer_scene_loader = SceneLoader(
    synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
    original_sensor_path=Path(cfg.original_sensor_path),
    data_path=Path(cfg.navsim_log_path),
    synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
    scene_filter=scene_filter_override,
    sensor_config=scorer_agent.get_sensor_config() if scorer_agent else None,
)

# pick the token you saw fail
token = "249204a76cf04b884"
print("Token:", token)

# get agent_input for this token (this is what builders use)
agent_input = scorer_scene_loader.get_agent_input_from_token(token)
fb = scorer_agent.get_feature_builders()[0]  # HydraFeatureBuilder
cfg_fb = fb._config
seq_len = cfg_fb.seq_len
print("feature builder seq_len:", seq_len)

# Inspect each history frame used by builder
for i, camera in enumerate(agent_input.cameras[-seq_len:]):
    idx = -seq_len + i
    print(f"\nHistory idx {idx} (relative):")
    for cam_name in ["cam_l2","cam_l1","cam_l0","cam_f0","cam_r0","cam_r1","cam_r2","cam_b0"]:
        cam = getattr(camera, cam_name)
        shape = None
        any_nonzero = None
        if cam.image is not None:
            arr = np.array(cam.image)
            shape = arr.shape
            any_nonzero = bool(np.any(arr))
        print(f"  {cam_name}: camera_path={cam.camera_path}, image_is_None={cam.image is None}, shape={shape}, any={any_nonzero}")

# Check crops used by builder (left/right crop logic)
print("\nCrop requirements check (must be >=):")
print("  crop removes 28 px top/bottom and 416 px left/right when lr crop used.")
print("  so width must be > 832 and height must be > 56")

# Run the feature builder and show exact outputs
features = fb.compute_features(agent_input)
print("\ncompute_features returned keys:", list(features.keys()))
cam_feat = features.get("camera_feature")
print("camera_feature type:", type(cam_feat))
if isinstance(cam_feat, list):
    for i, el in enumerate(cam_feat):
        print(f"  camera_feature[{i}] = {type(el)}, shape={getattr(el,'shape',None)}")
else:
    print("  camera_feature:", cam_feat)

status_feat = features.get("status_feature")
print("status_feature type:", type(status_feat))
if isinstance(status_feat, list):
    for i, el in enumerate(status_feat):
        print(f"  status_feature[{i}] = {type(el)}, shape={getattr(el,'shape',None)}")
else:
    print("  status_feature:", status_feat)