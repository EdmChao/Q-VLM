# single_token_check.py
from pathlib import Path
import hydra
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig, SceneFilter


def main():
    # Compose the same Hydra config used by gen_traj_and_project
    with hydra.initialize(config_path="config/pdm_scoring", version_base=None):
        cfg = hydra.compose(config_name="diffusion_to_projection")

    # Instantiate scorer agent if available in config
    scorer_agent = None
    try:
        if cfg.get('scorer_agent'):
            try:
                scorer_agent = instantiate(cfg.scorer_agent)
                if hasattr(scorer_agent, 'initialize'):
                    scorer_agent.initialize()
                print('Instantiated scorer_agent from config')
            except Exception as e:
                print('Failed to instantiate scorer_agent:', e)
    except Exception:
        # cfg may not have scorer_agent defined
        pass

    # Resolve sensor config
    if scorer_agent is not None and hasattr(scorer_agent, 'get_sensor_config'):
        sensor_config = scorer_agent.get_sensor_config()
    else:
        sensor_config = SensorConfig.build_all_sensors()

    # Build same scene_filter as gen_traj_and_project
    scene_filter_override = SceneFilter(
        num_history_frames=2,
        num_future_frames=1,
        frame_interval=1,
        has_route=False,
        include_synthetic_scenes=True,
    )

    # Paths from config
    synthetic_sensor_path = Path(cfg.synthetic_sensor_path)
    original_sensor_path = Path(cfg.original_sensor_path)
    navsim_log_path = Path(cfg.navsim_log_path)
    synthetic_scenes_path = Path(cfg.synthetic_scenes_path)

    loader = SceneLoader(
        synthetic_sensor_path=synthetic_sensor_path,
        original_sensor_path=original_sensor_path,
        data_path=navsim_log_path,
        synthetic_scenes_path=synthetic_scenes_path,
        scene_filter=scene_filter_override,
        sensor_config=sensor_config,
    )

    # Pick a token to inspect
    token = list(loader.tokens)[0]
    print("Inspecting token:", token)
    scene = loader.get_scene_from_token(token)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    print("frame_idx:", frame_idx)

    frame = scene.frames[frame_idx]
    cams = frame.cameras

    cam_names = ["cam_l2", "cam_l1", "cam_l0", "cam_f0", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
    for name in cam_names:
        cam = getattr(cams, name)
        print(f"{name}: camera_path={cam.camera_path}, image_is_None={cam.image is None}")
        if cam.camera_path is not None:
            full = (synthetic_sensor_path / cam.camera_path)
            print("  resolved path (synthetic):", full, "exists:", full.exists())
            full2 = (original_sensor_path / cam.camera_path)
            print("  resolved path (original):", full2, "exists:", full2.exists())


if __name__ == '__main__':
    main()