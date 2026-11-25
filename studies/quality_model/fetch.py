"""Fetch real LeRobot datasets (signals only) and cache per-episode action/state + robolint findings."""

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SETS = """lerobot/svla_so101_pickplace lerobot/svla_so100_pickplace lerobot/svla_so100_sorting lerobot/svla_so100_stacking
jchandru08/so100_press_red_button kmg0620/so101_bomb_pick_place_v1 Embodied-AI-6/dataset SO-101-robot-learning/so101_pick_place
imstevenpmwork/omx_sentry_1 EPITECH-LILLE/eval_training recast-robotics/rollout_dagger_r1_s2_b0_20260915_121327 brandonyang/artem-pour-water
lerobot/high_quality_folding lerobot/FineART RoboCOIN/Agilex_Cobot_Magic_heat_burger RoboCOIN/Agilex_Cobot_Magic_place_towel_flat
RoboCOIN/AI2_Alphabot_2_stamp_document RoboCOIN/Galaxea_R1_Lite_classify_object_green_tablecloth RoboCOIN/Airbot_MMK2_organize_plate
Embodied-AI-6/blue_so101_color_blocks_10ep_v1_20260526_175604 robot-learning-group47/eval2_bgr_g omkarmayekar555/robot-learning-tutorial-data
mliu48/telearms-yam-cube-v4 nurtayev-d/seasondish-shake-ag-1000ep adityx23/icl-dataset jnogga/droid_success
allenai/MolmoAct2-BimanualYAM-Dataset LeCAR-Lab/Wanda zekaiwang/trex_dataset BitRobot/HIW-500-LeRobot""".split()
OUT = Path(__file__).parent / "cache"


def one(src):
    out = OUT / (src.replace("/", "__") + ".pkl")
    if out.exists():
        return src, "cached"
    try:
        from robolint import lint

        r = lint(src, max_episodes=120)
        ds = r.dataset
        eps = []
        for ep in ds.episodes:
            f = ep.frames
            a = np.asarray(f["action"], dtype=np.float32) if "action" in f else None
            s = np.asarray(f["observation.state"], dtype=np.float32) if "observation.state" in f else None
            if a is None or a.ndim != 2:
                continue
            eps.append(
                {
                    "index": ep.index,
                    "action": a,
                    "state": s,
                    "findings": [(x.code, int(x.severity)) for x in r.episode_findings(ep.index)],
                }
            )
        pickle.dump(
            {
                "source": src,
                "fps": ds.fps,
                "robot": ds.info.get("robot_type"),
                "names": ds.feature_names("action") if "action" in ds.features else [],
                "episodes": eps,
            },
            open(out, "wb"),
        )
        return src, f"{len(eps)} eps"
    except Exception as e:
        return src, "FAIL " + repr(e)[:200]


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        for src, msg in ex.map(one, SETS):
            print(src, msg, flush=True)
