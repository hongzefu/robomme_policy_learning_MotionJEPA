"""RESET_HOOK_PARITY：同一局包与不包钩子各 reset 一次，图像与状态逐帧逐位比较。只 reset，不用模型。"""
import sys
import numpy as np
from robomme.env_record_wrapper import BenchmarkEnvBuilder
from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
KEYS = ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list", "eef_state_list")
episodes = frames = diff = 0
for task in ("PatternLock", "RouteStick"):
    b = BenchmarkEnvBuilder(env_id=task, dataset="test", action_space="joint_angle", gui_render=False, max_steps=1300)
    for ep in (0, 2, 3):     # easy、medium、hard
        runs = []
        for hooked in (False, True):
            env = b.make_env_for_episode(ep)
            if hooked:
                u, w = env.unwrapped, env
                while not isinstance(w, DemonstrationWrapper):
                    w = w.env
                orig = w._augment_obs_and_info
                def aug(obs, info, action, _o=orig, _u=u):
                    o, i = _o(obs, info, action)
                    i["subgoal_task_index"] = int(_u.current_task_index)
                    return o, i
                w._augment_obs_and_info = aug
            try:
                obs, info = env.reset()
            finally:
                if hooked:
                    del w._augment_obs_and_info
            runs.append({k: [np.asarray(x).copy() for x in obs[k]] for k in KEYS})
            env.close()
        a, c = runs
        bad = sum(len(a[k]) != len(c[k]) or any(not np.array_equal(x, y) for x, y in zip(a[k], c[k])) for k in KEYS)
        n = len(a[KEYS[0]])
        print(f"{task}:{ep} frames={n} bad_keys={bad}", flush=True)
        episodes += 1; frames += n; diff += bad
print(f"RESET_HOOK_PARITY={'PASS' if diff == 0 else 'FAIL'} episodes={episodes} frames={frames} diff={diff}", flush=True)
raise SystemExit(0 if diff == 0 else 1)
