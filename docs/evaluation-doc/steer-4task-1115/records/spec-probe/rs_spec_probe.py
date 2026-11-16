"""RouteStick episode_spec 实测：自然局 vs 原样 spec vs 翻第 j 步绕向 / 左右，比对布局与示范帧。
只读仿真，不写仓库。用法：python rs_spec_probe.py <seed> <difficulty> <j>
"""
import copy, hashlib, json, math, sys
import numpy as np
import torch
import gymnasium as gym
import robomme.robomme_env  # noqa: 注册环境
from robomme.robomme_env.RouteStick import NATIVE_SAMPLING, RouteStick
from robomme.env_record_wrapper.episode_config_resolver import _wrap_env
from robomme.injection_candidates import RUNTIME

seed, diff, J = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
CFG = {"easy": ((2, 3), False), "medium": ((4, 5), False), "hard": ((4, 7), True), "xhard": ((8, 10), True)}
ROUTE = [0, 2, 4, 6, 8]


def replay(seed, diff):
    (lo, hi), back = CFG[diff]
    g = torch.Generator(); g.manual_seed(seed)
    theta = (torch.rand(1, generator=g).item() * 60) - 30
    rgbs = [torch.rand(3, generator=g).tolist() for _ in range(4)]
    steps = int(torch.randint(lo, hi + 1, (1,), generator=g).item())
    idx = [torch.randint(0, 5, (1,), generator=g).item()]
    for _ in range(steps):
        cur = idx[-1]; prev = idx[-2] if len(idx) > 1 else None
        nb = [n for n in (cur - 1, cur + 1) if 0 <= n < 5]
        c = nb if (back or prev is None) else ([n for n in nb if n != prev] or nb)
        idx.append(c[torch.randint(0, len(c), (1,), generator=g).item()])
    dirs = ["clockwise" if torch.rand(1, generator=g).item() < 0.5 else "counterclockwise" for _ in range(steps)]
    return {"task": "RouteStick", "episode": 0, "difficulty": diff,
            "layout": {"rotation_deg": theta, "obstacle_rgb": rgbs},
            "objects": {"L": steps},
            "actions": {"nodes": [ROUTE[i] for i in idx], "node_slots": idx, "directions": dirs}}


def make(spec, sampling=None):
    kw = dict(RUNTIME["kwargs"]); kw.update(seed=seed, difficulty=diff)
    if spec is not None:
        kw["episode_spec"] = spec
        kw["sampling_config"] = sampling or {"parameters": copy.deepcopy(NATIVE_SAMPLING["parameters"]),
                                              "positions": copy.deepcopy(NATIVE_SAMPLING["positions"])}
    env = gym.make("RouteStick", **kw)
    return _wrap_env(env, env_id="RouteStick", action_space="joint_angle", gui_render=False, max_steps_without_demo=3002)


def run(spec, sampling=None, tag=""):
    env = make(spec, sampling)
    obs, info = env.reset()
    u = env.unwrapped
    btn = np.array([np.asarray(b.pose.p).reshape(-1)[:2] for b in u.buttons_grid])
    frames = None
    for k in ("front_rgb_list", "image", "images"):
        if isinstance(obs, dict) and k in obs:
            frames = obs[k]; break
    frames = np.asarray(frames) if frames is not None else None
    res = {"tag": tag, "route": [u.buttons_grid.index(b) for b in u.selected_buttons], "dirs": list(u.swing_directions),
           "btn_md5": hashlib.md5(np.round(btn, 6).tobytes()).hexdigest(), "obs_keys": list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__,
           "n_frames": None if frames is None else int(len(frames))}
    env.close()
    return res, frames


base = replay(seed, diff)
print("SPEC", json.dumps(base["actions"]), "theta", round(base["layout"]["rotation_deg"], 3))
nat, f_nat = run(None, tag="natural")
same, f_same = run(base, tag="spec_same")
v_sw = copy.deepcopy(base); d = v_sw["actions"]["directions"]
d[J - 1] = "counterclockwise" if d[J - 1] == "clockwise" else "clockwise"
sw, f_sw = run(v_sw, tag="flip_swing_j")

# 只翻左右：第 j 步改去另一侧，之后每步保持原相对方向，出界则反弹；用 backtrack=True 的采样覆盖
v_sd = copy.deepcopy(base); s = list(base["actions"]["node_slots"])
alt = s[J - 1] - (s[J] - s[J - 1])
side_res = None
if 0 <= alt < 5:
    new = s[:J] + [alt]
    for i in range(J, len(s) - 1):
        step = s[i + 1] - s[i]; nxt = new[-1] + step
        if not 0 <= nxt < 5:
            nxt = new[-1] - step
        new.append(nxt)
    v_sd["actions"]["node_slots"] = new; v_sd["actions"]["nodes"] = [ROUTE[i] for i in new]
    samp = {"parameters": copy.deepcopy(NATIVE_SAMPLING["parameters"]), "positions": copy.deepcopy(NATIVE_SAMPLING["positions"])}
    cfgs = copy.deepcopy(RouteStick.configs); cfgs[diff] = dict(cfgs[diff]); cfgs[diff]["backtrack"] = True
    samp["parameters"]["configs"] = cfgs
    try:
        side_res, f_sd = run(v_sd, samp, tag="flip_side_j")
    except Exception as e:
        side_res, f_sd = {"tag": "flip_side_j", "error": f"{type(e).__name__}: {e}"}, None
else:
    f_sd = None
    side_res = {"tag": "flip_side_j", "skipped": "端点出界"}

for r in (nat, same, sw, side_res):
    print(json.dumps(r, ensure_ascii=False))


def first_diff(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if not np.array_equal(a[i], b[i]):
            return i
    return n if len(a) == len(b) else f"len {len(a)} vs {len(b)}"


print("FRAMES natural==spec_same:", first_diff(f_nat, f_same), "/", None if f_nat is None else len(f_nat))
print("FRAMES spec_same vs flip_swing first diff frame:", first_diff(f_same, f_sw))
print("FRAMES spec_same vs flip_side first diff frame:", first_diff(f_same, f_sd))


# 示范实际绕向：按 eef xy 相对 prev→curr 的叉积均值（同 direction_fail），逐段
def demo_swings(spec, sampling=None):
    env = make(spec, sampling); obs, _ = env.reset(); u = env.unwrapped
    eef = np.asarray(obs["eef_state_list"])[:, :2]
    pos = {i: np.asarray(b.pose.p).reshape(-1)[:2] for i, b in enumerate(u.buttons_grid)}
    route = [u.buttons_grid.index(b) for b in u.selected_buttons]
    out = []
    for j in range(1, len(route)):
        a, b = pos[route[j - 1]], pos[route[j]]; seg = eef[50 * (j - 1) + 1: 50 * j + 1]
        lv = b - a; cr = np.mean(lv[0] * (seg[:, 1] - a[1]) - lv[1] * (seg[:, 0] - a[0]))
        out.append("clockwise" if cr > 0 else "counterclockwise")
    env.close(); return out


print("DEMO swings same :", demo_swings(base), "spec:", base["actions"]["directions"])
print("DEMO swings flipJ:", demo_swings(v_sw), "spec:", v_sw["actions"]["directions"])
