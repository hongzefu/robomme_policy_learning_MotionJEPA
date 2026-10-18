"""验证指定回合的原生渲染与一步执行。"""
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'examples/robomme'))
from env_runner import EnvRunner

out = Path(sys.argv[1])
task = sys.argv[2]
out.mkdir(parents=True, exist_ok=True)   # 续跑（ALLOW_RESUME）时目录已存在，重跑探针覆盖即可
runner = EnvRunner(task, str(out), max_steps=1300)
try:
    runner.make_env(0)
    initial = runner.get_init_obs()
    frames = [initial['images'][-1], initial['wrist_images'][-1]]
    obs, _, status = runner.step(initial['states'][-1])
    frames.extend(obs[:2])
    stats = []
    for index, frame in enumerate(frames):
        assert frame is not None and frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3 and float(frame.std()) > 1
        imageio.imwrite(out / f'camera_{index}.png', frame)
        stats.append(dict(shape=list(frame.shape), std=float(frame.std())))
    result = dict(render_pass=True, task=task, episode=0, status=str(status), frames=stats)
    (out / 'render.json').write_text(json.dumps(result, indent=2))
    print('RENDER_PASS', json.dumps(result), flush=True)
finally:
    runner.close_env()
