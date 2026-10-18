"""检查指定回合数量与完整视频；错误不能伪装成正常任务失败。"""
import json
from pathlib import Path
import sys
import imageio.v2 as imageio

def check_result(run_dir, policy, ckpt, task, episodes, seed):
    root = Path(run_dir) / policy / f'ckpt{ckpt}' / f'seed{seed}'
    progress = json.loads((root / 'progress.json').read_text())
    result = json.loads((root / 'log.json').read_text())
    assert set(progress) == {task}, progress
    assert set(progress[task]) == {str(i) for i in range(episodes)}, progress
    assert all(type(value) is bool for value in progress[task].values()), progress
    rate = sum(progress[task].values()) / episodes
    assert result['success_rate'] == {task: rate} and result['total_success_rate'] == rate, result
    videos = sorted((root / 'videos').glob('*.mp4'))
    assert len(videos) == episodes, videos
    records = []
    for video in videos:
        count = 0
        with imageio.get_reader(video) as reader:
            for frame in reader:
                assert frame.ndim == 3 and frame.shape[-1] == 3
                count += 1
        assert count > 1, video
        records.append(dict(path=str(video), frames=count, bytes=video.stat().st_size))
    summary = dict(pipeline_pass=True, task_successes=sum(progress[task].values()), episodes=episodes, videos=records)
    (root / 'check.json').write_text(json.dumps(summary, indent=2))
    print('EVAL_PASS', json.dumps(summary), flush=True)
    return summary

if __name__ == '__main__':
    run_dir, policy, ckpt, task, episodes, seed = sys.argv[1:]
    check_result(run_dir, policy, ckpt, task, int(episodes), seed)
