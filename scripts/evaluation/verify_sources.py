"""检查唯一源码来源、子模块锁定、官方 test metadata 与客户端依赖来源（1017-test-xhard5-12per 的 ENV_PIN）。

本轮评测集是 benchmark 仓库 tag ``v2-motionmem-eval-test1017-metadata`` 上的官方 test metadata
（ep0–49 官方三档 + ep50 起追加的 xhard 五档）导出的 1200 局清单，不再是 0918 的注入候选库。
起跑前钉死：子模块 gitlink = tag 所指 commit = 子模块实际 HEAD、子模块工作区 clean、url 是 fork 远端；
``robomme`` 与 ``robomme_xhard`` 都从子模块 ``src/`` 导入；ManiSkill 钉 ``07be6fbc``；
官方 test metadata 的 ep0–49 只有 easy / medium / hard、ep≥50 只有 xhard。任何一环漂移都在起跑前 AssertionError。

判定行：``ENV_PIN=PASS gitlink=<sha> tag=v2-motionmem-eval-test1017-metadata``；另打一行 ``SOURCES_PASS {...}`` 供留档。
可选环境变量 ``EXPECT_ENV_PIN_SHA``：给出则 gitlink 还必须等于它（主会话在 tag 推远端后登记）。
"""
from importlib.metadata import distribution
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
import robomme
import openpi_client

# robomme_xhard 只查落点、不导入：它的 __init__ 会注册 <Task>XHard 子类并连带导入仿真栈（应在建环境的进程里按需导入）
XHARD_ORIGIN = Path(importlib.util.find_spec('robomme_xhard').origin).resolve()

BENCHMARK = repo / 'third_party/robomme_benchmark'
TAG = 'v2-motionmem-eval-test1017-metadata'
FORK_URLS = (
    'git@github.com:hongzefu/robomme_benchmark_MotionJEPA.git',
    'https://github.com/hongzefu/robomme_benchmark_MotionJEPA.git',
)
MANISKILL_URL = 'https://github.com/YinpeiDai/ManiSkill'
MANISKILL_COMMIT = '07be6fbc66350ddca200abfb0a11b692f078f7fd'
BASE_DIFFICULTIES = {'easy', 'medium', 'hard'}
XHARD_DIFFICULTIES = {'xhard12', 'xhard14', 'xhard16', 'xhard18', 'xhard20'}


def git(*args, cwd=repo):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()


# ---- 源码唯一来源 ----
assert Path(robomme.__file__).resolve() == BENCHMARK / 'src/robomme/__init__.py', robomme.__file__
assert XHARD_ORIGIN == BENCHMARK / 'src/robomme_xhard/__init__.py', XHARD_ORIGIN
assert Path(openpi_client.__file__).resolve() == repo / 'packages/openpi-client/src/openpi_client/__init__.py'

# ---- submodule 锁定、tag 与来源 ----
gitlink = git('ls-tree', 'HEAD', 'third_party/robomme_benchmark').split()[2]
actual = git('rev-parse', 'HEAD', cwd=BENCHMARK)
tag_commit = git('rev-parse', f'refs/tags/{TAG}^{{commit}}', cwd=BENCHMARK)
assert actual == gitlink == tag_commit, ('子模块 HEAD / gitlink / tag 不一致', actual, gitlink, tag_commit)
expect = os.environ.get('EXPECT_ENV_PIN_SHA', '').strip()
assert not expect or gitlink == expect, (gitlink, expect)
url = git('config', '--file', '.gitmodules', '--get', 'submodule.third_party/robomme_benchmark.url')
assert url in FORK_URLS, url
assert not git('status', '--porcelain', cwd=BENCHMARK), '子模块工作区不干净'

# ---- ManiSkill fork ----
origin = json.loads(distribution('mani-skill').read_text('direct_url.json'))
assert origin['url'].removesuffix('.git') == MANISKILL_URL, origin['url']
assert origin['vcs_info']['commit_id'] == MANISKILL_COMMIT, origin['vcs_info']

# ---- 官方 test metadata：ep0–49 只有三档，ep≥50 只有 xhard；RouteStick ep0 沿用上一轮探针 ----
metadata_dir = BENCHMARK / 'src/robomme/env_metadata/test'
counts = {}
for path in sorted(metadata_dir.glob('record_dataset_*_metadata.json')):
    payload = json.loads(path.read_text(encoding='utf-8'))
    records = payload['records']
    assert payload['record_count'] == len(records), path
    base = [r for r in records if int(r['episode']) < 50]
    extra = [r for r in records if int(r['episode']) >= 50]
    assert len(base) == 50 and {r['difficulty'] for r in base} <= BASE_DIFFICULTIES, path
    assert {r['difficulty'] for r in extra} <= XHARD_DIFFICULTIES, path
    counts[payload['env_id']] = {'base': len(base), 'xhard': len(extra)}
assert len(counts) == 16, sorted(counts)
episode = next(row for row in json.loads((metadata_dir / 'record_dataset_RouteStick_metadata.json').read_text())['records']
               if row['episode'] == 0)
assert episode['seed'] == 660000 and episode['difficulty'] == 'easy', episode

print('SOURCES_PASS', json.dumps(dict(
    python=sys.executable,
    benchmark=actual,
    robomme=robomme.__file__,
    robomme_xhard=str(XHARD_ORIGIN),
    openpi_client=openpi_client.__file__,
    metadata_counts=counts,
    official_routestick_ep0=episode,
), ensure_ascii=False), flush=True)
print(f'ENV_PIN=PASS gitlink={gitlink} tag={TAG} xhard_records={sum(c["xhard"] for c in counts.values())}', flush=True)
