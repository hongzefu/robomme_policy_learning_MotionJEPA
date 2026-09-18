"""检查唯一源码来源、子模块锁定与客户端依赖来源。"""
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
import robomme
import openpi_client
import mani_skill

assert Path(robomme.__file__).resolve() == repo / 'third_party/robomme_benchmark/src/robomme/__init__.py'
assert Path(openpi_client.__file__).resolve() == repo / 'packages/openpi-client/src/openpi_client/__init__.py'
pinned = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', 'HEAD', 'third_party/robomme_benchmark'], text=True).split()[2]
actual = subprocess.check_output(['git', '-C', str(repo / 'third_party/robomme_benchmark'), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == pinned == '856bc3a189d4172f3f47dbee4424d585f8d78db3'
url = subprocess.check_output(['git', '-C', str(repo), 'config', '--file', '.gitmodules', '--get', 'submodule.third_party/robomme_benchmark.url'], text=True).strip()
assert url in ('git@github.com:RoboMME/robomme_benchmark.git', 'https://github.com/RoboMME/robomme_benchmark.git'), url
assert not subprocess.check_output(['git', '-C', str(repo / 'third_party/robomme_benchmark'), 'status', '--porcelain'], text=True).strip()
origin = json.loads(distribution('mani-skill').read_text('direct_url.json'))
assert origin['url'].removesuffix('.git') == 'https://github.com/YinpeiDai/ManiSkill'
assert origin['vcs_info']['commit_id'] == '07be6fbc66350ddca200abfb0a11b692f078f7fd'
metadata = repo / 'third_party/robomme_benchmark/src/robomme/env_metadata/test/record_dataset_RouteStick_metadata.json'
episode = next(row for row in json.loads(metadata.read_text())['records'] if row['episode'] == 0)
assert episode['seed'] == 660000 and episode['difficulty'] == 'easy', episode
print('SOURCES_PASS', json.dumps(dict(python=sys.executable, benchmark=actual, robomme=robomme.__file__, openpi_client=openpi_client.__file__, episode=episode)), flush=True)
