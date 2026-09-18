"""检查唯一源码来源、子模块锁定、候选库身份与客户端依赖来源。

本轮评测集是 fork 的 test/primary 700 条注入候选（不是官方 test split），来源链因此比
上一轮多两环：benchmark 换成了 PolicyEvalThirdParty 分支，评测输入换成了 candidates.jsonl。
两环都在这里钉死，任何一环漂移都在起跑前 AssertionError，不会带着错的评测集跑满 700 集。
"""
from importlib.metadata import distribution
import ast
import collections
import json
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
import robomme
import openpi_client
import mani_skill
from robomme.injection_candidates import load_candidates, candidate_key

BENCHMARK = repo / 'third_party/robomme_benchmark'
# benchmark 锁定提交：fork 的 PolicyEvalThirdParty-v2-vail-eval-0917 分支，
# 从 newtask-v2.1refractor 的 77681e106f005f1ff93157f36b973d1005cf0ebe 切出并加了策略侧建环境入口
PINNED = 'b4e97f22fe007078e297205898c07c1acbc69165'
FORK_URLS = (
    'git@github.com:hongzefu/robomme_benchmark_MotionJEPA.git',
    'https://github.com/hongzefu/robomme_benchmark_MotionJEPA.git',
)
CANDIDATES = BENCHMARK / 'artifacts/injection/20260912-contract-v3-10/candidates/candidates.jsonl'
# 候选库身份（load_candidates 内部已复算自洽，这里再钉死「是哪一份」，防止换成 parity 或 smoke run）
IDENTITY_SHA256 = '3b4de03a0b4639181f9c68086aea11e9b51ffcb65bbed701a44f573e74e0beb7'
CONTRACT_SHA256 = 'bfcf4c5984fb89f28c8f04ae918513514ec3d726edb0735d7a22e8587b450da6'
SAMPLING_SHA256 = '97c9af660ca51b7edd238a3a82e2933e0ffa8c6c34882838cc79cb6becaf51d6'
# 单条候选探针：内容级抽查，防止整份文件被换成结构相同但取值不同的伪造品
PROBE = ('RouteStick', 'xhard', 115)
PROBE_SEED = 27500
PROBE_SPEC_SHA256 = 'ae9abaed203437030b6237478f35e80df26d6811e34054e7c944a11ae4126d93'


def body_after_docstring(path: Path) -> str:
    """返回剥掉首个模块 docstring 之后的正文，用于比对同源副本。"""
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    head = tree.body[0] if tree.body else None
    if (isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)):
        return ''.join(source.splitlines(keepends=True)[head.end_lineno:])
    return source


# ---- 源码唯一来源 ----
assert Path(robomme.__file__).resolve() == BENCHMARK / 'src/robomme/__init__.py'
assert Path(openpi_client.__file__).resolve() == repo / 'packages/openpi-client/src/openpi_client/__init__.py'

# ---- submodule 锁定与来源 ----
pinned = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', 'HEAD', 'third_party/robomme_benchmark'], text=True).split()[2]
actual = subprocess.check_output(['git', '-C', str(BENCHMARK), 'rev-parse', 'HEAD'], text=True).strip()
assert actual == pinned == PINNED, (actual, pinned, PINNED)
url = subprocess.check_output(['git', '-C', str(repo), 'config', '--file', '.gitmodules', '--get', 'submodule.third_party/robomme_benchmark.url'], text=True).strip()
assert url in FORK_URLS, url
assert not subprocess.check_output(['git', '-C', str(BENCHMARK), 'status', '--porcelain'], text=True).strip()

# ---- injection_candidates.py 必须与 scripts 侧的读取器同源（除各自 docstring 外正文逐字节相同）----
copy_body = body_after_docstring(BENCHMARK / 'src/robomme/injection_candidates.py')
origin_body = body_after_docstring(BENCHMARK / 'scripts/injection/candidates/io.py')
assert copy_body == origin_body, 'injection_candidates.py 与 scripts/injection/candidates/io.py 已漂移'

# ---- ManiSkill fork ----
origin = json.loads(distribution('mani-skill').read_text('direct_url.json'))
assert origin['url'].removesuffix('.git') == 'https://github.com/YinpeiDai/ManiSkill'
assert origin['vcs_info']['commit_id'] == '07be6fbc66350ddca200abfb0a11b692f078f7fd'

# ---- 评测集：test/primary 700 条 ----
# repo_root 传 BENCHMARK => load_candidates 同时核验 header.sampling_config.sources 里
# 7 份采样源码（4 个任务 + object_generation / route / statechange）的 sha256
header, rows = load_candidates(CANDIDATES, repo_root=BENCHMARK)
assert header['identity_sha256'] == IDENTITY_SHA256, header['identity_sha256']
assert header['contract_sha256'] == CONTRACT_SHA256, header['contract_sha256']
assert header['sampling_config_sha256'] == SAMPLING_SHA256, header['sampling_config_sha256']
assert header['run_id'] == '20260912-contract-v3-10' and header['purpose'] == 'delivery'
primary = [r for r in rows if r['split'] == 'test' and r['role'] == 'primary']
assert len(primary) == 700, len(primary)
groups = collections.Counter((r['task'], r['difficulty']) for r in primary)
assert len(groups) == 14 and set(groups.values()) == {50}, dict(groups)
assert len({candidate_key(r) for r in primary}) == 700
probe = next(r for r in primary if candidate_key(r) == PROBE)
assert probe['seed'] == PROBE_SEED and probe['spec_sha256'] == PROBE_SPEC_SHA256, probe

# ---- 官方 test split 未被污染（fork 未改 env_metadata/test，保留上一轮的探针口径）----
metadata = BENCHMARK / 'src/robomme/env_metadata/test/record_dataset_RouteStick_metadata.json'
episode = next(row for row in json.loads(metadata.read_text())['records'] if row['episode'] == 0)
assert episode['seed'] == 660000 and episode['difficulty'] == 'easy', episode

print('SOURCES_PASS', json.dumps(dict(
    python=sys.executable,
    benchmark=actual,
    robomme=robomme.__file__,
    openpi_client=openpi_client.__file__,
    candidates=dict(run_id=header['run_id'], identity_sha256=header['identity_sha256'],
                    groups=len(groups), primary=len(primary)),
    probe=dict(key=list(PROBE), seed=probe['seed'], spec_sha256=probe['spec_sha256']),
    official_routestick_ep0=episode,
)), flush=True)
