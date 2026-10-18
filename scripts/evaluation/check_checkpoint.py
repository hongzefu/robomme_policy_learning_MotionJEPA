"""不加载大数组，先核对冻结配置、归一化与完整参数形状。
按 run 根目录名（checkpoint 的父目录名）查 RUNS 表，决定允许的 step 集合、motion 开关、config 名、budget 与两个 sha256 的期望值；
不在表里的 run 直接拒绝——评估只认这张表里登记过的权重。

1017-test-xhard5-12per：
* 两条新 run（512 = mixed8-200-m8x8-modul-b128-80k 登记 60000；512+motion = mixed8-200-m8x8-modul-motion-b128-80k
  登记 10000 / 60000）按主会话 2026-10-18 给定值预登记（键为 bucket 名），motion run 的 motion_source_run 已按快照原文登记；
* 官方 79999（无 history_config.resolved.yaml）走 ``official=True`` 分支：核 run 根 ``history_config.txt`` 的内容与 sha256、
  step 目录逐文件 sha256（登记值同样由主会话在 NFS 副本经 ``OFFICIAL_CKPT_SHA`` 核对后填入），再以
  ``history_config`` 字符串建模型做同一套参数树校验。"""
import dataclasses
import hashlib
from pathlib import Path
import sys

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import orbax.checkpoint as ocp
from mme_vla_suite.training.config import get_config
from mme_vla_suite.policies.policy_config import _load_resolved_snapshot
import mme_vla_suite.policies.policy_config as policy_source
import openpi.models.model as model_source

repo = Path(__file__).resolve().parents[2]
assert Path(policy_source.__file__).resolve() == repo / 'src/mme_vla_suite/policies/policy_config.py'
assert Path(model_source.__file__).resolve() == repo / 'src/openpi/models/model.py'
print('SERVER_SOURCE', sys.executable, policy_source.__file__, model_source.__file__, flush=True)

# 登记过的权重：键 = run 根目录名（download.sh 的落点目录名）
RUNS = {
    # bucket HongzeFu/robomme-vla-modul-60k-v1，run v2-1600ep-m8x8-modul-b128-60k，motion 关闭，61 叶；
    # 59999 = primary700-gl 基线，50000 = 与 motion 50000 同步数对照（2026-09-18 加入；norm_stats 两步同一份 sha）
    'robomme-vla-modul-60k-v1': dict(
        steps=('59999', '50000'), motion=False, config='mme_vla_suite', budget=512,
        resolved_sha256='804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
    ),
    # bucket HongzeFu/robomme-vla-modul-motion-80k-v1，run v2-1600ep-m8x8-modul-motion-b128-80k，motion 开启，65 叶；
    # resolved sha 与 bucket 根 history_config.resolved.sha256 及 motion_provenance.json.resolved_sha256 三方一致（2026-09-18 核）；
    # norm_stats 与 60k 同一份（开 motion 不重算归一化）
    'robomme-vla-modul-motion-80k-v1': dict(
        steps=('50000', '79999'), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256='9650f225fe42b802262f93b71139cebbda934dc5ccfaa849fa4dadaf30d73110',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=160,
        motion_source_run='wan-full1600-filter2-b176x4-72ep-a/checkpoint_epoch_72.pt#encoder',
    ),
    # bucket HongzeFu/robomme-vla-modul-2048-80k-v1，run v2-1600ep-m32x8x8-modul-b128-80k，motion 关闭，61 叶；
    # 与上面两条的唯一差别是感知上下文预算：budget 2048 = 32 帧 × token_per_image 64（上面是 512 = 8 帧 × 64）。
    # budget 只改运行期序列长度、不改参数树形状（位置编码表是无参数的 PosEmb3D，modulation 下记忆 token 走交叉注意力），
    # 所以参数树校验对它失明 —— 必须靠下面的 budget 断言钉死。norm_stats 全 16 个 step 同一份 sha（2026-09-22 核）。
    'robomme-vla-modul-2048-80k-v1': dict(
        steps=('5000', '10000', '15000', '20000', '25000', '30000', '35000', '40000',
               '45000', '50000', '55000', '60000', '65000', '70000', '75000', '79999'),
        motion=False, config='mme_vla_suite_b128_80k', budget=2048,
        resolved_sha256='91512306b3aaaaa541d248bc5dfaf8be2849632cd3b9e89a226207c09c6501c0',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
    ),
    # bucket HongzeFu/robomme-vla-modul-1024-80k-v1，run v2-1600ep-m16x8x8-modul-b128-80k，motion 关闭，61 叶；
    # 与 2048 条目只差感知上下文预算：budget 1024 = 16 帧 × token_per_image 64。resolved sha 与 bucket 根
    # history_config.resolved.sha256 及 motion_provenance.json.resolved_sha256 三方一致（2026-09-25 核）；norm_stats 沿用同一份。
    'robomme-vla-modul-1024-80k-v1': dict(
        steps=('5000', '10000', '15000', '20000', '25000', '30000', '35000', '40000',
               '45000', '50000', '55000', '60000', '65000', '70000', '75000', '79999'),
        motion=False, config='mme_vla_suite_b128_80k', budget=1024,
        resolved_sha256='9d7a18fe2cdb720dce21dc7d1c03b7a905fc6f65d98758a582268e5fa9e5390b',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
    ),
    # bucket HongzeFu/robomme-vla-modul-4096-80k-v1，run v2-1600ep-m64x8x8-modul-b128-80k，motion 关闭，61 叶；
    # budget 4096 = 64 帧 × 64。FrameSampMemory 的 max_steps=4096 硬上界限制的是全域帧号（本口径最大约 2518），与 budget 无关。
    # 两个 sha 三方一致（2026-09-25 核）。
    'robomme-vla-modul-4096-80k-v1': dict(
        steps=('5000', '10000', '15000', '20000', '25000', '30000', '35000', '40000',
               '45000', '50000', '55000', '60000', '65000', '70000', '75000', '79999'),
        motion=False, config='mme_vla_suite_b128_80k', budget=4096,
        resolved_sha256='9df78b4dac2acdc43fbce5040a88014805df1b50bb803feba027559fc31f96a3',
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
    ),
    # ── 1017-test-xhard5-12per（主会话 2026-10-18 预登记；键 = bucket 名 = 本地落点 v1-store/models/<bucket 名>/<step>）──
    # bucket HongzeFu/robomme-vla-mixed8-200-modul-512-80k-v1，run mixed8-200-m8x8-modul-b128-80k，motion 关闭。
    # resolved sha 与上面 robomme-vla-modul-60k-v1 相同（budget 512、modulation、token_per_image 64，NFS 本地快照只读核过）。
    'robomme-vla-mixed8-200-modul-512-80k-v1': dict(
        steps=('60000',), motion=False, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256='804b25382af668d67c8f8ea2d1cca414aee9a184ec737dcad704bdff253a2f92',
        norm_stats_sha256='36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394',
    ),
    # bucket HongzeFu/robomme-vla-mixed8-200-modul-motion-80k-v1，run mixed8-200-m8x8-modul-motion-b128-80k，motion 开启；
    # 10000 供 10k 冒烟、60000 供正式轮。motion_budget 160 是计划 T13 定值；motion_source_run 已登记：主会话 2026-10-18
    # 从 bucket 根 history_config.resolved.yaml 读出原文（同文件 motion.budget=160、demo_min_real_frames=17、
    # demo_tail_pad=repeat_last、stride=16、window_frames=33、online_gpu=4）。
    'robomme-vla-mixed8-200-modul-motion-80k-v1': dict(
        steps=('60000', '10000'), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256='53f070ff5363efff6bc4dc3c6c0ee602095bc1fa6ebcd0651dd7bf3f9ca0d0c3',
        norm_stats_sha256='36ee9acdea4c8383b50c7b07b45a740efb3fa073f6e60455977d5a6fb3204394',
        motion_budget=160,
        motion_source_run='wan-mixed8-200-filter2-stop075-b88x8-72ep-a/checkpoint_epoch_72.pt#encoder',
    ),
    # 官方 FrameSamp+Modulation 79999（NFS v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999）。
    # history_config.txt 内容与 sha 取自 aspen 本机 memER 副本（2026-10-18 只读实测，31 字节）；
    # 逐文件 sha256 为空 = 未登记，本脚本对官方权重直接拒绝，直到主会话把 NFS 副本经 OFFICIAL_CKPT_SHA 核对后的值填入
    # 官方 modul zip sha256 86ef2b63…（HF Yinpei/mme_vla_suite perceptual-framesamp-modul/79999.zip LFS，与
    # Yinpei/perceptual-framesamp-modul@c0f565dd 的 2bfde48a… 为同一权重的不同打包，18 个条目 CRC32 与大小逐条相同），
    # 逐文件 sha 由主会话登记。
    'perceptual-framesamp-modul': dict(
        official=True, steps=('79999',), motion=False, config='mme_vla_suite', budget=512,
        history_config_txt='perceptual-framesamp-modul.yaml',
        history_config_sha256='a675d785d239e7e142be25318313c20bae66be5db6a525c589dce9a95f20e1fc',
        file_sha256={},   # 相对 step 目录的路径 -> sha256；待登记
    ),
}

checkpoint = Path(sys.argv[1]).resolve()
run = RUNS.get(checkpoint.parent.name)
assert run is not None, f'未登记的 run 根目录 {checkpoint.parent.name}，已登记: {sorted(RUNS)}'
assert checkpoint.name in run['steps'], (checkpoint, run['steps'])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def check_param_tree(model_config) -> list:
    model = nnx.eval_shape(model_config.create, jax.random.key(0))
    expected = flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(checkpoint / 'params')
    assert set(metadata.tree) == {'params'}
    actual = flatten_dict(metadata.tree['params'])
    if all(key[-1] == 'value' for key in actual):
        actual = {key[:-1]: value for key, value in actual.items()}
    assert set(actual) == set(expected), (set(expected) - set(actual), set(actual) - set(expected))
    for key, value in expected.items():
        assert value.shape == actual[key].shape, (key, value.shape, actual[key].shape)
    motion_leaves = sorted('/'.join(map(str, k)) for k in actual if 'motion_pos_proj' in '/'.join(map(str, k)) or 'motion_encoder_static' in '/'.join(map(str, k)))
    assert bool(motion_leaves) == run['motion'], motion_leaves
    print(f'PARAM_TREE_EXACT=PASS n_model={len(expected)} n_ckpt={len(actual)} missing=0 extra=0 shape_mismatch=0 motion_leaves={len(motion_leaves)}', flush=True)
    return motion_leaves


if run.get('official'):
    # 官方权重：没有 resolved 快照，serve_policy 走 history_config.txt 兼容路径（policy_config.create_trained_policy 非 strict 分支）
    assert not (checkpoint.parent / 'history_config.resolved.yaml').exists(), '官方权重不应带 resolved 快照'
    txt = checkpoint.parent / 'history_config.txt'
    raw = txt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == run['history_config_sha256'], 'history_config.txt sha256 不符'
    assert raw.decode('utf-8') == run['history_config_txt'], raw
    assert run['file_sha256'], '官方 79999 的逐文件 sha256 尚未登记（主会话核 OFFICIAL_CKPT_SHA 后填入 RUNS），拒绝放行'
    files = sorted(str(p.relative_to(checkpoint)) for p in checkpoint.rglob('*') if p.is_file())
    assert files == sorted(run['file_sha256']), (sorted(set(run['file_sha256']) ^ set(files)))
    for rel in files:
        assert sha256_file(checkpoint / rel) == run['file_sha256'][rel], rel
    print(f'OFFICIAL_CKPT_FILES=PASS files={len(files)}', flush=True)
    model_config = dataclasses.replace(get_config(run['config']).model, history_config=run['history_config_txt'], use_history=True)
    check_param_tree(model_config)
    print(f"HISTORY_CONFIG=PASS MOTION_DISABLED=PASS official=1 config={run['config']} step={checkpoint.name} history_config={run['history_config_txt']}", flush=True)
    sys.exit(0)

cfg, motion_enabled = _load_resolved_snapshot(checkpoint.parent)
assert motion_enabled == run['motion'], (motion_enabled, run['motion'])
# budget 逐 run 钉死：它是模型语义参数（决定 static_image_emb 的序列长度与取帧数 budget//token_per_image），
# 参数树形状对它失明，配错了不会被 PARAM_TREE_EXACT 拦住
assert int(cfg.budget) == run['budget'], (cfg.budget, run['budget'])
assert cfg.token_per_image == 64 and cfg.integration_type == 'modulation'
assert cfg.perceptual_memory.type == 'frame_sampling'
assert hashlib.sha256((checkpoint.parent / 'history_config.resolved.yaml').read_bytes()).hexdigest() == run['resolved_sha256']
assert hashlib.sha256((checkpoint / 'assets/robomme/norm_stats.json').read_bytes()).hexdigest() == run['norm_stats_sha256']
if run['motion']:
    # budget 是模型语义参数（决定 mem_len=512+budget 与 RoPE 位置号），参数树形状对它失明，必须显式钉死
    assert int(cfg.motion.budget) == run['motion_budget'], cfg.motion.budget
    assert run['motion_source_run'] is not None, f"motion_source_run 未登记；快照里是 {cfg.motion.source_run!r}，核对后填入 RUNS"
    assert str(cfg.motion.source_run) == run['motion_source_run'], cfg.motion.source_run
    assert int(cfg.motion.stride) == 16 and int(cfg.motion.window_frames) == 33 and int(cfg.motion.frame_size) == 256
    print(f'BUDGET_CONSISTENT=PASS budget={int(cfg.motion.budget)} source_run={cfg.motion.source_run}', flush=True)
model_config = dataclasses.replace(get_config(run['config']).model, history_config=cfg, use_history=True)
check_param_tree(model_config)
print(f"HISTORY_CONFIG=PASS NORM_STATS=PASS {'MOTION_ENABLED=PASS' if run['motion'] else 'MOTION_DISABLED=PASS'} config={run['config']} step={checkpoint.name} budget={int(cfg.budget)}", flush=True)
