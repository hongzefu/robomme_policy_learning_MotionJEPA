"""不加载大数组，先核对冻结配置、归一化与完整参数形状。
按 run 根目录名（checkpoint 的父目录名）查 RUNS 表，决定允许的 step 集合、motion 开关、config 名、budget 与两个 sha256 的期望值；
不在表里的 run 直接拒绝——评估只认这张表里登记过的权重。

另一用法（1108 计划 C8 / D1.5）：``check_checkpoint.py --resolved-sha256 <history YAML>`` 按
``train.py::init_history_config`` 同口径复算 resolved_sha256 并打印 ``RESOLVED_SHA256=<sha> yaml=<路径>``，不查 checkpoint。"""
import dataclasses
import hashlib
import json
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

# 1108 计划的池化 run：kind / readout 与 S1 的 wan_pool768.POOL768_ENCODER_KIND / POOL768_READOUT 同值（计划 §3.1 写死）；
# 这里写字面量而不 import，评估 server venv 不必依赖离线池化模块。
WANTP768_KIND = 'wan_fixedpool768'
WANTP768_READOUT = 'wan_raw_timepool8_adaptive6x8_v1'
WANTP768_YAML = 'src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-wantp768.yaml'
# 由 S2 定稿 YAML（S2 分支 b5c5b5b，文件字节 sha256 e049ed62…f5eb）经 `check_checkpoint.py --resolved-sha256 <WANTP768_YAML>`
# 按 resolved_sha256_of_yaml 口径独立复算得出，与 S2 预算值逐位一致（2026-11-08）；正式 run 与冒烟 run 共用。
WANTP768_RESOLVED_SHA256 = '4e70e93373d5b20061a4067c911021100375d2bc28cff276acaf7559b2441910'

# 1112 计划（1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.1 第 15 行）的不池化 run：kind / readout 与
# shared/wan_raw9.py 的 RAW9_ENCODER_KIND / RAW9_READOUT 同值，同样写字面量不 import。
WANRAW9_KIND = 'wan_raw9'
WANRAW9_READOUT = 'wan_raw9_flat16384_s32_v1'
WANRAW9_YAML = 'src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul-8frame-8x8-wanraw9s32.yaml'
# 按 resolved_sha256_of_yaml 口径对定稿 YAML（budget: 80 写死）预算；正式 run 与四条冒烟 run 共用。
# 预算时 S2 YAML 尚未合入：以 wantp768.yaml 为底，按计划 1.1 第 7 行改 dim 16384 / budget 80 / stride 32 /
# store_path …/motion-wan-raw9-s32 / encoder_kind wan_raw9 / readout wan_raw9_flat16384_s32_v1，tokens_per_window: 9
# 紧跟在 dim 之后（OmegaConf.to_yaml 保留键序，位置不同 sha 即不同）。
# ⚠ S2 YAML 合入后须以 `check_checkpoint.py --resolved-sha256 <WANRAW9_YAML>` 复算核对，不符即改此值。
WANRAW9_RESOLVED_SHA256 = '819a001c58b00d4a4110835b913ac1fb58238fb71d272eaa33a429f6dcb53062'


def resolved_sha256_of_yaml(yaml_path) -> str:
    """与 scripts/training/train.py::init_history_config 写 history_config.resolved.yaml 同口径：
    训练侧 get_history_config(name) 即 OmegaConf.load(src/mme_vla_suite/models/config/robomme/<name>)，
    快照字节 = OmegaConf.to_yaml(cfg, resolve=True).encode('utf-8')，resolved_sha256 = 该字节的 sha256。
    runner 的覆盖项（--num_train_steps、--exp_name 等 TrainConfig 字段）不进 history 快照，故不参与计算。"""
    from omegaconf import OmegaConf
    raw = OmegaConf.to_yaml(OmegaConf.load(str(yaml_path)), resolve=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def assert_motion(cfg, run, prov):
    """motion 断言按 run['motion_kind'] 分支：未写 motion_kind 的旧条目走 encoder 分支，断言与改动前逐字相同。"""
    # budget 是模型语义参数（决定 mem_len=512+budget 与 RoPE 位置号），参数树形状对它失明，必须显式钉死
    assert int(cfg.motion.budget) == run['motion_budget'], cfg.motion.budget
    kind = run.get('motion_kind')
    if kind is None:
        assert str(cfg.motion.source_run) == run['motion_source_run'], cfg.motion.source_run
        assert int(cfg.motion.stride) == 16 and int(cfg.motion.window_frames) == 33 and int(cfg.motion.frame_size) == 256
        print(f'BUDGET_CONSISTENT=PASS budget={int(cfg.motion.budget)} source_run={cfg.motion.source_run}', flush=True)
        return
    if kind == WANRAW9_KIND:
        # 不池化分支（1112 计划 1.1 第 15 行）：窗步长 32、每窗 9 个 16384 维 token；prov 须带 raw9 源码 sha256、无 encoder
        assert cfg.motion.get('encoder_kind') == kind, cfg.motion.get('encoder_kind')
        assert cfg.motion.get('readout') == run['readout'], cfg.motion.get('readout')
        assert int(cfg.motion.stride) == 32 and int(cfg.motion.window_frames) == 33 and int(cfg.motion.frame_size) == 256
        assert int(cfg.motion.dim) == 16384, cfg.motion.dim
        assert int(cfg.motion.get('tokens_per_window', 1)) == 9, cfg.motion.get('tokens_per_window')
        assert prov.get('encoder_kind') == kind and prov.get('readout') == run['readout'], \
            (prov.get('encoder_kind'), prov.get('readout'))
        assert prov.get('encoder') is None and prov.get('vae') is not None and prov.get('raw9_source_sha256'), \
            (prov.get('encoder'), prov.get('vae'), prov.get('raw9_source_sha256'))
        print(f'BUDGET_CONSISTENT=PASS budget={int(cfg.motion.budget)} motion_kind={kind} readout={run["readout"]} '
              f'tokens_per_window=9 raw9_source_sha256={str(prov["raw9_source_sha256"])[:16]}', flush=True)
        return
    # 池化分支（1108 计划 1.1 第 6 行）：不要求 motion_source_run，改核快照与 motion_provenance.json 的 kind / readout
    assert kind == WANTP768_KIND, kind
    assert cfg.motion.get('encoder_kind') == kind, cfg.motion.get('encoder_kind')
    assert cfg.motion.get('readout') == run['readout'], cfg.motion.get('readout')
    assert int(cfg.motion.stride) == 16 and int(cfg.motion.window_frames) == 33 and int(cfg.motion.frame_size) == 256
    assert int(cfg.motion.dim) == 768, cfg.motion.dim
    assert prov.get('encoder_kind') == kind and prov.get('readout') == run['readout'], \
        (prov.get('encoder_kind'), prov.get('readout'))
    assert prov.get('encoder') is None and prov.get('vae') is not None and prov.get('pool_source_sha256'), \
        (prov.get('encoder'), prov.get('vae'), prov.get('pool_source_sha256'))
    print(f'BUDGET_CONSISTENT=PASS budget={int(cfg.motion.budget)} motion_kind={kind} readout={run["readout"]} '
          f'pool_source_sha256={str(prov["pool_source_sha256"])[:16]}', flush=True)


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
    # ── 1108 计划（1108-4task-modul-wantp768-50k-plan.html F4 / C8）：Wan 重叠窗零参数池化 768 替换 motion encoder ──
    # bucket HongzeFu/robomme-vla-4task-modul-wantp768-50k-v1，run v2-1600ep-m8x8-modul-wantp768-b128-50k，motion 开启，只评 @50000。
    # motion 分支按 motion_kind（见 assert_motion）：池化分支不要求 motion_source_run，改核快照 encoder_kind / readout。
    # resolved_sha256 按 train.py::init_history_config 的口径从 S2 定稿 YAML 预计算（resolved_sha256_of_yaml）；
    # history YAML 不含步数与 run 名，故正式 run 与冒烟 run 同值。
    'v2-1600ep-m8x8-modul-wantp768-b128-50k': dict(
        steps=('50000',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANTP768_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=160, motion_kind=WANTP768_KIND, readout=WANTP768_READOUT,
    ),
    # G6 冒烟 run（runner --smoke20 --run-name w768-smoke20，num_train_steps=20，按 step == num_train_steps - 1
    # 规则最后一份 ckpt 为 19）；G6b 用它走一次真实评估入口，跑完即删。与正式条目只差 run 名与步数。
    'w768-smoke20': dict(
        steps=('19',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANTP768_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=160, motion_kind=WANTP768_KIND, readout=WANTP768_READOUT,
    ),
    # ── 1112 计划（1112-4task-modul-wanraw9-s32-50k-plan.html 第二部分 1.1 第 15 行）：Wan raw latent 不池化，
    # 每窗 9 个 16384 维 token、窗步长 32、motion.budget = 80 窗；同一 4task-v2-1600ep 库，norm_stats 与 v11 同一份。
    # 正式 run 只评 @50000；冒烟 run（runner --smoke20，num_train_steps=20，最后一份 ckpt 为 19）供 G6b 走一次真实评估入口，
    # 未触发 L 级用 r9-smoke20，触发 S5-L<k> 则为 r9-smoke20-l<k>，跑完即删。
    # 正式 run（bucket HongzeFu/robomme-4task-modul-wanraw9-s32-50k-v1）
    'v2-1600ep-m8x8-modul-wanraw9s32-b128-50k': dict(
        steps=('50000',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANRAW9_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=80, motion_kind=WANRAW9_KIND, readout=WANRAW9_READOUT,
    ),
    # 冒烟 run r9-smoke20
    'r9-smoke20': dict(
        steps=('19',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANRAW9_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=80, motion_kind=WANRAW9_KIND, readout=WANRAW9_READOUT,
    ),
    # 冒烟 run r9-smoke20-l1
    'r9-smoke20-l1': dict(
        steps=('19',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANRAW9_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=80, motion_kind=WANRAW9_KIND, readout=WANRAW9_READOUT,
    ),
    # 冒烟 run r9-smoke20-l2
    'r9-smoke20-l2': dict(
        steps=('19',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANRAW9_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=80, motion_kind=WANRAW9_KIND, readout=WANRAW9_READOUT,
    ),
    # 冒烟 run r9-smoke20-l3
    'r9-smoke20-l3': dict(
        steps=('19',), motion=True, config='mme_vla_suite_b128_80k', budget=512,
        resolved_sha256=WANRAW9_RESOLVED_SHA256,
        norm_stats_sha256='856c75ea504bd104c552027987b98a512d2d0b406738a7a8a500ada96d8ed173',
        motion_budget=80, motion_kind=WANRAW9_KIND, readout=WANRAW9_READOUT,
    ),
}


def main(argv):
    if len(argv) == 3 and argv[1] == '--resolved-sha256':
        print(f'RESOLVED_SHA256={resolved_sha256_of_yaml(argv[2])} yaml={argv[2]}', flush=True)
        return
    checkpoint = Path(argv[1]).resolve()
    run = RUNS.get(checkpoint.parent.name)
    assert run is not None, f'未登记的 run 根目录 {checkpoint.parent.name}，已登记: {sorted(RUNS)}'
    assert checkpoint.name in run['steps'], (checkpoint, run['steps'])
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
        prov = json.loads((checkpoint.parent / 'motion_provenance.json').read_text(encoding='utf-8'))
        assert_motion(cfg, run, prov)
    model_config = dataclasses.replace(get_config(run['config']).model, history_config=cfg, use_history=True)
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
    print(f"HISTORY_CONFIG=PASS NORM_STATS=PASS {'MOTION_ENABLED=PASS' if run['motion'] else 'MOTION_DISABLED=PASS'} config={run['config']} step={checkpoint.name} budget={int(cfg.budget)}", flush=True)


if __name__ == '__main__':
    main(sys.argv)
