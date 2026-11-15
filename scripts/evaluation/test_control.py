"""执行真实 evaluate 控制流，验证单回合、异常、恢复和收尾。"""
import ast
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'examples/robomme/eval_692.py'

class ControlTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        tree.body = [node for node in tree.body if getattr(node, 'name', None) in {'Args', 'setup_save_directory', 'setup_log_dict', 'evaluate', 'summarize', 'load_plan', 'episode_deadline'}]
        self.visited = []
        self.closed = 0
        self.outcomes = {}
        self.make_error = False
        owner = self
        class Runner:
            num_episodes = 3
            def __init__(self, *args, **kwargs): pass
            def make_env(self, episode_id):
                self.episode_id = episode_id
                owner.visited.append(episode_id)
                if owner.make_error: raise RuntimeError('建环境失败')
            def close_env(self): owner.closed += 1
        class Evaluator:
            def __init__(self, *args): pass
            def eval_each_episode(self, runner, *args):
                value = owner.outcomes.get(runner.episode_id, 'success')
                if isinstance(value, Exception): raise value
                return value
        self.ns = dict(dataclasses=dataclasses, json=json, os=os, shutil=shutil, Path=Path, contextlib=contextlib, signal=signal,
                       check_args=lambda _: None, TASK_NAME_LIST=['RouteStick'], EnvRunner=Runner,
                       EpisodeEvaluator=Evaluator, time=SimpleNamespace(sleep=lambda _: None))
        exec(compile(tree, str(SOURCE), 'exec'), self.ns)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = self.ns['Args'](save_dir=self.tmp.name, only_tasks='RouteStick', max_episodes=1)

    def run_eval(self):
        self.ns['evaluate'](self.args)
        root = self.ns['setup_save_directory'](self.args)
        return json.loads((root / 'progress.json').read_text()), json.loads((root / 'log.json').read_text())

    def test_single_episode(self):
        progress, result = self.run_eval()
        self.assertEqual(progress, {'RouteStick': {'0': True}})
        self.assertEqual(self.visited, [0])
        self.assertEqual(self.closed, 1)
        self.assertEqual(result['total_success_rate'], 1)

    def test_inference_exception_finishes(self):
        self.outcomes[0] = RuntimeError('推理失败')
        progress, result = self.run_eval()
        self.assertEqual(progress['RouteStick']['0'], 'error')
        self.assertEqual(result['total_success_rate'], 0)
        self.assertEqual(self.closed, 1)

    def test_make_exception_is_recorded_and_closed(self):
        self.make_error = True
        progress, result = self.run_eval()
        self.assertEqual(progress['RouteStick']['0'], 'error')
        self.assertEqual(result['total_success_rate'], 0)
        self.assertEqual(self.closed, 1)

    def test_unknown_is_error(self):
        self.outcomes[0] = 'unknown'
        progress, result = self.run_eval()
        self.assertEqual(progress['RouteStick']['0'], 'error')
        self.assertEqual(result['total_success_rate'], 0)

    def test_resume_skips_completed(self):
        root = self.ns['setup_save_directory'](self.args)
        (root / 'progress.json').write_text(json.dumps({'RouteStick': {'0': False}}))
        progress, result = self.run_eval()
        self.assertEqual(self.visited, [])
        self.assertEqual(result['total_success_rate'], 0)

    def test_negative_limit(self):
        self.args.max_episodes = -1
        with self.assertRaises(ValueError): self.run_eval()

    def test_cli_accepts_692_args(self):
        # 1114 计划 S4-10：run_shard.sh 以 --args.max_steps=2000 与 --args.demo_prefix_store 调 eval_692.py，server 无需改动
        import tyro
        parsed = []
        def evaluate(args: self.ns['Args']): parsed.append(args)
        tyro.cli(evaluate, args=['--args.max_steps=2000', '--args.demo_prefix_store=/abs/store',
                                 '--args.episode_plan=/abs/plan.json', '--args.episode_wall_s=2400'])
        self.assertEqual((parsed[0].max_steps, parsed[0].demo_prefix_store, parsed[0].episode_wall_s),
                         (2000, '/abs/store', 2400))

    def test_client_imports_692_modules(self):
        # 692 客户端只从改名后的 692 副本取 utils / env_runner，不触碰 v4 官方客户端
        imports = {(n.module, a.name) for n in ast.walk(ast.parse(SOURCE.read_text()))
                   if isinstance(n, ast.ImportFrom) for a in n.names}
        self.assertIn(('env_runner_692', 'SpecEnvRunner'), imports)
        self.assertIn(('utils_692', 'RolloutRecorder'), imports)
        self.assertFalse({m for m, _ in imports} & {'utils', 'env_runner'})

if __name__ == '__main__': unittest.main()
