"""验证成功、正常失败、错误和额外回合的验收边界。"""
import json
from pathlib import Path
import tempfile
import unittest
import imageio.v2 as imageio
import numpy as np
from check_result import check_result

class ResultTests(unittest.TestCase):
    def check(self, value, extra=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'test-policy/ckpt59999/seed7'
            (root / 'videos').mkdir(parents=True)
            progress = {'RouteStick': {'0': value}}
            if extra: progress['RouteStick']['1'] = False
            (root / 'progress.json').write_text(json.dumps(progress))
            rate = float(value is True)
            (root / 'log.json').write_text(json.dumps(dict(success_rate={'RouteStick': rate}, total_success_rate=rate)))
            with imageio.get_writer(root / 'videos/episode.mp4', fps=5) as writer:
                for brightness in (0, 255, 32): writer.append_data(np.full((32, 32, 3), brightness, dtype=np.uint8))
            return check_result(tmp, 'test-policy', '59999', 'RouteStick', 1, '7')
    def test_success(self): self.assertEqual(self.check(True)['task_successes'], 1)
    def test_normal_failure(self): self.assertEqual(self.check(False)['task_successes'], 0)
    def test_error_rejected(self):
        with self.assertRaises(AssertionError): self.check('error')
    def test_extra_rejected(self):
        with self.assertRaises(AssertionError): self.check(True, extra=True)

if __name__ == '__main__': unittest.main()
