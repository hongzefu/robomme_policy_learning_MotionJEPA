"""到位ETA只读估算的负控。"""
import unittest

from scripts.evaluation.watch_60k import estimate_ready


class ReadyEstimateTest(unittest.TestCase):
    def setUp(self):
        self.events = [{"run_id": "run", "source_identity": "delivered-wandb-sha", "step": s, "timestamp": t}
                       for s, t in ((58000, 10000), (58500, 10500), (59000, 11000))]
        self.samples = [{"run_id": "run", "source_identity": "earlier-checkpoint-manifest",
                         "save_seconds": 100, "publish_seconds": 100, "transfer_seconds": 10000, "verify_seconds": 100}]

    def test_delivery_eta_controls_submit(self):
        result = estimate_ready(self.events, self.samples, run_id="run", now=11000)
        self.assertEqual(result["eta_train"], 12000)
        self.assertEqual(result["eta_ready"], 22300)
        self.assertFalse(result["submit_due"])
        self.assertFalse(result["actual_ready"])

    def test_missing_delivery_stale_and_wrong_run_are_unknown(self):
        for events, samples, now, run in ((self.events, [], 11000, "run"),
                                          (self.events, self.samples, 13000, "run"),
                                          (self.events, self.samples, 11000, "other")):
            with self.subTest(run=run, now=now):
                self.assertEqual(estimate_ready(events, samples, run_id=run, now=now)["status"], "UNKNOWN")

    def test_paused_backward_step_and_unstable_rate(self):
        for edit in (lambda e: e[-1].update(paused=True), lambda e: e[-1].update(step=58001),
                     lambda e: e[-1].update(timestamp=10501)):
            events = [dict(e) for e in self.events]
            edit(events)
            self.assertEqual(estimate_ready(events, self.samples, run_id="run", now=11000)["status"], "UNKNOWN")

    def test_explicit_eta_needs_no_training_access(self):
        result = estimate_ready([], [], run_id="run", now=11000, explicit_eta_ready=18000)
        self.assertTrue(result["submit_due"])
        self.assertEqual(result["source"], "user_explicit")


if __name__ == "__main__":
    unittest.main()
