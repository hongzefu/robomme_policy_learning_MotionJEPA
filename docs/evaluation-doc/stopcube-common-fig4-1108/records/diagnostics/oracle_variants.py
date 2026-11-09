"""探索性：to_press 的几种真值参考线（独热、线性、同 hover_probe 折与 λ）：[t]、[t,N]、[t,M]、[t,N,M] 加性、[t, N×M 配对]。
用法：python oracle_variants.py <hover_points.json>
"""
import sys
import numpy as np
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis.rollout_probe import metrics

s = hp.load_samples(sys.argv[1]); y = s["labels"]["to_press"]
oh = hp._one_hot
variants = {"[t]": [oh(s["t"])], "[t,N]": [oh(s["t"]), oh(s["N"])], "[t,M]": [oh(s["t"]), oh(s["M"])],
            "[t,N,M] additive": [oh(s["t"]), oh(s["N"]), oh(s["M"])], "[t, N x M pair]": [oh(s["t"]), oh(s["N"] * 1000 + s["M"])]}
for name, cols in variants.items():
    X = np.hstack(cols)
    p, _, _, _ = hp.crossfit(lambda f: X, s, y[:, None], False)
    m = metrics(y, p[:, 0], s["groups"])
    print(f"{name}: r2={m['r2']:.3f} mae={m['mae']:.1f}")
