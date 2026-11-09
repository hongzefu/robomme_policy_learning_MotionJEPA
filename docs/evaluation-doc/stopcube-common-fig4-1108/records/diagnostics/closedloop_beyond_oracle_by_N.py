"""探索性：闭环（各模型自己的轨迹，Q4）上「[t,N]（独热×100）+ 组特征」相对只用 [t,N] 的局级 MAE 增量，按 N≤3／N≥4 分档（模型内比较）。
复用 closedloop_beyond_oracle.make_samples。组：Text 第 9 层、History 第 8 层。用法：python closedloop_beyond_oracle_by_N.py <rollout 产物根>
"""
import os, sys
from pathlib import Path
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from closedloop_beyond_oracle import make_samples, COL
from scripts.interpret.analysis import hover_probe as hp
from scripts.interpret.analysis import rollout_probe as rp
from scripts.interpret.analysis.rollout_attention import load_align
from scripts.interpret.analysis.oracle_press_prob import bootstrap

root = Path(sys.argv[1]); align = load_align(root)
for model in ("ctx", "mot"):
    s, prefix, _, _ = make_samples(model, root, align["models"][model])
    T = np.hstack([hp._one_hot(s["t"]), hp._one_hot(s["N"])]) * 100.0
    base, _, _, _ = hp.crossfit(lambda f: T, s, s["y"][:, None], False)
    for g, l in (("text", 9), ("history", 8)):
        X = prefix[:, l + 1, COL[g]].astype(np.float64)
        def f(fold):
            tr = s["outer"] != fold; pca = hp.fit_pca(X[tr], rp.episode_weights(s["groups"][tr]))
            return np.hstack([T, hp.pca_transform(pca, X)])
        p, _, _, _ = hp.crossfit(f, s, s["y"][:, None], False)
        for tag, mask in (("N<=3", s["N"] <= 3), ("N>=4", s["N"] >= 4)):
            eb = rp.episode_mae(s["y"][mask], base[mask, 0], s["groups"][mask]); ep = rp.episode_mae(s["y"][mask], p[mask, 0], s["groups"][mask])
            d = bootstrap([eb[k] - ep[k] for k in eb], 2000, 42)
            print(f"{model} {g} L{l} {tag}: gain {d['mean']:+.1f}[{d['ci95'][0]:.1f},{d['ci95'][1]:.1f}] n_eps={d['n_episodes']}")
