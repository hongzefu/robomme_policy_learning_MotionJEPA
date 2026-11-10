# 两模型都原样整局成功的接收局上的并排读数

```
RS_SWAP_INTERSECT models=motion,wan recv=10 motion:Z=6/10,Rnd=6/10,A=9/10,Bsw_swing=8/10,Bside_dir=8/10,C_own=24/30,D1_swing=16/30,D2_dir=18/70,D2_abs=0/70,W0_swing=3/10,W0_dir=4/10 wan:Z=1/10,Rnd=4/10,A=10/10,Bsw_swing=0/10,Bside_dir=0/10,C_own=30/30,D1_swing=0/30,D2_dir=0/70,D2_abs=0/70,W0_swing=0/10,W0_dir=0/10
```

接收局（10 局）：[0, 2, 3, 6, 10, 14, 18, 20, 28, 29]

| 组 | motion 局数 | motion 第一步成功 | motion own | motion donor_dir | motion donor_abs | motion donor_swing | wan 局数 | wan 第一步成功 | wan own | wan donor_dir | wan donor_abs | wan donor_swing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Z | 10 | 6 | 7 | 0 | 0 | 0 | 10 | 1 | 7 | 0 | 0 | 0 |
| Rnd | 10 | 6 | 7 | 0 | 0 | 0 | 10 | 4 | 6 | 0 | 0 | 0 |
| A | 10 | 9 | 10 | 0 | 0 | 0 | 10 | 10 | 10 | 0 | 0 | 0 |
| Bsw | 10 | 2 | 9 | 0 | 0 | 8 | 10 | 10 | 10 | 0 | 0 | 0 |
| Bside | 10 | 2 | 2 | 8 | 0 | 3 | 10 | 10 | 10 | 0 | 0 | 0 |
| C | 30 | 22 | 24 | 0 | 2 | 0 | 30 | 30 | 30 | 0 | 0 | 0 |
| D1 | 30 | 11 | 26 | 0 | 2 | 16 | 30 | 30 | 30 | 0 | 0 | 0 |
| D2 | 70 | 38 | 52 | 18 | 0 | 8 | 70 | 70 | 70 | 0 | 0 | 0 |
| W0 | 20 | 13 | 16 | 4 | 0 | 4 | 20 | 20 | 20 | 0 | 0 | 0 |
