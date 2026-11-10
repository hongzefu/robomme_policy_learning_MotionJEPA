# StopCube motion 速度替换读数（wan）

## 判定行

- `SC_POOL=PASS model=wan eps=72 mismatch=0`
- `WINDOW_GEOM=PASS model=wan donors=216 resolved=216 formula_dev_max=0.0`
- `PRESS_LABEL=FAIL model=wan runs=360 mismatch=7`
- `SPEED_SCOPE=PASS model=wan runs=216 unreplaced=0 scope_bad=0`
- `SEED_PRESS model=wan pairs=58 mad=2.0 mean_abs=11.8 press_change=8`
- `SPEED_FOLLOW model=wan beta=0.055 ci=[-0.001,0.132] beta0=0.053 same_delta=1.0 ci=[-0.8,4.0] n=126`
- `SPEED_FOLLOW_SYM model=wan beta=0.024 ci=[-0.004,0.074] beta0=0.024 same_delta=1.0 ci=[-0.8,4.0] n=123`
- `SPEED_FOLLOW_ADJ model=wan beta=0.022 ci=[-0.003,0.069] beta0=0.022 n=130`
- `SPEED_SUCCESS model=wan group=fast new_speed=13/72 base=14/72 diff=-0.014 ci=[-0.062,0.029] true_speed=40/72 neither=18 nopress=1 overlap=0`
- `SPEED_SUCCESS model=wan group=slow new_speed=5/72 base=0/72 diff=0.069 ci=[0.014,0.127] true_speed=37/72 neither=24 nopress=6 overlap=0`
- `SPEED_SUCCESS model=wan group=all new_speed=18/144 base=14/144 diff=0.028 ci=[-0.007,0.069] true_speed=77/144 neither=42 nopress=7 overlap=0`
- `SPEED_NEWWIN_LATE model=wan group=slow w_new_after_MN=66/72`
- `SPEED_LATE model=wan group=fast runs=72 late_after_MN=3 after_cube_stop=0 timeout_off=72/72`
- `SPEED_LATE model=wan group=slow runs=72 late_after_MN=8 after_cube_stop=5 timeout_off=72/72`
- `SPEED_LATE model=wan group=same runs=72 late_after_MN=1 after_cube_stop=0 timeout_off=72/72`
- `SPEED_SAME model=wan true_speed=38/72 base_true=41/72 nopress=6`
- `SC_INVALID model=wan total=0 `

## 各条件按下率与官方成功

| 条件 | 计划 | 有效 | 按下 | 成功 |
|---|---|---|---|---|
| N | 72 | 72 | 64 | 40 |
| S | 72 | 72 | 60 | 32 |
| same | 72 | 72 | 66 | 36 |
| fast | 72 | 72 | 71 | 40 |
| slow | 72 | 72 | 66 | 35 |

## Δ 按 (M→M′) 格（Δ = p_R − p_N，两者都按下）

| 格 | Δ*（按 N） | n | 平均 Δ | 95% 区间 |
|---|---|---|---|---|
| 120->120 | N2:+0, N3:+0, N4:+0, N5:+0 | 24 | 0.1 | [-0.2,0.4] |
| 120->60 | N2:-90, N3:-150, N4:-210, N5:-270 | 24 | -7.8 | [-25.0,1.3] |
| 120->80 | N2:-60, N3:-100, N4:-140, N5:-180 | 24 | 0.2 | [-0.1,0.6] |
| 60->120 | N2:+90, N3:+150, N4:+210, N5:+270 | 19 | 14.9 | [-0.3,44.0] |
| 60->60 | N2:+0, N3:+0, N4:+0, N5:+0 | 18 | -0.9 | [-2.7,0.5] |
| 60->80 | N2:+30, N3:+50, N4:+70, N5:+90 | 17 | 0.7 | [-0.2,1.9] |
| 80->120 | N2:+60, N3:+100, N4:+140, N5:+180 | 21 | 4.0 | [-4.2,14.6] |
| 80->60 | N2:-30, N3:-50, N4:-70, N5:-90 | 21 | 5.9 | [-1.0,16.3] |
| 80->80 | N2:+0, N3:+0, N4:+0, N5:+0 | 21 | 3.5 | [-1.2,12.2] |

## 新速度成功按 (M→M′, N) 分格

| 格 | 新速度成功 | 基线 | 真实速度成功 | 都不 | 未按 | 重叠 |
|---|---|---|---|---|---|---|
| 120->60|N=2 | 0/6 | 0/6 | 2/6 | 4 | 0 | 0 |
| 120->60|N=3 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 120->60|N=4 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 120->60|N=5 | 5/6 | 5/6 | 0/6 | 1 | 0 | 0 |
| 120->80|N=2 | 3/6 | 3/6 | 2/6 | 1 | 0 | 0 |
| 120->80|N=3 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 120->80|N=4 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 120->80|N=5 | 0/6 | 0/6 | 1/6 | 5 | 0 | 0 |
| 60->120|N=2 | 0/6 | 0/6 | 3/6 | 3 | 0 | 0 |
| 60->120|N=3 | 1/6 | 0/6 | 3/6 | 2 | 0 | 0 |
| 60->120|N=4 | 2/6 | 0/6 | 1/6 | 3 | 0 | 0 |
| 60->120|N=5 | 1/6 | 0/6 | 5/6 | 0 | 0 | 0 |
| 60->80|N=2 | 0/6 | 0/6 | 3/6 | 3 | 0 | 0 |
| 60->80|N=3 | 0/6 | 0/6 | 4/6 | 0 | 2 | 0 |
| 60->80|N=4 | 0/6 | 0/6 | 2/6 | 1 | 3 | 0 |
| 60->80|N=5 | 0/6 | 0/6 | 5/6 | 0 | 1 | 0 |
| 80->120|N=2 | 1/6 | 0/6 | 0/6 | 5 | 0 | 0 |
| 80->120|N=3 | 0/6 | 0/6 | 2/6 | 4 | 0 | 0 |
| 80->120|N=4 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 80->120|N=5 | 0/6 | 0/6 | 3/6 | 3 | 0 | 0 |
| 80->60|N=2 | 4/6 | 4/6 | 0/6 | 2 | 0 | 0 |
| 80->60|N=3 | 0/6 | 0/6 | 2/6 | 4 | 0 | 0 |
| 80->60|N=4 | 0/6 | 0/6 | 6/6 | 0 | 0 | 0 |
| 80->60|N=5 | 1/6 | 2/6 | 3/6 | 1 | 1 | 0 |

## 无效计数


