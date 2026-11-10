# StopCube motion 速度替换读数（motion）

## 判定行

- `SC_POOL=PASS model=motion eps=72 mismatch=0`
- `WINDOW_GEOM=PASS model=motion donors=216 resolved=216 formula_dev_max=0.0`
- `PRESS_LABEL=FAIL model=motion runs=360 mismatch=13`
- `SPEED_SCOPE=PASS model=motion runs=216 unreplaced=0 scope_bad=0`
- `SEED_PRESS model=motion pairs=65 mad=3.0 mean_abs=15.2 press_change=3`
- `SPEED_FOLLOW model=motion beta=0.913 ci=[0.837,0.967] beta0=0.914 same_delta=-16.9 ci=[-34.1,-4.6] n=133`
- `SPEED_FOLLOW_SYM model=motion beta=0.684 ci=[0.517,0.858] beta0=0.856 same_delta=-16.9 ci=[-34.1,-4.6] n=86`
- `SPEED_FOLLOW_ADJ model=motion beta=0.806 ci=[0.640,0.924] beta0=0.806 n=143`
- `SPEED_SUCCESS model=motion group=fast new_speed=32/72 base=2/72 diff=0.417 ci=[0.313,0.514] true_speed=0/72 neither=40 nopress=0 overlap=0`
- `SPEED_SUCCESS model=motion group=slow new_speed=44/72 base=0/72 diff=0.611 ci=[0.493,0.726] true_speed=3/72 neither=24 nopress=1 overlap=0`
- `SPEED_SUCCESS model=motion group=all new_speed=76/144 base=2/144 diff=0.514 ci=[0.438,0.590] true_speed=3/144 neither=64 nopress=1 overlap=0`
- `SPEED_NEWWIN_LATE model=motion group=slow w_new_after_MN=66/72`
- `SPEED_LATE model=motion group=fast runs=72 late_after_MN=0 after_cube_stop=0 timeout_off=72/72`
- `SPEED_LATE model=motion group=slow runs=72 late_after_MN=54 after_cube_stop=30 timeout_off=72/72`
- `SPEED_LATE model=motion group=same runs=72 late_after_MN=2 after_cube_stop=0 timeout_off=72/72`
- `SPEED_SAME model=motion true_speed=40/72 base_true=43/72 nopress=0`
- `SC_INVALID model=motion total=0 `

## 各条件按下率与官方成功

| 条件 | 计划 | 有效 | 按下 | 成功 |
|---|---|---|---|---|
| N | 72 | 72 | 67 | 38 |
| S | 72 | 72 | 66 | 36 |
| same | 72 | 72 | 72 | 36 |
| fast | 72 | 72 | 72 | 0 |
| slow | 72 | 72 | 71 | 3 |

## Δ 按 (M→M′) 格（Δ = p_R − p_N，两者都按下）

| 格 | Δ*（按 N） | n | 平均 Δ | 95% 区间 |
|---|---|---|---|---|
| 120->120 | N2:+0, N3:+0, N4:+0, N5:+0 | 24 | -39.0 | [-81.5,-6.6] |
| 120->60 | N2:-90, N3:-150, N4:-210, N5:-270 | 24 | -161.8 | [-196.1,-126.0] |
| 120->80 | N2:-60, N3:-100, N4:-140, N5:-180 | 24 | -107.8 | [-131.1,-82.5] |
| 60->120 | N2:+90, N3:+150, N4:+210, N5:+270 | 20 | 148.8 | [93.9,195.3] |
| 60->60 | N2:+0, N3:+0, N4:+0, N5:+0 | 20 | -10.6 | [-19.6,-3.2] |
| 60->80 | N2:+30, N3:+50, N4:+70, N5:+90 | 19 | 49.2 | [24.7,70.1] |
| 80->120 | N2:+60, N3:+100, N4:+140, N5:+180 | 23 | 106.8 | [78.9,134.4] |
| 80->60 | N2:-30, N3:-50, N4:-70, N5:-90 | 23 | -56.7 | [-71.0,-41.3] |
| 80->80 | N2:+0, N3:+0, N4:+0, N5:+0 | 23 | 0.8 | [-1.1,4.0] |

## 新速度成功按 (M→M′, N) 分格

| 格 | 新速度成功 | 基线 | 真实速度成功 | 都不 | 未按 | 重叠 |
|---|---|---|---|---|---|---|
| 120->60|N=2 | 3/6 | 0/6 | 0/6 | 3 | 0 | 0 |
| 120->60|N=3 | 1/6 | 0/6 | 0/6 | 5 | 0 | 0 |
| 120->60|N=4 | 2/6 | 0/6 | 0/6 | 4 | 0 | 0 |
| 120->60|N=5 | 4/6 | 0/6 | 0/6 | 2 | 0 | 0 |
| 120->80|N=2 | 2/6 | 1/6 | 0/6 | 4 | 0 | 0 |
| 120->80|N=3 | 2/6 | 0/6 | 0/6 | 4 | 0 | 0 |
| 120->80|N=4 | 5/6 | 0/6 | 0/6 | 1 | 0 | 0 |
| 120->80|N=5 | 3/6 | 0/6 | 0/6 | 3 | 0 | 0 |
| 60->120|N=2 | 2/6 | 0/6 | 2/6 | 2 | 0 | 0 |
| 60->120|N=3 | 4/6 | 0/6 | 0/6 | 2 | 0 | 0 |
| 60->120|N=4 | 6/6 | 0/6 | 0/6 | 0 | 0 | 0 |
| 60->120|N=5 | 5/6 | 0/6 | 0/6 | 1 | 0 | 0 |
| 60->80|N=2 | 2/6 | 0/6 | 1/6 | 3 | 0 | 0 |
| 60->80|N=3 | 1/6 | 0/6 | 0/6 | 5 | 0 | 0 |
| 60->80|N=4 | 5/6 | 0/6 | 0/6 | 1 | 0 | 0 |
| 60->80|N=5 | 2/6 | 0/6 | 0/6 | 3 | 1 | 0 |
| 80->120|N=2 | 2/6 | 0/6 | 0/6 | 4 | 0 | 0 |
| 80->120|N=3 | 3/6 | 0/6 | 0/6 | 3 | 0 | 0 |
| 80->120|N=4 | 6/6 | 0/6 | 0/6 | 0 | 0 | 0 |
| 80->120|N=5 | 6/6 | 0/6 | 0/6 | 0 | 0 | 0 |
| 80->60|N=2 | 2/6 | 1/6 | 0/6 | 4 | 0 | 0 |
| 80->60|N=3 | 2/6 | 0/6 | 0/6 | 4 | 0 | 0 |
| 80->60|N=4 | 0/6 | 0/6 | 0/6 | 6 | 0 | 0 |
| 80->60|N=5 | 6/6 | 0/6 | 0/6 | 0 | 0 | 0 |

## 无效计数


