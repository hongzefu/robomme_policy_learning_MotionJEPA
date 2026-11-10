# 第一次拿起交换读数（motion）

## 判定行

- `FIRST_PICK_LABEL=PASS model=motion eps=270 on_turn=268 before_buttons=1 wrong_but_success=0 correct_but_fail=0 outcome_mismatch=0`
- `FP_REPLAY=PASS model=motion eps=270 excluded_f3=0 excluded_schedule=0 mismatches=0`
- `FP_FRAMEMAP=FAIL model=motion video=180 button=90 bad=120`
- `SWAP_SCOPE=PASS model=motion runs=468 scope_bad=0 count_mismatch=0 scope_early_pick=0`
- `MULTI_SWAP_EQ=FAIL runs=0 differ=0`
- `SEED_FIRST_PICK pairs=60 agree=45`
- `FP_SWAP=PASS model=motion task=VideoUnmaskSwap runs=90 invalid=0 excluded=0 scope_early_pick=0 N=28/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP=PASS model=motion task=ButtonUnmaskSwap runs=90 invalid=0 excluded=0 scope_early_pick=0 N=34/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP=PASS model=motion task=VideoRepick runs=90 invalid=0 excluded=0 scope_early_pick=0 N=40/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=1 k=1 runs=18 invalid=0 excluded=0 Bsame_k=correct:7/9,swap_blind_k:0/9,other:2/9,none:0/9 Bsame_k_correct_ci=[0.44,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:8/9,swap_blind_k:0/9,other:1/9,none:0/9 Z_k_correct_ci=[0.67,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=9/15 k_matters=3/9`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=2 k=1 runs=26 invalid=0 excluded=0 Bsame_k=correct:13/13,swap_blind_k:0/13,other:0/13,none:0/13 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:11/13,swap_blind_k:1/13,other:1/13,none:0/13 Z_k_correct_ci=[0.62,1.00] Z_k_blind_ci=[0.00,0.23] N_blind=6/17 k_matters=7/13`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=2 k=2 runs=26 invalid=0 excluded=0 Bsame_k=correct:11/13,swap_blind_k:1/13,other:1/13,none:0/13 Bsame_k_correct_ci=[0.62,1.00] Bsame_k_blind_ci=[0.00,0.23] Z_k=correct:10/13,swap_blind_k:2/13,other:1/13,none:0/13 Z_k_correct_ci=[0.54,1.00] Z_k_blind_ci=[0.00,0.38] N_blind=6/21 k_matters=12/13`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=3 k=1 runs=12 invalid=0 excluded=0 Bsame_k=correct:6/6,swap_blind_k:0/6,other:0/6,none:0/6 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:5/6,swap_blind_k:0/6,other:1/6,none:0/6 Z_k_correct_ci=[0.50,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=7/21 k_matters=2/6`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=3 k=2 runs=12 invalid=0 excluded=0 Bsame_k=correct:6/6,swap_blind_k:0/6,other:0/6,none:0/6 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:4/6,swap_blind_k:0/6,other:2/6,none:0/6 Z_k_correct_ci=[0.33,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=5/16 k_matters=3/6`
- `FP_SWAP_K=PASS model=motion task=VideoUnmaskSwap n=3 k=3 runs=12 invalid=0 excluded=0 Bsame_k=correct:4/6,swap_blind_k:0/6,other:2/6,none:0/6 Bsame_k_correct_ci=[0.33,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:5/6,swap_blind_k:0/6,other:1/6,none:0/6 Z_k_correct_ci=[0.50,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=2/10 k_matters=2/6`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=1 k=1 runs=18 invalid=0 excluded=0 Bsame_k=correct:7/9,swap_blind_k:1/9,other:1/9,none:0/9 Bsame_k_correct_ci=[0.44,1.00] Bsame_k_blind_ci=[0.00,0.33] Z_k=correct:5/9,swap_blind_k:1/9,other:2/9,none:1/9 Z_k_correct_ci=[0.22,0.89] Z_k_blind_ci=[0.00,0.33] N_blind=6/15 k_matters=4/9`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=2 k=1 runs=20 invalid=0 excluded=0 Bsame_k=correct:9/10,swap_blind_k:1/10,other:0/10,none:0/10 Bsame_k_correct_ci=[0.70,1.00] Bsame_k_blind_ci=[0.00,0.30] Z_k=correct:8/10,swap_blind_k:1/10,other:1/10,none:0/10 Z_k_correct_ci=[0.50,1.00] Z_k_blind_ci=[0.00,0.30] N_blind=8/21 k_matters=7/10`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=2 k=2 runs=20 invalid=0 excluded=0 Bsame_k=correct:8/10,swap_blind_k:0/10,other:2/10,none:0/10 Bsame_k_correct_ci=[0.50,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:7/10,swap_blind_k:2/10,other:1/10,none:0/10 Z_k_correct_ci=[0.40,1.00] Z_k_blind_ci=[0.00,0.50] N_blind=6/18 k_matters=7/10`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=3 k=1 runs=30 invalid=0 excluded=0 Bsame_k=correct:9/15,swap_blind_k:2/15,other:4/15,none:0/15 Bsame_k_correct_ci=[0.33,0.80] Bsame_k_blind_ci=[0.00,0.33] Z_k=correct:7/15,swap_blind_k:4/15,other:3/15,none:1/15 Z_k_correct_ci=[0.20,0.73] Z_k_blind_ci=[0.07,0.47] N_blind=3/15 k_matters=11/15`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=3 k=2 runs=30 invalid=0 excluded=0 Bsame_k=correct:8/15,swap_blind_k:2/15,other:5/15,none:0/15 Bsame_k_correct_ci=[0.27,0.80] Bsame_k_blind_ci=[0.00,0.33] Z_k=correct:9/15,swap_blind_k:2/15,other:4/15,none:0/15 Z_k_correct_ci=[0.33,0.87] Z_k_blind_ci=[0.00,0.33] N_blind=5/18 k_matters=9/15`
- `FP_SWAP_K=PASS model=motion task=ButtonUnmaskSwap n=3 k=3 runs=30 invalid=0 excluded=0 Bsame_k=correct:12/15,swap_blind_k:0/15,other:3/15,none:0/15 Bsame_k_correct_ci=[0.60,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:10/15,swap_blind_k:0/15,other:5/15,none:0/15 Z_k_correct_ci=[0.40,0.87] Z_k_blind_ci=[0.00,0.00] N_blind=4/14 k_matters=5/15`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=1 k=1 runs=30 invalid=0 excluded=0 Bsame_k=correct:11/15,swap_blind_k:2/15,other:2/15,none:0/15 Bsame_k_correct_ci=[0.53,0.93] Bsame_k_blind_ci=[0.00,0.33] Z_k=correct:8/15,swap_blind_k:3/15,other:4/15,none:0/15 Z_k_correct_ci=[0.27,0.80] Z_k_blind_ci=[0.00,0.40] N_blind=10/30 k_matters=15/15`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=2 k=1 runs=26 invalid=0 excluded=0 Bsame_k=correct:12/13,swap_blind_k:0/13,other:0/13,none:1/13 Bsame_k_correct_ci=[0.77,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:9/13,swap_blind_k:1/13,other:3/13,none:0/13 Z_k_correct_ci=[0.46,0.92] Z_k_blind_ci=[0.00,0.23] N_blind=5/19 k_matters=11/13`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=2 k=2 runs=26 invalid=0 excluded=0 Bsame_k=correct:11/13,swap_blind_k:1/13,other:1/13,none:0/13 Bsame_k_correct_ci=[0.62,1.00] Bsame_k_blind_ci=[0.00,0.23] Z_k=correct:10/13,swap_blind_k:1/13,other:1/13,none:1/13 Z_k_correct_ci=[0.54,1.00] Z_k_blind_ci=[0.00,0.23] N_blind=6/26 k_matters=10/13`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=3 k=1 runs=24 invalid=0 excluded=0 Bsame_k=correct:12/12,swap_blind_k:0/12,other:0/12,none:0/12 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:12/12,swap_blind_k:0/12,other:0/12,none:0/12 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=4/17 k_matters=7/12`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=3 k=2 runs=24 invalid=0 excluded=0 Bsame_k=correct:12/12,swap_blind_k:0/12,other:0/12,none:0/12 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:12/12,swap_blind_k:0/12,other:0/12,none:0/12 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=6/22 k_matters=7/12`
- `FP_SWAP_K=PASS model=motion task=VideoRepick n=3 k=3 runs=24 invalid=0 excluded=0 Bsame_k=correct:11/12,swap_blind_k:0/12,other:1/12,none:0/12 Bsame_k_correct_ci=[0.75,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:11/12,swap_blind_k:0/12,other:0/12,none:1/12 Z_k_correct_ci=[0.75,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=0/5 k_matters=4/12`

## VideoUnmaskSwap

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 28 | 16 | 46 | 0 | 16/45 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

## ButtonUnmaskSwap

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 34 | 12 | 44 | 0 | 12/45 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

## VideoRepick

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 40 | 22 | 27 | 1 | 22/61 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

