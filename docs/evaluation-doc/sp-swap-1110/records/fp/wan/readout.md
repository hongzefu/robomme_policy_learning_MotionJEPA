# 第一次拿起交换读数（wan）

## 判定行

- `FIRST_PICK_LABEL=PASS model=wan eps=270 on_turn=269 before_buttons=0 wrong_but_success=0 correct_but_fail=0 outcome_mismatch=0`
- `FP_REPLAY=PASS model=wan eps=270 excluded_f3=0 excluded_schedule=0 mismatches=0`
- `FP_FRAMEMAP=FAIL model=wan video=180 button=90 bad=120`
- `SWAP_SCOPE=PASS model=wan runs=412 scope_bad=0 count_mismatch=0 scope_early_pick=0`
- `MULTI_SWAP_EQ=FAIL runs=0 differ=0`
- `SEED_FIRST_PICK pairs=60 agree=46`
- `FP_SWAP=PASS model=wan task=VideoUnmaskSwap runs=90 invalid=0 excluded=0 scope_early_pick=0 N=27/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP=PASS model=wan task=ButtonUnmaskSwap runs=90 invalid=0 excluded=0 scope_early_pick=0 N=26/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP=PASS model=wan task=VideoRepick runs=90 invalid=0 excluded=0 scope_early_pick=0 N=33/90 Z=0/0 Rnd=0/0 S0=0/0 A=0/0 B=0/0 Zall=0/0 Z_blind=0/0 Rnd_blind=0/0 S0_blind=0/0 A_blind=0/0 B_blind=0/0`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=1 k=1 runs=16 invalid=0 excluded=0 Bsame_k=correct:7/8,swap_blind_k:0/8,other:1/8,none:0/8 Bsame_k_correct_ci=[0.62,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:2/8,swap_blind_k:0/8,other:0/8,none:6/8 Z_k_correct_ci=[0.00,0.62] Z_k_blind_ci=[0.00,0.00] N_blind=8/15 k_matters=4/8`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=2 k=1 runs=20 invalid=0 excluded=0 Bsame_k=correct:10/10,swap_blind_k:0/10,other:0/10,none:0/10 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:1/10,swap_blind_k:0/10,other:4/10,none:5/10 Z_k_correct_ci=[0.00,0.30] Z_k_blind_ci=[0.00,0.00] N_blind=6/17 k_matters=6/10`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=2 k=2 runs=20 invalid=0 excluded=0 Bsame_k=correct:10/10,swap_blind_k:0/10,other:0/10,none:0/10 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:9/10,swap_blind_k:0/10,other:1/10,none:0/10 Z_k_correct_ci=[0.70,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=9/21 k_matters=7/10`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=3 k=1 runs=18 invalid=0 excluded=0 Bsame_k=correct:8/9,swap_blind_k:0/9,other:1/9,none:0/9 Bsame_k_correct_ci=[0.67,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:0/9,swap_blind_k:0/9,other:0/9,none:9/9 Z_k_correct_ci=[0.00,0.00] Z_k_blind_ci=[0.00,0.00] N_blind=0/21 k_matters=7/9`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=3 k=2 runs=18 invalid=0 excluded=0 Bsame_k=correct:9/9,swap_blind_k:0/9,other:0/9,none:0/9 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:9/9,swap_blind_k:0/9,other:0/9,none:0/9 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=1/16 k_matters=6/9`
- `FP_SWAP_K=PASS model=wan task=VideoUnmaskSwap n=3 k=3 runs=18 invalid=0 excluded=0 Bsame_k=correct:9/9,swap_blind_k:0/9,other:0/9,none:0/9 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:9/9,swap_blind_k:0/9,other:0/9,none:0/9 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=0/10 k_matters=1/9`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=1 k=1 runs=14 invalid=0 excluded=0 Bsame_k=correct:6/7,swap_blind_k:1/7,other:0/7,none:0/7 Bsame_k_correct_ci=[0.57,1.00] Bsame_k_blind_ci=[0.00,0.43] Z_k=correct:0/7,swap_blind_k:0/7,other:0/7,none:7/7 Z_k_correct_ci=[0.00,0.00] Z_k_blind_ci=[0.00,0.00] N_blind=5/15 k_matters=5/7`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=2 k=1 runs=22 invalid=0 excluded=0 Bsame_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:0/11,swap_blind_k:0/11,other:0/11,none:11/11 Z_k_correct_ci=[0.00,0.00] Z_k_blind_ci=[0.00,0.00] N_blind=8/21 k_matters=7/11`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=2 k=2 runs=22 invalid=0 excluded=0 Bsame_k=correct:10/11,swap_blind_k:1/11,other:0/11,none:0/11 Bsame_k_correct_ci=[0.73,1.00] Bsame_k_blind_ci=[0.00,0.27] Z_k=correct:9/11,swap_blind_k:1/11,other:1/11,none:0/11 Z_k_correct_ci=[0.55,1.00] Z_k_blind_ci=[0.00,0.27] N_blind=4/18 k_matters=8/11`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=3 k=1 runs=16 invalid=0 excluded=0 Bsame_k=correct:6/8,swap_blind_k:0/8,other:2/8,none:0/8 Bsame_k_correct_ci=[0.38,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:0/8,swap_blind_k:0/8,other:0/8,none:8/8 Z_k_correct_ci=[0.00,0.00] Z_k_blind_ci=[0.00,0.00] N_blind=8/15 k_matters=4/8`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=3 k=2 runs=16 invalid=0 excluded=0 Bsame_k=correct:6/8,swap_blind_k:0/8,other:1/8,none:1/8 Bsame_k_correct_ci=[0.38,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:6/8,swap_blind_k:1/8,other:1/8,none:0/8 Z_k_correct_ci=[0.50,1.00] Z_k_blind_ci=[0.00,0.38] N_blind=10/18 k_matters=4/8`
- `FP_SWAP_K=PASS model=wan task=ButtonUnmaskSwap n=3 k=3 runs=16 invalid=0 excluded=0 Bsame_k=correct:8/8,swap_blind_k:0/8,other:0/8,none:0/8 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:7/8,swap_blind_k:0/8,other:1/8,none:0/8 Z_k_correct_ci=[0.62,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=3/14 k_matters=5/8`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=1 k=1 runs=18 invalid=0 excluded=0 Bsame_k=correct:9/9,swap_blind_k:0/9,other:0/9,none:0/9 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:8/9,swap_blind_k:1/9,other:0/9,none:0/9 Z_k_correct_ci=[0.67,1.00] Z_k_blind_ci=[0.00,0.33] N_blind=12/30 k_matters=9/9`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=2 k=1 runs=26 invalid=0 excluded=0 Bsame_k=correct:13/13,swap_blind_k:0/13,other:0/13,none:0/13 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:12/13,swap_blind_k:0/13,other:0/13,none:1/13 Z_k_correct_ci=[0.77,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=6/19 k_matters=9/13`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=2 k=2 runs=26 invalid=0 excluded=0 Bsame_k=correct:13/13,swap_blind_k:0/13,other:0/13,none:0/13 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:12/13,swap_blind_k:0/13,other:0/13,none:1/13 Z_k_correct_ci=[0.77,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=10/26 k_matters=10/13`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=3 k=1 runs=22 invalid=0 excluded=0 Bsame_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=6/17 k_matters=8/11`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=3 k=2 runs=22 invalid=0 excluded=0 Bsame_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Z_k_correct_ci=[1.00,1.00] Z_k_blind_ci=[0.00,0.00] N_blind=9/22 k_matters=9/11`
- `FP_SWAP_K=PASS model=wan task=VideoRepick n=3 k=3 runs=22 invalid=0 excluded=0 Bsame_k=correct:11/11,swap_blind_k:0/11,other:0/11,none:0/11 Bsame_k_correct_ci=[1.00,1.00] Bsame_k_blind_ci=[0.00,0.00] Z_k=correct:7/11,swap_blind_k:0/11,other:1/11,none:3/11 Z_k_correct_ci=[0.36,0.91] Z_k_blind_ci=[0.00,0.00] N_blind=2/5 k_matters=1/11`

## VideoUnmaskSwap

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 27 | 13 | 50 | 0 | 13/45 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

## ButtonUnmaskSwap

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 26 | 14 | 50 | 0 | 14/45 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

## VideoRepick

| 组 | 有效局 | correct | swap_blind | other | none | moved 局 swap_blind |
|---|---|---|---|---|---|---|
| N | 90 | 33 | 20 | 36 | 1 | 20/61 |
| Z | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Rnd | 0 | 0 | 0 | 0 | 0 | 0/0 |
| S0 | 0 | 0 | 0 | 0 | 0 | 0/0 |
| A | 0 | 0 | 0 | 0 | 0 | 0/0 |
| B | 0 | 0 | 0 | 0 | 0 | 0/0 |
| Zall | 0 | 0 | 0 | 0 | 0 | 0/0 |

