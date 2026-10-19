# Per-model analysis (208 runs)

## L3Classifier (n = 35)

| alpha summary | rho vs train acc (p) | rho vs val acc (p) | rho vs gap (p) | n |
|---|---|---|---|---|
| all analysed layers | -0.28 (0.103) | -0.32 (0.066) | -0.03 (0.887) | 34 |
| dense layers (mlp, w_up, w_mix) | -0.25 (0.153) | -0.12 (0.483) | -0.14 (0.421) | 34 |
| lookup tables (embed, w_k, w_v) | -0.22 (0.251) | -0.39 (0.032) | 0.18 (0.353) | 30 |

Observation 1 (alpha < 2 => degraded / over-fit), using a_dense (best val acc of this model: 0.848):
- alpha < 2: n=9, val 0.766 (min 0.652, max 0.842), train 0.839, gap +0.072 (max +0.267)
- alpha >= 2: n=25, val 0.749 (min 0.519, max 0.848), train 0.837, gap +0.089 (max +0.276)
Observation 2 (alpha in [2,3] can still be good): 1/14 runs in [2,3] are within 0.03 of the best val acc; 3/9 runs with alpha < 2 are.
Observation 3 (alpha falls without val improvement): among 96 epoch-to-epoch alpha drops (> 0.02) in 26 runs with per-epoch WW, 21 (22 %) came with no val-acc gain.

## MLP3 (n = 122)

| alpha summary | rho vs train acc (p) | rho vs val acc (p) | rho vs gap (p) | n |
|---|---|---|---|---|
| all analysed layers | -0.27 (0.004) | -0.34 (0.000) | -0.03 (0.779) | 113 |
| dense layers (mlp, w_up, w_mix) | -0.27 (0.004) | -0.34 (0.000) | -0.03 (0.779) | 113 |
| fc1 | -0.16 (0.096) | -0.24 (0.016) | 0.00 (0.969) | 105 |
| fc2 | -0.24 (0.011) | -0.35 (0.000) | 0.04 (0.677) | 109 |

Observation 1 (alpha < 2 => degraded / over-fit), using a_dense (best val acc of this model: 0.892):
- alpha < 2: n=44, val 0.778 (min 0.099, max 0.877), train 0.703, gap -0.074 (max +0.080)
- alpha >= 2: n=69, val 0.700 (min 0.104, max 0.892), train 0.626, gap -0.074 (max +0.105)
Observation 2 (alpha in [2,3] can still be good): 2/24 runs in [2,3] are within 0.03 of the best val acc; 2/44 runs with alpha < 2 are.
Observation 3 (alpha falls without val improvement): among 132 epoch-to-epoch alpha drops (> 0.02) in 41 runs with per-epoch WW, 23 (17 %) came with no val-acc gain.

## SNNL3Classifier (n = 33)

| alpha summary | rho vs train acc (p) | rho vs val acc (p) | rho vs gap (p) | n |
|---|---|---|---|---|
| all analysed layers | -0.68 (0.000) | -0.67 (0.000) | -0.39 (0.027) | 33 |
| dense layers (mlp, w_up, w_mix) | -0.64 (0.000) | -0.64 (0.000) | -0.39 (0.032) | 31 |
| lookup tables (embed, w_k, w_v) | -0.66 (0.000) | -0.64 (0.000) | -0.43 (0.017) | 31 |

Observation 1 (alpha < 2 => degraded / over-fit), using a_dense (best val acc of this model: 0.795):
- alpha < 2: n=10, val 0.628 (min 0.327, max 0.795), train 0.674, gap +0.046 (max +0.100)
- alpha >= 2: n=21, val 0.466 (min 0.195, max 0.733), train 0.489, gap +0.023 (max +0.071)
Observation 2 (alpha in [2,3] can still be good): 0/12 runs in [2,3] are within 0.03 of the best val acc; 1/10 runs with alpha < 2 are.
Observation 3 (alpha falls without val improvement): among 23 epoch-to-epoch alpha drops (> 0.02) in 14 runs with per-epoch WW, 4 (17 %) came with no val-acc gain.

| factor | value | n | mean alpha_dense | mean alpha fc1 | mean val acc | mean dead-row frac |
|---|---|---|---|---|---|---|
| k_sg | 1.0 | 8 | 3.13 | nan | 0.329 | 0.000 |
| k_sg | 2.0 | 6 | 2.51 | nan | 0.525 | 0.000 |
| k_sg | 4.0 | 6 | 2.42 | nan | 0.557 | 0.000 |
| k_sg | 8.0 | 13 | 2.41 | nan | 0.567 | 0.006 |
| T | 4 | 13 | 2.49 | nan | 0.527 | 0.006 |
| T | 8 | 20 | 2.62 | nan | 0.482 | 0.000 |
| beta | 0.5 | 20 | 2.60 | nan | 0.472 | 0.000 |
| beta | 0.9 | 13 | 2.53 | nan | 0.543 | 0.006 |
| mode | rate_query | 17 | 2.62 | nan | 0.500 | 0.000 |
| mode | spiking_memory | 16 | 2.51 | nan | 0.500 | 0.005 |
| h | 128 | 20 | 2.50 | nan | 0.470 | 0.000 |
| h | 256 | 13 | 2.69 | nan | 0.547 | 0.006 |

Spearman(dead-row fraction, alpha_dense) = -0.36 (p 0.048, n 31)

## SNNMLP3 (n = 18)

| alpha summary | rho vs train acc (p) | rho vs val acc (p) | rho vs gap (p) | n |
|---|---|---|---|---|
| all analysed layers | 0.15 (0.553) | 0.11 (0.675) | 0.24 (0.336) | 18 |
| dense layers (mlp, w_up, w_mix) | 0.15 (0.553) | 0.11 (0.675) | 0.24 (0.336) | 18 |
| fc1 | 0.20 (0.433) | 0.14 (0.570) | 0.25 (0.320) | 18 |
| fc2 | 0.10 (0.681) | 0.07 (0.791) | 0.24 (0.336) | 18 |

Observation 1 (alpha < 2 => degraded / over-fit), using a_dense (best val acc of this model: 0.871):
- alpha < 2: n=11, val 0.767 (min 0.640, max 0.871), train 0.777, gap +0.010 (max +0.024)
- alpha >= 2: n=7, val 0.804 (min 0.741, max 0.865), train 0.817, gap +0.013 (max +0.023)
Observation 2 (alpha in [2,3] can still be good): 2/7 runs in [2,3] are within 0.03 of the best val acc; 3/11 runs with alpha < 2 are.
Observation 3 (alpha falls without val improvement): among 40 epoch-to-epoch alpha drops (> 0.02) in 14 runs with per-epoch WW, 5 (12 %) came with no val-acc gain.

| factor | value | n | mean alpha_dense | mean alpha fc1 | mean val acc | mean dead-row frac |
|---|---|---|---|---|---|---|
| k_sg | 1.0 | 2 | 1.58 | 1.60 | 0.710 | 0.122 |
| k_sg | 2.0 | 6 | 1.94 | 1.89 | 0.787 | 0.018 |
| k_sg | 4.0 | 6 | 1.87 | 1.85 | 0.799 | 0.019 |
| k_sg | 8.0 | 4 | 2.03 | 1.99 | 0.783 | 0.068 |
| T | 4 | 7 | 2.01 | 1.96 | 0.776 | 0.035 |
| T | 8 | 6 | 1.80 | 1.79 | 0.794 | 0.025 |
| T | 16 | 5 | 1.86 | 1.82 | 0.774 | 0.068 |
| beta | 0.5 | 10 | 1.80 | 1.77 | 0.805 | 0.049 |
| beta | 0.9 | 8 | 2.02 | 1.98 | 0.753 | 0.030 |
| h | 64 | 2 | 1.94 | 1.85 | 0.648 | 0.000 |
| h | 128 | 5 | 1.92 | 1.90 | 0.806 | 0.009 |
| h | 256 | 7 | 1.95 | 1.92 | 0.779 | 0.045 |
| h | 512 | 4 | 1.75 | 1.72 | 0.823 | 0.094 |

Spearman(dead-row fraction, alpha_dense) = -0.79 (p 0.000, n 18)

