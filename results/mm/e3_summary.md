# E3 summary (45 runs, 3 seeds, 40 epochs, Adam 1e-3)

| train set | shots | adapter | params | zero-shot test | test acc (mean +- std) | best epoch acc | transfer acc retained (other 2 sets) | COCO R@1 | end alphas (mean over seeds) | memory firing |
|---|---|---|---|---|---|---|---|---|---|---|
| imagenette_train | 16 | l3_dense | 705k | 0.985 | 0.956 +- 0.010 | 0.985 | 0.236 | 0.095 | l3_wk 13.64, l3_wv 11.58, l3_wup 4.07, l3_wmix 1.69 | - |
| imagenette_train | 16 | l3_spike_mem | 705k | 0.985 | 0.977 +- 0.001 | 0.985 | 0.163 | 0.020 | l3_wk 5.83, l3_wv 6.60, l3_wup 4.15, l3_wmix 1.68 | 0.28 |
| imagenette_train | 16 | l3_spike_rate | 705k | 0.985 | 0.977 +- 0.002 | 0.985 | 0.155 | 0.017 | l3_wk 4.82, l3_wv 6.77, l3_wup 4.18, l3_wmix 1.69 | - |
| imagenette_train | 16 | mlp_dense | 263k | 0.985 | 0.982 +- 0.002 | 0.985 | 0.863 | 0.173 | adapter_down 4.46, adapter_up 1.50 | - |
| imagenette_train | 16 | mlp_spike | 263k | 0.985 | 0.984 +- 0.001 | 0.985 | 0.861 | 0.183 | adapter_down 4.11, adapter_up 1.51 | - |
| pets_train | 16 | l3_dense | 705k | 0.834 | 0.729 +- 0.002 | 0.834 | 0.245 | 0.058 | l3_wk 8.03, l3_wv 10.38, l3_wup 5.14, l3_wmix 1.60 | - |
| pets_train | 16 | l3_spike_mem | 705k | 0.834 | 0.744 +- 0.002 | 0.834 | 0.169 | 0.070 | l3_wk 5.18, l3_wv 3.66, l3_wup 2.41, l3_wmix 1.56 | 0.17 |
| pets_train | 16 | l3_spike_rate | 705k | 0.834 | 0.763 +- 0.010 | 0.834 | 0.165 | 0.066 | l3_wk 3.11, l3_wv 4.36, l3_wup 2.49, l3_wmix 1.57 | - |
| pets_train | 16 | mlp_dense | 263k | 0.834 | 0.854 +- 0.011 | 0.863 | 0.927 | 0.193 | adapter_down 2.73, adapter_up 1.42 | - |
| pets_train | 16 | mlp_spike | 263k | 0.834 | 0.842 +- 0.020 | 0.868 | 0.926 | 0.167 | adapter_down 2.85, adapter_up 1.42 | - |
| pets_train | all | l3_dense | 705k | 0.834 | 0.809 +- 0.014 | 0.834 | 0.179 | 0.049 | l3_wk 2.72, l3_wv 9.04, l3_wup 5.34, l3_wmix 1.51 | - |
| pets_train | all | l3_spike_mem | 705k | 0.834 | 0.849 +- 0.004 | 0.853 | 0.447 | 0.149 | l3_wk 3.99, l3_wv 3.93, l3_wup 2.75, l3_wmix 1.54 | 0.25 |
| pets_train | all | l3_spike_rate | 705k | 0.834 | 0.848 +- 0.004 | 0.850 | 0.404 | 0.128 | l3_wk 2.68, l3_wv 4.34, l3_wup 2.74, l3_wmix 1.55 | - |
| pets_train | all | mlp_dense | 263k | 0.834 | 0.867 +- 0.005 | 0.870 | 0.875 | 0.204 | adapter_down 1.85, adapter_up 1.35 | - |
| pets_train | all | mlp_spike | 263k | 0.834 | 0.852 +- 0.005 | 0.884 | 0.886 | 0.087 | adapter_down 2.09, adapter_up 1.41 | - |

## Alpha vs accuracy across runs

- MLP adapters, alpha of first matrix (adapter_down / l3_wup) vs test: Spearman +0.54 (p=0.021, n=18)
- MLP adapters, alpha of first matrix (adapter_down / l3_wup) vs gain: Spearman -0.73 (p=0.001, n=18)
- MLP adapters, alpha of first matrix (adapter_down / l3_wup) vs transfer: Spearman -0.27 (p=0.285, n=18)
- MLP adapters, alpha of output matrix (adapter_up / l3_wmix) vs test: Spearman +0.66 (p=0.003, n=18)
- MLP adapters, alpha of output matrix (adapter_up / l3_wmix) vs gain: Spearman -0.61 (p=0.007, n=18)
- MLP adapters, alpha of output matrix (adapter_up / l3_wmix) vs transfer: Spearman -0.30 (p=0.231, n=18)
- L3 adapters, alpha of first matrix (adapter_down / l3_wup) vs test: Spearman +0.25 (p=0.217, n=27)
- L3 adapters, alpha of first matrix (adapter_down / l3_wup) vs gain: Spearman -0.02 (p=0.905, n=27)
- L3 adapters, alpha of first matrix (adapter_down / l3_wup) vs transfer: Spearman -0.03 (p=0.889, n=27)
- L3 adapters, alpha of output matrix (adapter_up / l3_wmix) vs test: Spearman +0.47 (p=0.013, n=27)
- L3 adapters, alpha of output matrix (adapter_up / l3_wmix) vs gain: Spearman -0.20 (p=0.328, n=27)
- L3 adapters, alpha of output matrix (adapter_up / l3_wmix) vs transfer: Spearman -0.28 (p=0.161, n=27)
- all, alpha of first matrix (adapter_down / l3_wup) vs test: Spearman +0.24 (p=0.106, n=45)
- all, alpha of first matrix (adapter_down / l3_wup) vs gain: Spearman -0.42 (p=0.005, n=45)
- all, alpha of first matrix (adapter_down / l3_wup) vs transfer: Spearman -0.29 (p=0.050, n=45)
- all, alpha of output matrix (adapter_up / l3_wmix) vs test: Spearman -0.01 (p=0.971, n=45)
- all, alpha of output matrix (adapter_up / l3_wmix) vs gain: Spearman -0.67 (p=0.000, n=45)
- all, alpha of output matrix (adapter_up / l3_wmix) vs transfer: Spearman -0.78 (p=0.000, n=45)
