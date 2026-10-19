# Sweep analysis

Inputs: results/mlp_sweep, results/l3_sweep, results/snn_sweep, results/snn_sweep2  
Runs: 208

## Runs per model

| model | runs |
|---|---|
| L3Classifier | 35 |
| MLP3 | 122 |
| SNNL3Classifier | 33 |
| SNNMLP3 | 18 |


## Config coverage

| config | values (count) |
|---|---|
| h | 64:21, 128:66, 256:73, 512:25, 1024:23 |
| lr | min=0.0001021 median=0.0015 max=0.02688 |
| weight_decay | 0:42, 1e-05:37, 0.0001:46, 0.001:30, 0.01:53 |
| batch_size | 32:37, 64:57, 128:58, 256:56 |
| optimizer | adam:145, sgd:63 |
| momentum | 0.9:208 |
| epochs | min=1 median=6 max=20 |
| init_gain | 0.5:42, 1:37, 2:43 |
| dropout | 0:49, 0.2:36, 0.5:37 |
| label_noise | 0:37, 0.1:46, 0.3:39 |
| train_subset | 2000:61, 5000:15, 10000:72, 60000:60 |
| V | 256:10, 512:25, 1024:33 |
| n_emb | 2048:37, 4096:23, 8192:8 |
| k | 4:21, 8:13, 16:34 |
| d_emb | 32:23, 64:23, 128:22 |
| d_up | 128:41, 256:27 |
| n_layers | 1:19, 2:20, 3:29 |
| n_patches | 16:68 |
| model_type | snn_l3:33, snn_mlp3:18 |
| T | 4:20, 8:26, 16:5 |
| k_sg | 1:10, 2:12, 4:12, 8:17 |
| beta | 0.5:30, 0.9:21 |
| theta | 1:51 |
| mode | rate_query:17, spiking_memory:16 |


- **train_acc**: mean=0.6656 min=0.0955 max=0.9995 (n=208)
- **val_acc**: mean=0.6868 min=0.0952 max=0.8919 (n=208)
- **test_acc**: mean=0.6778 min=0.0988 max=0.8805 (n=208)
- **gap**: mean=-0.0212 min=-0.2468 max=0.2764 (n=208)
- **alpha_avg**: mean=4.5833 min=0.3997 max=23.7238 (n=208)
- **alpha_default_avg**: mean=4.1409 min=0.7334 max=17.3524 (n=208)
- **alpha_min50_avg**: mean=4.5833 min=0.3997 max=23.7238 (n=208)

## Spearman correlations (each WW summary vs accuracy / gap)

rho_<target> = Spearman rho of the metric with <target>; p_<target> its p-value; n = runs with the metric.
Martin's post reports *negative* correlation of average alpha with train accuracy (lower alpha = better fit)

| metric | n | rho_train_acc | p_train_acc | rho_val_acc | p_val_acc | rho_test_acc | p_test_acc | rho_gap | p_gap |
|---|---|---|---|---|---|---|---|---|---|
| alpha_avg | 208 | -0.292 | 1.8e-05 | -0.472 | 6.4e-13 | -0.471 | 7.2e-13 | 0.122 | 0.08 |
| alpha_default_avg | 208 | -0.321 | 2.3e-06 | -0.453 | 6.4e-12 | -0.451 | 8.4e-12 | 0.096 | 0.17 |
| alpha_min50_avg | 208 | -0.292 | 1.8e-05 | -0.472 | 6.4e-13 | -0.471 | 7.2e-13 | 0.122 | 0.08 |
| alpha_median | 208 | -0.316 | 3.2e-06 | -0.434 | 6.1e-11 | -0.430 | 9e-11 | 0.020 | 0.77 |
| alpha_weighted_avg | 208 | 0.219 | 0.0015 | -0.072 | 0.3 | -0.069 | 0.32 | 0.509 | 4e-15 |
| log_spectral_norm_avg | 208 | 0.357 | 1.2e-07 | 0.220 | 0.0014 | 0.223 | 0.0012 | 0.396 | 3.1e-09 |
| log_norm_avg | 208 | 0.245 | 0.00036 | 0.048 | 0.49 | 0.051 | 0.46 | 0.366 | 5.3e-08 |
| log_alpha_norm_avg | 208 | 0.216 | 0.0017 | -0.066 | 0.34 | -0.063 | 0.36 | 0.495 | 3e-14 |
| stable_rank_avg | 208 | -0.228 | 0.00092 | -0.348 | 2.7e-07 | -0.346 | 3e-07 | 0.060 | 0.39 |
| mp_softrank_avg | 208 | -0.336 | 6.9e-07 | -0.482 | 1.8e-13 | -0.480 | 2.1e-13 | 0.026 | 0.71 |
| rand_distance_avg | 208 | 0.299 | 1.2e-05 | 0.402 | 1.8e-09 | 0.399 | 2.4e-09 | 0.027 | 0.7 |
| top_sv_energy_avg | 208 | 0.130 | 0.061 | 0.120 | 0.084 | 0.118 | 0.089 | 0.149 | 0.032 |
| mean_dir_overlap_left_avg | 208 | -0.006 | 0.93 | 0.107 | 0.13 | 0.110 | 0.11 | 0.043 | 0.53 |
| mean_dir_overlap_right_avg | 208 | -0.100 | 0.15 | 0.019 | 0.79 | 0.024 | 0.73 | 0.009 | 0.9 |
| effective_rank_avg | 208 | -0.155 | 0.025 | -0.223 | 0.0012 | -0.218 | 0.0015 | 0.004 | 0.96 |
| spectral_gap_avg | 208 | 0.077 | 0.27 | 0.054 | 0.44 | 0.058 | 0.41 | 0.246 | 0.00035 |
| frac_dead_rows_avg | 208 | 0.146 | 0.036 | 0.312 | 4.6e-06 | 0.308 | 6.1e-06 | -0.108 | 0.12 |
| frac_alpha_lt2 | 208 | 0.207 | 0.0027 | 0.291 | 1.9e-05 | 0.288 | 2.5e-05 | -0.012 | 0.87 |
| frac_alpha_2_4 | 208 | 0.147 | 0.034 | 0.117 | 0.092 | 0.124 | 0.075 | 0.099 | 0.15 |
| n_over_trained | 208 | 0.222 | 0.0013 | 0.216 | 0.0017 | 0.211 | 0.0022 | 0.116 | 0.094 |
| n_under_trained | 208 | -0.265 | 0.00011 | -0.547 | 1.3e-17 | -0.548 | 1e-17 | 0.313 | 4.1e-06 |
| alpha_role_embed | 68 | -0.041 | 0.74 | -0.149 | 0.22 | -0.146 | 0.23 | 0.111 | 0.37 |
| alpha_role_l3_wk | 68 | -0.246 | 0.043 | -0.303 | 0.012 | -0.300 | 0.013 | -0.088 | 0.47 |
| alpha_role_l3_wmix | 68 | -0.301 | 0.013 | -0.277 | 0.022 | -0.273 | 0.024 | -0.223 | 0.068 |
| alpha_role_l3_wup | 45 | -0.168 | 0.27 | -0.102 | 0.5 | -0.100 | 0.52 | -0.136 | 0.37 |
| alpha_role_l3_wv | 45 | -0.232 | 0.13 | -0.192 | 0.21 | -0.190 | 0.21 | -0.112 | 0.47 |
| alpha_role_mlp | 140 | -0.373 | 5.6e-06 | -0.412 | 4.3e-07 | -0.407 | 5.9e-07 | -0.099 | 0.25 |


## Per-layer alpha by role (final epoch, all runs)

| role | count | mean | median | min | max |
|---|---|---|---|---|---|
| embed | 136 | 6.453 | 5.305 | 1.186 | 24.063 |
| head | 208 | 2.715 | 2.376 | 1.095 | 8.812 |
| l3_wk | 146 | 6.863 | 3.463 | -1.000 | 42.800 |
| l3_wmix | 146 | 3.122 | 2.309 | 1.190 | 15.208 |
| l3_wup | 146 | 3.116 | 2.772 | -1.000 | 14.664 |
| l3_wv | 146 | 7.563 | 5.641 | -1.000 | 55.407 |
| mlp | 280 | 4.076 | 2.386 | 1.126 | 43.730 |


## Top runs by val_acc

| run_id | model | val_acc | train_acc | gap | alpha_avg | alpha_default_avg | alpha_min50_avg | cfg_h | cfg_lr | cfg_epochs | cfg_train_subset | cfg_label_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run_s1_0107 | MLP3 | 0.8919 | 0.9113 | 0.01944 | 3.164 | 3.044 | 3.164 | 1024 | 0.0004716 | 8 | 60000 | 0 |
| run_s1_0071 | MLP3 | 0.8851 | 0.8113 | -0.0738 | 3.576 | 3.58 | 3.576 | 256 | 0.003838 | 20 | 60000 | 0.1 |
| run_s1_0058 | MLP3 | 0.8814 | 0.8978 | 0.01636 | 2.888 | 2.959 | 2.888 | 128 | 0.0003339 | 14 | 60000 | 0 |
| run_s1_0091 | MLP3 | 0.8772 | 0.6329 | -0.2443 | 1.77 | 2.312 | 1.77 | 256 | 0.0001157 | 15 | 60000 | 0.3 |
| run_s1_0099 | MLP3 | 0.8724 | 0.631 | -0.2414 | 1.868 | 2.477 | 1.868 | 512 | 0.002655 | 11 | 60000 | 0.3 |
| run_s1_0119 | MLP3 | 0.8712 | 0.7941 | -0.0771 | 2.469 | 2.338 | 2.469 | 256 | 0.009979 | 17 | 60000 | 0.1 |
| run_s3_0018 | SNNMLP3 | 0.8707 | 0.8829 | 0.01222 | 1.717 | 1.917 | 1.717 | 128 | 0.001727 | 5 | 60000 |  |
| run_s3_0005 | SNNMLP3 | 0.8699 | 0.8758 | 0.00586 | 1.641 | 1.802 | 1.641 | 128 | 0.003044 | 4 | 60000 |  |
| run_s1_0012 | MLP3 | 0.8662 | 0.8702 | 0.00402 | 4.175 | 3.652 | 4.175 | 256 | 0.003834 | 20 | 60000 | 0 |
| run_s3_0003 | SNNMLP3 | 0.8648 | 0.8719 | 0.00714 | 2.114 | 2.201 | 2.114 | 256 | 0.0008762 | 3 | 60000 |  |


## Figures

- `alpha_vs_acc__alpha_default_avg.png`
- `alpha_vs_acc__alpha_min50_avg.png`
- `alpha_vs_acc__alpha_weighted_avg.png`
- `alpha_by_role_vs_val_acc.png`
- `alpha_trajectories.png`
- `alpha_hist_by_role.png`
