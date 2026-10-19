# Control experiments: what moves alpha?

Columns: alpha at init / after training (WeightWatcher, min_evals=50), warning, alpha after clipping fingers (fix_fingers=clip_xmax), 
alpha after projecting out the layer's input-mean direction (W(I - m m^T)), alpha of the live-unit sub-matrix, energy of W along the input mean, 
overlap of the top right-singular vector with the column-mean direction, top-singular-value energy, fraction of live input units, Hill tail index of |W| entries, excess kurtosis of entries, xmin and number of eigenvalues used by the PL fit.

| model | val acc | train acc | layer | a_init | a_trained | warning | a_clip | a_meanproj | a_live | E_along_mean | ovl_mean | top_sv_E | live_in | hill_mu | kurt | xmin | n_pl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l3_classifier | 0.845 | 0.897 | embed | 19.13 | 4.72 |  | 4.72 |  |  |  | 0.02 | 0.011 |  | 5.9 | 0.02 | 782.578 | 60 |
| l3_classifier | 0.845 | 0.897 | layers.0.w_k | 17.70 | 2.70 |  | 2.70 |  |  |  | 0.06 | 0.042 |  | 5.3 | 0.01 | 4.056 | 245 |
| l3_classifier | 0.845 | 0.897 | layers.0.w_v | 11.76 | 3.57 |  | 3.57 |  |  |  | 0.08 | 0.060 |  | 6.5 | -0.56 | 16.794 | 64 |
| l3_classifier | 0.845 | 0.897 | layers.0.w_mix | 20.31 | 2.01 |  | 2.01 |  |  |  | 0.01 | 0.104 |  | 6.2 | -0.10 | 0.221 | 156 |
| l3_classifier | 0.845 | 0.897 | layers.0.w_up | 4.92 | 2.17 |  | 2.17 |  |  |  | 0.12 | 0.068 |  | 8.3 | -0.81 | 0.756 | 45 |
| l3_classifier | 0.845 | 0.897 | layers.1.w_k | 17.77 | 2.62 |  | 2.62 |  |  |  | 0.10 | 0.040 |  | 5.4 | -0.03 | 3.951 | 244 |
| l3_classifier | 0.845 | 0.897 | layers.1.w_v | 9.79 | 3.55 |  | 3.55 |  |  |  | 0.09 | 0.067 |  | 6.5 | -0.56 | 16.477 | 64 |
| l3_classifier | 0.845 | 0.897 | layers.1.w_mix | 4.79 | 2.00 |  | 2.00 |  |  |  | 0.11 | 0.231 |  | 6.0 | 0.03 | 0.192 | 167 |
| l3_classifier | 0.845 | 0.897 | layers.1.w_up | 13.50 | 2.07 |  | 2.07 |  |  |  | 0.03 | 0.100 |  | 8.5 | -0.72 | 0.644 | 43 |
| mlp3 | 0.891 | 0.919 | fc1 | 34.93 | 1.73 | over-trained | 1.73 | 1.73 | 1.70 | 0.016 | 0.47 | 0.073 | 1.00 | 3.6 | 3.45 | 0.775 | 200 |
| mlp3 | 0.891 | 0.919 | fc2 | 8.29 | 2.00 | over-trained | 2.00 | 2.00 | 1.91 | 0.014 | 0.21 | 0.127 | 0.80 | 3.7 | 2.73 | 0.713 | 161 |
| mlp3_centered | 0.883 | 0.907 | fc1 | 34.93 | 1.69 | over-trained | 1.69 | 1.69 | 1.67 | 0.020 | 0.30 | 0.078 | 1.00 | 3.5 | 3.69 | 0.446 | 257 |
| mlp3_centered | 0.883 | 0.907 | fc2 | 8.29 | 1.88 | over-trained | 1.88 | 1.88 | 1.85 | 0.002 | 0.22 | 0.149 | 0.86 | 3.4 | 3.63 | 0.494 | 195 |
| snnl3_spiking_memory | 0.783 | 0.820 | embed | 19.13 | 16.99 | under-trained | 4.86 |  |  |  | 0.02 | 0.011 |  | 5.9 | 0.01 | 1271.605 | 10 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.0.w_k | 17.70 | 3.32 |  | 3.32 |  |  |  | 0.98 | 0.042 |  | 6.7 | -0.54 | 4.623 | 213 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.0.w_v | 11.76 | 4.46 |  | 4.46 |  |  |  | 0.05 | 0.134 |  | 7.5 | -0.80 | 17.730 | 61 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.0.w_up | 2.92 | 2.14 |  | 2.14 |  |  |  | 0.06 | 0.313 |  | 9.8 | -0.80 | 0.853 | 49 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.0.w_mix | 5.00 | 2.23 |  | 2.23 |  |  |  | 0.09 | 0.341 |  | 5.6 | 0.41 | 0.366 | 125 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.1.w_k | 17.77 | 3.16 |  | 3.16 |  |  |  | 0.99 | 0.054 |  | 6.5 | -0.49 | 18.106 | 11 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.1.w_v | 9.79 | 3.89 |  | 3.89 |  |  |  | 0.08 | 0.121 |  | 6.6 | -0.64 | 17.009 | 64 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.1.w_up | 5.57 | 2.07 |  | 2.07 |  |  |  | 0.03 | 0.294 |  | 13.2 | -0.95 | 0.880 | 49 |
| snnl3_spiking_memory | 0.783 | 0.820 | blocks.1.w_mix | 5.24 | 2.11 |  | 2.11 |  |  |  | 0.09 | 0.307 |  | 6.7 | 0.31 | 0.311 | 143 |
| snnmlp3 | 0.887 | 0.914 | fc1 | 34.93 | 1.73 | over-trained | 1.73 | 1.74 | 1.72 | 0.021 | 0.42 | 0.075 | 1.00 | 3.8 | 2.60 | 1.158 | 171 |
| snnmlp3 | 0.887 | 0.914 | fc2 | 8.29 | 1.89 | over-trained | 1.89 | 1.90 | 1.86 | 0.006 | 0.14 | 0.118 | 0.79 | 3.8 | 2.53 | 0.674 | 181 |
| snnmlp3 |  |  | spike rates | fc1: rate 0.18, silent 0.07; fc2: rate 0.37, silent 0.01 | | | | | | | | | | | | | |
| snnmlp3_centered | 0.881 | 0.908 | fc1 | 34.93 | 1.71 | over-trained | 1.71 | 1.72 | 1.73 | 0.050 | 0.83 | 0.079 | 1.00 | 3.7 | 2.58 | 0.722 | 232 |
| snnmlp3_centered | 0.881 | 0.908 | fc2 | 8.29 | 1.87 | over-trained | 1.87 | 1.88 | 1.82 | 0.011 | 0.03 | 0.116 | 0.72 | 4.0 | 1.87 | 0.560 | 194 |
| snnmlp3_centered |  |  | spike rates | fc1: rate 0.40, silent 0.12; fc2: rate 0.35, silent 0.05 | | | | | | | | | | | | | |
| snnmlp3_k25 | 0.203 | 0.200 | fc1 | 34.93 | 3.16 |  | 3.16 | 3.58 |  | 0.152 | 0.91 | 0.168 | 1.00 | 3.2 | 1.59 | 0.559 | 131 |
| snnmlp3_k25 | 0.203 | 0.200 | fc2 | 8.29 | 3.98 |  | 3.98 | 4.03 |  | 0.007 | 0.23 | 0.046 | 0.05 | 5.6 | 3.63 | 0.820 | 74 |
| snnmlp3_k25 |  |  | spike rates | fc1: rate 0.05, silent 0.94; fc2: rate 0.03, silent 0.92 | | | | | | | | | | | | | |
