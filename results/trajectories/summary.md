| run | epochs | best val acc (epoch) | final train / val acc | alpha_avg at best-val epoch | alpha_avg final | epoch alpha_avg first < 2 | layers |
|---|---|---|---|---|---|---|---|
| l3_2layer | 12 | 0.846 (3) | 0.900 / 0.833 | 3.27 | 3.62 | None | layers.0.w_v, layers.0.w_up, layers.1.w_v, layers.1.w_mix, layers.0.w_k, layers.0.w_mix, layers.1.w_k, layers.1.w_up, embed |
| mlp3_h1024_sgd | 40 | 0.904 (30) | 0.973 / 0.901 | 2.02 | 2.0 | 35 | fc1, fc2 |
| mlp3_h512_clean | 40 | 0.905 (26) | 0.980 / 0.904 | 1.9 | 2.11 | 6 | fc2, fc1 |
| mlp3_h512_noise30 | 40 | 0.882 (13) | 0.794 / 0.780 | 1.99 | 2.72 | 8 | fc2, fc1 |
| snnl3_rate_query | 6 | 0.800 (6) | 0.859 / 0.800 | 2.99 | 2.99 | None | blocks.0.w_v, blocks.0.w_up, blocks.0.w_mix, blocks.1.w_v, blocks.1.w_up, blocks.1.w_mix, blocks.0.w_k, blocks.1.w_k, embed |
| snnl3_spiking_memory | 1 | 0.643 (1) | 0.649 / 0.643 | 5.17 | 5.17 | None | blocks.0.w_v, blocks.0.w_up, blocks.0.w_mix, blocks.1.w_v, blocks.1.w_up, blocks.1.w_mix, blocks.0.w_k, blocks.1.w_k |
| snnmlp3_h512 | 20 | 0.897 (18) | 0.934 / 0.888 | 1.79 | 1.79 | 4 | fc2, fc1 |
