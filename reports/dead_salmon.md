*AI-assistance disclosure: see [`AI_ASSISTANCE.md`](./AI_ASSISTANCE.md).*

# Dead-Salmon Random-Direction Control

Tests DIRECTION-specificity of the suppression off-manifold effect (complement to `reconstruction_error.md`'s sign-specificity): do the paper's SELECTED max-activating features fall further off-manifold than a random per-layer POOL of SAE dictionary latents steered identically, or is the collapse generic (a dead salmon)?

Control = a shared pool of n_pool=100 random dictionary latents per layer (seed 20260714). The bump is feature-agnostic, so controls are not matched to individual selected features; the test is therefore an UNPAIRED per-layer Mann-Whitney U (selected population vs. control pool), p_fdr = Benjamini-Hochberg across the 26 layers. Primary metric `offmanifold_err`; output score reported secondarily.


## s = -10 — off-manifold error (primary)

Positive `sel-ctrl` = selected features fall FURTHER off-manifold than the random control pool (direction-specific). ~0 = dead salmon (generic).

| layer | n_sel | n_ctrl | sel | ctrl | sel-ctrl | rb | mwu p | p_fdr | unsteered base |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 100 | 2162.4817 | 2174.6500 | -12.1683 | +0.029 | 7.22e-01 | 9.00e-01 | 7.6324 |
| 1 | 100 | 100 | 1172.7026 | 1220.3599 | -47.6572 | -0.119 | 1.48e-01 | 9.00e-01 | 11.0113 |
| 2 | 100 | 100 | 542.3447 | 576.8046 | -34.4599 | -0.089 | 2.80e-01 | 9.00e-01 | 11.1749 |
| 3 | 100 | 100 | 1184.9180 | 1268.7854 | -83.8674 | -0.116 | 1.58e-01 | 9.00e-01 | 17.0303 |
| 4 | 100 | 100 | 938.9177 | 906.5911 | +32.3265 | -0.004 | 9.58e-01 | 9.91e-01 | 19.9910 |
| 5 | 100 | 100 | 1289.2614 | 1299.5330 | -10.2716 | -0.025 | 7.61e-01 | 9.00e-01 | 28.8003 |
| 6 | 100 | 100 | 1221.1470 | 1213.9109 | +7.2360 | -0.001 | 9.91e-01 | 9.91e-01 | 30.7281 |
| 7 | 100 | 100 | 1256.6881 | 1189.1560 | +67.5321 | +0.047 | 5.63e-01 | 9.00e-01 | 33.5285 |
| 8 | 100 | 100 | 982.9152 | 974.9887 | +7.9265 | -0.041 | 6.14e-01 | 9.00e-01 | 32.9258 |
| 9 | 100 | 100 | 1006.4240 | 1062.5312 | -56.1072 | -0.051 | 5.32e-01 | 9.00e-01 | 40.3278 |
| 10 | 100 | 100 | 969.4660 | 1059.7379 | -90.2718 | -0.114 | 1.63e-01 | 9.00e-01 | 44.8877 |
| 11 | 100 | 100 | 916.5122 | 838.6116 | +77.9006 | +0.066 | 4.18e-01 | 9.00e-01 | 46.8081 |
| 12 | 100 | 100 | 1157.7774 | 1062.5802 | +95.1971 | +0.058 | 4.79e-01 | 9.00e-01 | 48.6967 |
| 13 | 100 | 100 | 1049.9367 | 1026.8926 | +23.0441 | +0.043 | 5.97e-01 | 9.00e-01 | 55.1557 |
| 14 | 100 | 100 | 963.2962 | 948.1897 | +15.1065 | +0.036 | 6.59e-01 | 9.00e-01 | 58.2486 |
| 15 | 100 | 100 | 1496.6278 | 1416.9959 | +79.6319 | +0.025 | 7.63e-01 | 9.00e-01 | 61.5322 |
| 16 | 100 | 100 | 1447.6034 | 1471.9562 | -24.3528 | -0.043 | 5.99e-01 | 9.00e-01 | 70.8809 |
| 17 | 100 | 100 | 2603.6757 | 2647.8034 | -44.1277 | -0.057 | 4.84e-01 | 9.00e-01 | 76.4641 |
| 18 | 100 | 100 | 2700.6683 | 2814.6054 | -113.9371 | -0.021 | 8.02e-01 | 9.00e-01 | 85.7985 |
| 19 | 100 | 100 | 2724.7285 | 2641.5984 | +83.1301 | +0.100 | 2.24e-01 | 9.00e-01 | 87.9532 |
| 20 | 100 | 100 | 2625.7054 | 2460.8751 | +164.8302 | +0.101 | 2.20e-01 | 9.00e-01 | 95.9100 |
| 21 | 100 | 100 | 2350.5683 | 2435.5127 | -84.9444 | -0.038 | 6.47e-01 | 9.00e-01 | 111.9536 |
| 22 | 100 | 100 | 2608.6797 | 2641.1674 | -32.4877 | -0.062 | 4.48e-01 | 9.00e-01 | 124.8297 |
| 23 | 100 | 100 | 3223.0462 | 3069.5507 | +153.4955 | +0.123 | 1.34e-01 | 9.00e-01 | 135.2414 |
| 24 | 100 | 100 | 3162.5407 | 3210.8897 | -48.3490 | -0.018 | 8.31e-01 | 9.00e-01 | 153.8924 |
| 25 | 100 | 100 | 1094.8677 | 1065.1613 | +29.7064 | +0.075 | 3.63e-01 | 9.00e-01 | 156.2935 |

**0/26 layers FDR<0.05; sel>ctrl in 13/26 layers. Largest |rb| = 0.123, well under the design's minimum detectable |rb| ≈ 0.23** (n=100/100, two-sided α=0.05, 80% power; Noether's WMW formula). The MDE is a post-hoc DESIGN bound, not observed power; it uses nominal α, so under BH-FDR the truly detectable effect is only larger — i.e. this is a conservative upper bound on any hidden selected-vs-random difference, not proof of none.


## s = -10 — output score (secondary)

Same unpaired per-layer comparison on the (signed) output score. Logit-lens token sets always exist for Gemma, so no feature is dropped; sign interpretation on a neutral prompt is murky, hence secondary.

| layer | n_sel | n_ctrl | sel | ctrl | sel-ctrl | rb | mwu p | p_fdr |
|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 100 | 0.0014 | 0.0057 | -0.0043 | -0.017 | 8.36e-01 | 9.34e-01 |
| 1 | 100 | 100 | 0.0041 | 0.0028 | +0.0012 | -0.172 | 3.55e-02 | 7.97e-01 |
| 2 | 100 | 100 | 0.0007 | 0.0004 | +0.0003 | -0.039 | 6.36e-01 | 8.30e-01 |
| 3 | 100 | 100 | 0.0023 | 0.0022 | +0.0001 | -0.067 | 4.17e-01 | 7.97e-01 |
| 4 | 100 | 100 | 0.0023 | 0.0003 | +0.0020 | +0.042 | 6.07e-01 | 8.30e-01 |
| 5 | 100 | 100 | 0.0019 | 0.0110 | -0.0092 | +0.039 | 6.38e-01 | 8.30e-01 |
| 6 | 100 | 100 | 0.0035 | 0.0001 | +0.0034 | -0.012 | 8.84e-01 | 9.34e-01 |
| 7 | 100 | 100 | 0.0027 | 0.0010 | +0.0017 | -0.120 | 1.42e-01 | 7.97e-01 |
| 8 | 100 | 100 | 0.0006 | 0.0003 | +0.0003 | +0.002 | 9.81e-01 | 9.81e-01 |
| 9 | 100 | 100 | 0.0010 | 0.0004 | +0.0007 | -0.085 | 3.00e-01 | 7.97e-01 |
| 10 | 100 | 100 | 0.0014 | 0.0003 | +0.0011 | +0.053 | 5.21e-01 | 7.97e-01 |
| 11 | 100 | 100 | 0.0000 | 0.0025 | -0.0025 | -0.096 | 2.40e-01 | 7.97e-01 |
| 12 | 100 | 100 | 0.0006 | 0.0007 | -0.0001 | -0.068 | 4.07e-01 | 7.97e-01 |
| 13 | 100 | 100 | 0.0050 | 0.0009 | +0.0041 | +0.063 | 4.44e-01 | 7.97e-01 |
| 14 | 100 | 100 | 0.0046 | 0.0029 | +0.0017 | +0.011 | 8.98e-01 | 9.34e-01 |
| 15 | 100 | 100 | 0.0012 | 0.0009 | +0.0004 | +0.116 | 1.55e-01 | 7.97e-01 |
| 16 | 100 | 100 | 0.0006 | 0.0061 | -0.0056 | -0.056 | 4.93e-01 | 7.97e-01 |
| 17 | 100 | 100 | 0.0014 | 0.0000 | +0.0014 | +0.102 | 2.14e-01 | 7.97e-01 |
| 18 | 100 | 100 | 0.0039 | 0.0000 | +0.0039 | +0.150 | 6.67e-02 | 7.97e-01 |
| 19 | 100 | 100 | 0.0002 | 0.0001 | +0.0002 | -0.103 | 2.10e-01 | 7.97e-01 |
| 20 | 100 | 100 | 0.0001 | 0.0004 | -0.0003 | +0.068 | 4.10e-01 | 7.97e-01 |
| 21 | 100 | 100 | 0.0041 | 0.0001 | +0.0040 | -0.078 | 3.39e-01 | 7.97e-01 |
| 22 | 100 | 100 | 0.0000 | 0.0004 | -0.0004 | -0.087 | 2.90e-01 | 7.97e-01 |
| 23 | 100 | 100 | 0.0000 | 0.0000 | -0.0000 | -0.030 | 7.11e-01 | 8.73e-01 |
| 24 | 100 | 100 | 0.0000 | 0.0000 | -0.0000 | -0.027 | 7.39e-01 | 8.73e-01 |
| 25 | 100 | 100 | 0.0000 | 0.0000 | -0.0000 | -0.053 | 5.18e-01 | 7.97e-01 |

**0/26 layers FDR<0.05; sel>ctrl in 16/26 layers. Largest |rb| = 0.172, well under the design's minimum detectable |rb| ≈ 0.23** (n=100/100, two-sided α=0.05, 80% power; Noether's WMW formula). The MDE is a post-hoc DESIGN bound, not observed power; it uses nominal α, so under BH-FDR the truly detectable effect is only larger — i.e. this is a conservative upper bound on any hidden selected-vs-random difference, not proof of none.


### Candidate spotlight (s = -10, off-manifold)

Percentile = fraction of the layer's control pool with LOWER off-manifold error than the candidate (near 50% = ordinary, i.e. fails specificity).

| candidate | sel off-mani | pool mean | pool max | pctile vs pool |
|---|---|---|---|---|
| 14_6669 | 734.8422 | 948.1897 | 3126.5926 | 58% |
| 2_13823 | 124.3919 | 576.8046 | 1264.2216 | 0% |
| 7_12166 | 1605.8407 | 1189.1560 | 2735.4213 | 78% |
| 7_6944 | 1255.7278 | 1189.1560 | 2735.4213 | 66% |

## s = +10 — off-manifold error (primary)

Positive `sel-ctrl` = selected features fall FURTHER off-manifold than the random control pool (direction-specific). ~0 = dead salmon (generic).

| layer | n_sel | n_ctrl | sel | ctrl | sel-ctrl | rb | mwu p | p_fdr | unsteered base |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 100 | 2674.3224 | 2610.0443 | +64.2780 | -0.039 | 6.35e-01 | 9.52e-01 | 7.6324 |
| 1 | 100 | 100 | 1415.1991 | 1390.6943 | +24.5047 | +0.053 | 5.21e-01 | 9.52e-01 | 11.0113 |
| 2 | 100 | 100 | 721.7128 | 719.6333 | +2.0795 | -0.021 | 8.02e-01 | 9.52e-01 | 11.1749 |
| 3 | 100 | 100 | 1346.4507 | 1397.9117 | -51.4610 | -0.028 | 7.31e-01 | 9.52e-01 | 17.0303 |
| 4 | 100 | 100 | 1151.8141 | 1119.7504 | +32.0637 | +0.044 | 5.92e-01 | 9.52e-01 | 19.9910 |
| 5 | 100 | 100 | 1472.3894 | 1204.2402 | +268.1492 | +0.074 | 3.64e-01 | 9.52e-01 | 28.8003 |
| 6 | 100 | 100 | 1180.7980 | 1220.1172 | -39.3192 | -0.015 | 8.56e-01 | 9.52e-01 | 30.7281 |
| 7 | 100 | 100 | 1197.0044 | 1201.1749 | -4.1705 | +0.076 | 3.55e-01 | 9.52e-01 | 33.5285 |
| 8 | 100 | 100 | 900.5607 | 866.7254 | +33.8353 | +0.017 | 8.38e-01 | 9.52e-01 | 32.9258 |
| 9 | 100 | 100 | 887.9443 | 933.8961 | -45.9518 | +0.040 | 6.29e-01 | 9.52e-01 | 40.3278 |
| 10 | 100 | 100 | 928.3019 | 919.6982 | +8.6038 | +0.031 | 7.08e-01 | 9.52e-01 | 44.8877 |
| 11 | 100 | 100 | 798.2864 | 760.8740 | +37.4125 | -0.018 | 8.25e-01 | 9.52e-01 | 46.8081 |
| 12 | 100 | 100 | 1087.9852 | 1015.6794 | +72.3058 | +0.033 | 6.90e-01 | 9.52e-01 | 48.6967 |
| 13 | 100 | 100 | 1037.9004 | 1023.3775 | +14.5229 | +0.012 | 8.82e-01 | 9.52e-01 | 55.1557 |
| 14 | 100 | 100 | 927.2856 | 977.6321 | -50.3465 | +0.035 | 6.70e-01 | 9.52e-01 | 58.2486 |
| 15 | 100 | 100 | 1353.7743 | 1359.6691 | -5.8948 | +0.057 | 4.88e-01 | 9.52e-01 | 61.5322 |
| 16 | 100 | 100 | 1475.2214 | 1374.5200 | +100.7014 | +0.148 | 7.00e-02 | 9.52e-01 | 70.8809 |
| 17 | 100 | 100 | 2316.9587 | 2379.6505 | -62.6918 | -0.047 | 5.65e-01 | 9.52e-01 | 76.4641 |
| 18 | 100 | 100 | 2319.4172 | 2567.1894 | -247.7723 | +0.010 | 9.04e-01 | 9.52e-01 | 85.7985 |
| 19 | 100 | 100 | 2161.0821 | 2151.7213 | +9.3608 | -0.097 | 2.38e-01 | 9.52e-01 | 87.9532 |
| 20 | 100 | 100 | 1982.1839 | 2080.7937 | -98.6097 | -0.012 | 8.86e-01 | 9.52e-01 | 95.9100 |
| 21 | 100 | 100 | 1940.2097 | 2114.2038 | -173.9941 | -0.118 | 1.48e-01 | 9.52e-01 | 111.9536 |
| 22 | 100 | 100 | 2249.3601 | 2269.1906 | -19.8305 | -0.005 | 9.50e-01 | 9.52e-01 | 124.8297 |
| 23 | 100 | 100 | 2812.1156 | 2891.9425 | -79.8269 | -0.058 | 4.78e-01 | 9.52e-01 | 135.2414 |
| 24 | 100 | 100 | 3071.3761 | 3089.4312 | -18.0551 | -0.005 | 9.52e-01 | 9.52e-01 | 153.8924 |
| 25 | 100 | 100 | 1311.0774 | 1265.4535 | +45.6239 | +0.046 | 5.77e-01 | 9.52e-01 | 156.2935 |

**0/26 layers FDR<0.05; sel>ctrl in 13/26 layers. Largest |rb| = 0.148, well under the design's minimum detectable |rb| ≈ 0.23** (n=100/100, two-sided α=0.05, 80% power; Noether's WMW formula). The MDE is a post-hoc DESIGN bound, not observed power; it uses nominal α, so under BH-FDR the truly detectable effect is only larger — i.e. this is a conservative upper bound on any hidden selected-vs-random difference, not proof of none.


## s = +10 — output score (secondary)

Same unpaired per-layer comparison on the (signed) output score. Logit-lens token sets always exist for Gemma, so no feature is dropped; sign interpretation on a neutral prompt is murky, hence secondary.

| layer | n_sel | n_ctrl | sel | ctrl | sel-ctrl | rb | mwu p | p_fdr |
|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 100 | 0.0093 | 0.0240 | -0.0147 | -0.097 | 2.38e-01 | 5.58e-01 |
| 1 | 100 | 100 | 0.0106 | 0.0090 | +0.0016 | +0.166 | 4.32e-02 | 5.58e-01 |
| 2 | 100 | 100 | 0.0029 | 0.0018 | +0.0010 | -0.050 | 5.42e-01 | 7.42e-01 |
| 3 | 100 | 100 | 0.0100 | 0.0081 | +0.0018 | -0.033 | 6.90e-01 | 8.54e-01 |
| 4 | 100 | 100 | 0.0159 | 0.0025 | +0.0134 | +0.085 | 3.00e-01 | 5.59e-01 |
| 5 | 100 | 100 | 0.0076 | 0.0226 | -0.0149 | -0.105 | 1.98e-01 | 5.58e-01 |
| 6 | 100 | 100 | 0.0191 | 0.0052 | +0.0139 | -0.006 | 9.44e-01 | 9.80e-01 |
| 7 | 100 | 100 | 0.0241 | 0.0163 | +0.0078 | -0.053 | 5.17e-01 | 7.42e-01 |
| 8 | 100 | 100 | 0.0113 | 0.0048 | +0.0064 | +0.005 | 9.48e-01 | 9.80e-01 |
| 9 | 100 | 100 | 0.0089 | 0.0075 | +0.0014 | +0.002 | 9.80e-01 | 9.80e-01 |
| 10 | 100 | 100 | 0.0144 | 0.0156 | -0.0012 | -0.119 | 1.46e-01 | 5.58e-01 |
| 11 | 100 | 100 | 0.0021 | 0.0098 | -0.0077 | -0.096 | 2.42e-01 | 5.58e-01 |
| 12 | 100 | 100 | 0.0042 | 0.0226 | -0.0184 | -0.110 | 1.78e-01 | 5.58e-01 |
| 13 | 100 | 100 | 0.0265 | 0.0145 | +0.0120 | +0.038 | 6.45e-01 | 8.39e-01 |
| 14 | 100 | 100 | 0.0312 | 0.0103 | +0.0209 | +0.112 | 1.72e-01 | 5.58e-01 |
| 15 | 100 | 100 | 0.0693 | 0.0480 | +0.0213 | +0.094 | 2.53e-01 | 5.58e-01 |
| 16 | 100 | 100 | 0.0398 | 0.0589 | -0.0191 | +0.028 | 7.31e-01 | 8.64e-01 |
| 17 | 100 | 100 | 0.1629 | 0.1400 | +0.0229 | +0.113 | 1.69e-01 | 5.58e-01 |
| 18 | 100 | 100 | 0.2031 | 0.1852 | +0.0179 | +0.017 | 8.33e-01 | 9.41e-01 |
| 19 | 100 | 100 | 0.1747 | 0.1823 | -0.0076 | -0.085 | 3.01e-01 | 5.59e-01 |
| 20 | 100 | 100 | 0.1600 | 0.1764 | -0.0163 | -0.054 | 5.13e-01 | 7.42e-01 |
| 21 | 100 | 100 | 0.1948 | 0.1625 | +0.0324 | +0.076 | 3.51e-01 | 6.09e-01 |
| 22 | 100 | 100 | 0.2480 | 0.2208 | +0.0271 | +0.094 | 2.50e-01 | 5.58e-01 |
| 23 | 100 | 100 | 0.2330 | 0.2829 | -0.0499 | -0.124 | 1.29e-01 | 5.58e-01 |
| 24 | 100 | 100 | 0.2622 | 0.2930 | -0.0307 | -0.066 | 4.24e-01 | 6.88e-01 |
| 25 | 100 | 100 | 0.2096 | 0.1670 | +0.0426 | +0.093 | 2.57e-01 | 5.58e-01 |

**0/26 layers FDR<0.05; sel>ctrl in 16/26 layers. Largest |rb| = 0.166, well under the design's minimum detectable |rb| ≈ 0.23** (n=100/100, two-sided α=0.05, 80% power; Noether's WMW formula). The MDE is a post-hoc DESIGN bound, not observed power; it uses nominal α, so under BH-FDR the truly detectable effect is only larger — i.e. this is a conservative upper bound on any hidden selected-vs-random difference, not proof of none.


### Candidate spotlight (s = +10, off-manifold)

Percentile = fraction of the layer's control pool with LOWER off-manifold error than the candidate (near 50% = ordinary, i.e. fails specificity).

| candidate | sel off-mani | pool mean | pool max | pctile vs pool |
|---|---|---|---|---|
| 14_6669 | 550.0146 | 977.6321 | 3346.9868 | 30% |
| 2_13823 | 2647.0031 | 719.6333 | 3280.5711 | 99% |
| 7_12166 | 359.6771 | 1201.1749 | 5996.4903 | 2% |
| 7_6944 | 2543.0866 | 1201.1749 | 5996.4903 | 94% |

## Decoder-norm covariate

The only residual magnitude confound is ‖W_dec[feature]‖ (the bump is otherwise feature-agnostic). If the selected and control-pool decoder-norm distributions match per layer, the contrast above is not a norm artifact.

| layer | mean sel ‖W_dec‖ | mean ctrl ‖W_dec‖ | ratio |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.000 |
| 1 | 1.0000 | 1.0000 | 1.000 |
| 2 | 1.0000 | 1.0000 | 1.000 |
| 3 | 1.0000 | 1.0000 | 1.000 |
| 4 | 1.0000 | 1.0000 | 1.000 |
| 5 | 1.0000 | 1.0000 | 1.000 |
| 6 | 1.0000 | 1.0000 | 1.000 |
| 7 | 1.0000 | 1.0000 | 1.000 |
| 8 | 1.0000 | 1.0000 | 1.000 |
| 9 | 1.0000 | 1.0000 | 1.000 |
| 10 | 1.0000 | 1.0000 | 1.000 |
| 11 | 1.0000 | 1.0000 | 1.000 |
| 12 | 1.0000 | 1.0000 | 1.000 |
| 13 | 1.0000 | 1.0000 | 1.000 |
| 14 | 1.0000 | 1.0000 | 1.000 |
| 15 | 1.0000 | 1.0000 | 1.000 |
| 16 | 1.0000 | 1.0000 | 1.000 |
| 17 | 1.0000 | 1.0000 | 1.000 |
| 18 | 1.0000 | 1.0000 | 1.000 |
| 19 | 1.0000 | 1.0000 | 1.000 |
| 20 | 1.0000 | 1.0000 | 1.000 |
| 21 | 1.0000 | 1.0000 | 1.000 |
| 22 | 1.0000 | 1.0000 | 1.000 |
| 23 | 1.0000 | 1.0000 | 1.000 |
| 24 | 1.0000 | 1.0000 | 1.000 |
| 25 | 1.0000 | 1.0000 | 1.000 |

## Summary interpretation

**Dead salmon confirmed at both signs.** 0/26 layers pass FDR<0.05 for `offmanifold_err` at
s=-10 and 0/26 at s=+10 (same for the secondary output-score metric at both signs). Selected
features fall further off-manifold than their layer's random control pool in only 13/26 layers
at each sign — a coin-flip, not a directional pattern. No layer, including the 17-23 band flagged
in `reconstruction_error.md`, comes close to surviving correction (all `p_fdr` in that band
sit at ~0.9-0.95, indistinguishable from the rest of the range).

**The null is bounded small, not merely non-significant.** Beyond failing to reach significance,
the effect sizes are tiny: the largest per-layer rank-biserial `|rb|` anywhere in the grid is 0.17
(off-manifold and output score, both signs), well under the `|rb|≈0.23` this design could reliably
detect at n=100/100 and 80% power (Noether's WMW formula, nominal α=0.05; BH-FDR makes the truly
detectable effect only larger, so 0.23 is a conservative ceiling). So any selected-vs-random
off-manifold difference — if one exists at all — is bounded below `|rb|≈0.23`. This is an upper
bound on a hidden effect, computed as a post-hoc *design* property (not observed power), not proof
of exact equivalence; it closes the "was the null just underpowered?" objection without over-claiming.

The decoder-norm covariate rules out the one residual confound outright rather than just
controlling for it: ‖W_dec‖ = 1.0000 for both selected and control features in every layer
(ratio 1.000 throughout) — Gemma Scope's dictionaries are unit-norm by construction, so the
additive steering bump is identical in magnitude for selected and random features with no norm
adjustment needed.

The 4 spot-check candidates give no rescue either: percentiles vs. their layer's control pool
are unstable across sign (`2_13823`: 0% at s=-10 vs 99% at s=+10; `7_12166`: 78% vs 2%; `7_6944`:
66% vs 94%; `14_6669`: 58% vs 30%) — large swings in *opposite* directions for the same feature
under sign flip look like sampling noise against a 100-point control pool, not a stable
specificity signal in either direction.

**Conclusion:** the off-manifold collapse under steering is generic to the SAE dictionary, not
targeted at the paper's selected max-activating features. This closes the direction-specificity
loophole `reconstruction_error.md` could not address, and — combined with that report's own
layer-localized/reversed partial signal and `step4_coherence_spotcheck.md`'s coherence-tracks-|s|
finding — leaves the artifact reading as the
best-supported account of the apparent suppression asymmetry. Per the conditional-sweep rule, no
gap appeared at s=±10, so the -2/-20 salmon sweep is not run.
