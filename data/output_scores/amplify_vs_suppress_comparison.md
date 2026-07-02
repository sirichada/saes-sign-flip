# Amplification vs. Suppression output_score Comparison

- amplify cache: `C:\Users\sirichada\Github\saes-for-steering\data\output_scores\gemma2_2b_amplify.json` (2600 entries)
- suppress cache: `C:\Users\sirichada\Github\saes-for-steering\data\output_scores\gemma2_2b_suppress.json` (2600 entries)

## Key alignment

- common keys: 2600
- keys only in amplify: 0
- keys only in suppress: 0

## Overall distribution

| metric | amplify | suppress |
|---|---|---|
| mean | 0.0829 | 0.0017 |
| median | 0.0003 | 0.0000 |
| stdev | 0.1727 | 0.0162 |
| min | 0.0000 | 0.0000 |
| max | 0.9929 | 0.3571 |

## Paired delta (amplify - suppress)

- n = 2600
- mean delta: 0.0812
- median delta: 0.0003
- stdev delta: 0.1718
- fraction amplify > suppress: 2467/2600 = 94.9%
- fraction suppress > amplify: 133/2600 = 5.1%
- fraction tied: 0/2600 = 0.0%

- median relative reduction (1 - suppress/amplify, where amplify>0, n=2600): 1.0000
- mean relative reduction: -53.2030

## Significance test

- Wilcoxon signed-rank test: statistic=86102.00, p-value=0.000e+00

## Per-layer means (gemma2_2b, 26 layers, 0=earliest, 25=latest)

| layer | depth% | amplify mean | suppress mean | delta |
|---|---|---|---|---|
| 0 | 0.0% | 0.0093 | 0.0014 | 0.0079 |
| 1 | 4.0% | 0.0106 | 0.0041 | 0.0065 |
| 2 | 8.0% | 0.0029 | 0.0007 | 0.0022 |
| 3 | 12.0% | 0.0100 | 0.0023 | 0.0076 |
| 4 | 16.0% | 0.0159 | 0.0023 | 0.0135 |
| 5 | 20.0% | 0.0076 | 0.0019 | 0.0058 |
| 6 | 24.0% | 0.0191 | 0.0035 | 0.0156 |
| 7 | 28.0% | 0.0241 | 0.0027 | 0.0214 |
| 8 | 32.0% | 0.0113 | 0.0006 | 0.0107 |
| 9 | 36.0% | 0.0089 | 0.0010 | 0.0079 |
| 10 | 40.0% | 0.0144 | 0.0014 | 0.0129 |
| 11 | 44.0% | 0.0021 | 0.0000 | 0.0020 |
| 12 | 48.0% | 0.0042 | 0.0006 | 0.0036 |
| 13 | 52.0% | 0.0265 | 0.0050 | 0.0215 |
| 14 | 56.0% | 0.0312 | 0.0046 | 0.0266 |
| 15 | 60.0% | 0.0693 | 0.0012 | 0.0681 |
| 16 | 64.0% | 0.0398 | 0.0006 | 0.0392 |
| 17 | 68.0% | 0.1629 | 0.0014 | 0.1614 |
| 18 | 72.0% | 0.2031 | 0.0039 | 0.1991 |
| 19 | 76.0% | 0.1747 | 0.0002 | 0.1745 |
| 20 | 80.0% | 0.1600 | 0.0001 | 0.1599 |
| 21 | 84.0% | 0.1948 | 0.0041 | 0.1908 |
| 22 | 88.0% | 0.2480 | 0.0000 | 0.2479 |
| 23 | 92.0% | 0.2330 | 0.0000 | 0.2330 |
| 24 | 96.0% | 0.2622 | 0.0000 | 0.2622 |
| 25 | 100.0% | 0.2096 | 0.0000 | 0.2096 |

## Early vs. late layer bucketing

- early layers (0-50% depth, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]): amplify mean=0.0108, suppress mean=0.0017, delta=0.0090
- late layers (66-100% depth, [17, 18, 19, 20, 21, 22, 23, 24, 25]): amplify mean=0.2054, suppress mean=0.0011, delta=0.2043

## Correlation between amplify and suppress scores

- Pearson r = 0.1064 (p=5.400e-08)

(A high correlation would suggest amplify/suppress scores share a common confound — e.g. how predictable a feature's logit-lens tokens are in general — rather than reflecting the causal direction of the intervention. A low/negative correlation better supports output_score being sensitive to the intervention itself.)

## Top 10 features by delta (strongest suppression effect: amplify >> suppress)

| layer_feature | amplify | suppress | delta |
|---|---|---|---|
| 20_375 | 0.9929 | 0.0005 | 0.9924 |
| 17_12364 | 0.9911 | 0.0000 | 0.9911 |
| 25_14258 | 0.9910 | 0.0000 | 0.9910 |
| 21_15842 | 0.9822 | 0.0000 | 0.9822 |
| 21_3664 | 0.9561 | 0.0000 | 0.9561 |
| 24_9160 | 0.9503 | 0.0000 | 0.9503 |
| 18_7724 | 0.9484 | 0.0000 | 0.9484 |
| 24_10071 | 0.9458 | 0.0000 | 0.9458 |
| 19_9631 | 0.9672 | 0.0240 | 0.9432 |
| 22_280 | 0.9411 | 0.0000 | 0.9411 |

## Top 10 reversals (suppress > amplify by largest margin)

| layer_feature | amplify | suppress | delta (amp-sup) |
|---|---|---|---|
| 14_6669 | 0.0727 | 0.2343 | -0.1616 |
| 4_14288 | 0.0056 | 0.1478 | -0.1422 |
| 7_6944 | 0.0002 | 0.0902 | -0.0900 |
| 7_11721 | 0.0638 | 0.1450 | -0.0812 |
| 0_2832 | 0.0000 | 0.0487 | -0.0487 |
| 4_15169 | 0.0129 | 0.0546 | -0.0417 |
| 13_9962 | 0.1363 | 0.1729 | -0.0366 |
| 1_9779 | 0.1208 | 0.1487 | -0.0280 |
| 0_725 | 0.0046 | 0.0303 | -0.0257 |
| 2_13823 | 0.0002 | 0.0202 | -0.0200 |

## Summary interpretation

Across 2600 paired (layer, feature) entries, amplification produced a higher output_score than suppression in 94.9% of cases (mean delta = 0.0812, median delta = 0.0003).
This is directionally consistent with the expectation that suppressing a feature (amp_factor=-10) reduces how strongly the model's next-token prediction reflects that feature's associated logit-lens tokens, relative to amplifying it (amp_factor=+10) — supporting output_score as a metric that is sensitive to the causal direction of the steering intervention, not just to how 'predictable' a feature's tokens are in general.

