
## Scores on the validation set

| ID | NaN approach | Balancing | Features | threshold | max_iter | gamma | lambda_1 | lambda_2 | converged? | threshold_pred | F1 | Accuracy | Confusion matrix (tp, fp, fn, tn) |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 1 | Drop NaN | None | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.2031 | 91.49 | (356, 285, 2507, 29665) |
| 2 | Drop NaN | Undersample to match | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.3681 | 76.66 | (2230, 7024, 633, 22926) |
| 3 | Drop NaN | Oversample to match | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.3682 | 76.69 | (2228, 7011, 635, 22939) |
| 4 | Drop NaN | Undersample 0.5, ratio for oversampling 1 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.3688 | 76.75 | (2228, 6993, 635, 22957) |
| 5 | Drop NaN | Undersample 0.5, ratio for oversampling 2 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4118 | 85.29 | (1690, 3655, 1173, 26295) |
| 6 | Drop NaN | Undersample 0.5, ratio for oversampling 3 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4092 | 88.51 | (1306, 2214, 1557, 27736) |
| 7 | Drop NaN | Oversample to half majority | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4117 | 85.31 | (1686, 3642, 1177, 26308) |
| 8 | Drop NaN | Undersample 0.5, ratio for oversampling 1.5 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4031 | 82.37 | (1953, 4875, 910, 25075) |
| 9 | Drop NaN | Undersample 0.5, ratio for oversampling 2.5 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4113 | 87.18 | (1469, 2811, 1394, 27139) |
| 10 | Mean/Mode | Undersample 0.5, ratio for oversampling 2 | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4074 | 84.41 | (1758, 4009, 1105, 25941) |
| 11 | Mean/Mode | Oversample to half majority | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.4076 | 84.42 | (1759, 4009, 1104, 25941) |
| 12 | Mean/Mode | None | The 19 | 1e-8 | 10000 | 0.05 | 0 | 0 | YES | 0.5 | 0.2070 | 91.48 | (365, 299, 2498, 29651) |