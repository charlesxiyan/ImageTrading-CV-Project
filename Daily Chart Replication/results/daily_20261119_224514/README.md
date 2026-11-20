# Daily JKX replication: full run

Settings: 5 seeds per cell, about 170,000 training images per cell (2016–2022), validation 2023, test 2024–2025.
The CNN has 708,866 parameters for I20 and 155,138 for I5.

| CNN AUC (95% date-bootstrap CI) | validation 2023 | test 2024–25 |
|---|---|---|
| dir I5/R5 | 0.515 (0.488–0.544) | 0.506 (0.489–0.523) |
| dir I20/R5 | 0.497 (0.460–0.529) | 0.484 (0.464–0.504) |
| dir I20/R20 | 0.509 (0.466–0.558) | 0.479 (0.458–0.505) |
| xsdir I5/R5 | 0.502 (0.491–0.514) | 0.503 (0.495–0.512) |
| xsdir I20/R5 | 0.496 (0.485–0.504) | 0.498 (0.490–0.504) |
| xsdir I20/R20 | 0.498 (0.488–0.508) | 0.495 (0.474–0.515) |

No daily signal in these large caps. Every CNN interval contains 0.5 (or lies below it), and the CNN's log loss is
worse than the training prior in every cell (−0.3% to −3.6%). No CNN decile long-short has |t| > 2.

HGB on daily features and the reversal/momentum rules are also null: one rule interval clears 0.5, xsdir I5/R5 for
HGB on the test years (0.505, 0.495–0.515). Accuracy (0.50–0.54) stays below the 55% that "always up" gets on `dir`.

Reading: the hypothesis "signal at daily, not at 5 minutes" is not supported on this universe and period. The image
signal is absent at both frequencies. JKX differ in three ways:
- universe: all CRSP stocks, with the effect strongest in small caps;
- period: 1993–2019, which includes pre-publication years;
- training size: millions of images.
A CRSP all-stock run would separate these.
