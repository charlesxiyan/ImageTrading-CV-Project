Final model `ewsd_tab_rv` (Step 6b run final_20261125_222625). Sharpe ratios annualised with the decision times traded.

**vol_h1, validation 2023**
- IC final CNN (ewsd_tab_rv): 0.117 (ICIR 1.46, t 57.8, positive 92%); vs signed return +0.003 (t 0.5)
- IC HGB 43 (38 + multi-day RV): 0.124 (ICIR 1.52, t 60.1, positive 93%); vs signed return +0.004 (t 0.8)
- long-short final CNN (ewsd_tab_rv): Sharpe -0.11 gross, -8.42 at 1 bp; max drawdown -517 bp
- long-short HGB 43 (38 + multi-day RV): Sharpe -0.18 gross, -8.00 at 1 bp; max drawdown -600 bp
- long-short no forecast (training median |r|): Sharpe -0.18 gross, -8.64 at 1 bp; max drawdown -586 bp
- equal weight: Sharpe +0.38, ann vol 0.066, max drawdown -624 bp, vol of daily vol 0.89
- inverse median vol (no forecast): Sharpe +0.38, ann vol 0.062, max drawdown -569 bp, vol of daily vol 0.88
- risk parity | final CNN: Sharpe +0.38, ann vol 0.059, max drawdown -544 bp, vol of daily vol 0.86
- volatility target | final CNN: Sharpe -0.06, ann vol 0.028, max drawdown -387 bp, vol of daily vol 0.44

**vol_h1, test 2024-25**
- IC final CNN (ewsd_tab_rv): 0.129 (ICIR 1.55, t 86.6, positive 94%); vs signed return +0.012 (t 3.5)
- IC HGB 43 (38 + multi-day RV): 0.137 (ICIR 1.61, t 90.3, positive 94%); vs signed return +0.010 (t 3.0)
- long-short final CNN (ewsd_tab_rv): Sharpe -1.65 gross, -8.02 at 1 bp; max drawdown -4191 bp
- long-short HGB 43 (38 + multi-day RV): Sharpe -1.30 gross, -7.41 at 1 bp; max drawdown -4018 bp
- long-short no forecast (training median |r|): Sharpe -1.24 gross, -7.78 at 1 bp; max drawdown -3964 bp
- equal weight: Sharpe +1.12, ann vol 0.076, max drawdown -1101 bp, vol of daily vol 1.18
- inverse median vol (no forecast): Sharpe +1.07, ann vol 0.071, max drawdown -1020 bp, vol of daily vol 1.15
- risk parity | final CNN: Sharpe +1.00, ann vol 0.067, max drawdown -1015 bp, vol of daily vol 1.12
- volatility target | final CNN: Sharpe -0.11, ann vol 0.025, max drawdown -1497 bp, vol of daily vol 0.48

**vol_h12, validation 2023**
- IC final CNN (ewsd_tab_rv): 0.103 (ICIR 1.19, t 47.1, positive 88%); vs signed return +0.007 (t 1.5)
- IC HGB 43 (38 + multi-day RV): 0.113 (ICIR 1.23, t 48.4, positive 90%); vs signed return +0.009 (t 1.7)
- long-short final CNN (ewsd_tab_rv): Sharpe -0.82 gross, -4.85 at 1 bp; max drawdown -1736 bp
- long-short HGB 43 (38 + multi-day RV): Sharpe -1.05 gross, -4.68 at 1 bp; max drawdown -2379 bp
- long-short no forecast (training median |r|): Sharpe -1.29 gross, -5.33 at 1 bp; max drawdown -2129 bp
- equal weight: Sharpe +1.19, ann vol 0.144, max drawdown -1429 bp, vol of daily vol 0.53
- inverse median vol (no forecast): Sharpe +1.15, ann vol 0.136, max drawdown -1358 bp, vol of daily vol 0.53
- risk parity | final CNN: Sharpe +1.17, ann vol 0.132, max drawdown -1340 bp, vol of daily vol 0.52
- volatility target | final CNN: Sharpe +1.00, ann vol 0.098, max drawdown -1200 bp, vol of daily vol 0.44

**vol_h12, test 2024-25**
- IC final CNN (ewsd_tab_rv): 0.114 (ICIR 1.31, t 73.5, positive 92%); vs signed return -0.001 (t -0.3)
- IC HGB 43 (38 + multi-day RV): 0.124 (ICIR 1.34, t 75.0, positive 92%); vs signed return -0.001 (t -0.3)
- long-short final CNN (ewsd_tab_rv): Sharpe -0.22 gross, -4.02 at 1 bp; max drawdown -3395 bp
- long-short HGB 43 (38 + multi-day RV): Sharpe -0.01 gross, -3.47 at 1 bp; max drawdown -3744 bp
- long-short no forecast (training median |r|): Sharpe +0.02 gross, -3.69 at 1 bp; max drawdown -3127 bp
- equal weight: Sharpe +0.95, ann vol 0.145, max drawdown -1909 bp, vol of daily vol 0.77
- inverse median vol (no forecast): Sharpe +1.00, ann vol 0.136, max drawdown -1731 bp, vol of daily vol 0.75
- risk parity | final CNN: Sharpe +1.01, ann vol 0.133, max drawdown -1641 bp, vol of daily vol 0.75
- volatility target | final CNN: Sharpe +0.56, ann vol 0.086, max drawdown -1019 bp, vol of daily vol 0.48

