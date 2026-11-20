# Daily Chart Replication

This folder tests whether the chart-image signal of Jiang, Kelly & Xiu (2023) is a low-frequency effect. It uses
the JKX daily setup (I5 and I20 images; 5- and 20-day labels; same CNN, training rule and 5-run ensemble), applied
to the same stocks and years as our 5-minute experiments. Only the frequency changes. The daily bars are built from
the 5-minute bars in `AllData & AllModel/alldata_bundle_v1.zip`.

- `Daily_JKX_Replication.ipynb` is the Colab notebook. It reads the bundle and the 5-minute reference run from
  `../AllData & AllModel`, and writes to `runs/<RUN_ID>/`.
- `daily_jkx.py` builds the daily bars, the JKX images and labels, trains the JKX CNN, the HGB and rule benchmarks,
  and runs the evaluation (AUC, accuracy, decile long-short) and the daily-vs-5-minute figure.
- `tools/make_daily_notebook.py` generates the notebook; `tools/smoke_daily_notebook.py` runs it end to end on
  synthetic data.

The parameter counts match JKX: I20 CNN 708,866 and I5 CNN 155,138.
