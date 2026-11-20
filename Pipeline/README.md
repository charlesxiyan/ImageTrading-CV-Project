# Pipeline: training windows and the Colab bundle

Step 4 of the data pipeline. `us_pipeline` builds the audited 78-bar windows and the train, validation
and test splits from adjusted bars. See `US_PIPELINE.md`. Configs are in `configs/`:
- `us_5min_2016_2025_v3.json` is the current 2016-2025 archive (train 2016-2022, validation 2023, test 2024-2025);
- `us_5min_2016_2025.json` is the v1 archive behind the results so far;
- `us_5min.json` and `us_5min_v2.json` are the 2023 H1 demo.

```bash
cd Pipeline
pip install -r requirements-pipeline.txt
python -m us_pipeline prepare --config configs/us_5min_2016_2025_v3.json
python -m us_pipeline audit   --config configs/us_5min_2016_2025_v3.json
cd ..
V3=~/"Market Data/US Equity/v3"
python "AllData & AllModel/tools/build_alldata_bundle.py" --prepared "$V3/03_prepared/prepared" \
       --market "$V3/02_adjusted_spy" --output "$V3/04_bundle/alldata_bundle_v3.zip"
```

The bundle builder stays in `AllData & AllModel/tools/` because it packages that folder's `chartcnn` code with the data.
