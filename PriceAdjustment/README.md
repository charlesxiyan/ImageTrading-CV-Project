# PriceAdjustment: split, dividend and share-volume adjustment

Step 3 of the data pipeline. It reads raw bars and CRSP factors and writes a new adjusted copy. The raw bars are never changed.
The policy and formulas are in `../DATA_PIPELINE.md`. Model inputs are re-based per window to the anchor
session (point in time), so only ratios between sessions are used.

| Script | What it does |
|---|---|
| `prepare_archive_inputs.py` | Turns an archive's CRSP CIZ tables into this step's inputs: factors, distributions and a manifest that points at the raw bars. |
| `adjust_dividend_bars.py` | Cash-dividend-inclusive prices and share-adjusted volume. |
| `validate_dividend_bars.py` | Re-checks every adjusted file against the raw bars and the formulas. |
| `adjust_bar_prices.py` | Split-only (`cfacpr`) adjustment; its helpers are shared by the scripts above. |

```bash
cd PriceAdjustment
V3=~/"Market Data/US Equity/v3"
for S in "" _spy; do
  python prepare_archive_inputs.py --archive "$V3/01_raw$S" --output "$V3/02_adjust_inputs$S"
  python adjust_dividend_bars.py --raw-root "$V3/02_adjust_inputs$S" --inputs "$V3/02_adjust_inputs$S" \
         --output "$V3/02_adjusted$S"
  python validate_dividend_bars.py --root "$V3/02_adjusted$S"
done
```
