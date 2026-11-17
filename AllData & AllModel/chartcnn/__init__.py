"""chartcnn (AllData & AllModel copy): chart CNNs, 1D CNNs and HGB on 2016-2025 5-minute bars, with SPY."""
from .data import Store, build_split, ChartDataset, SeriesDataset, make_dataset
from .models import ChartCNN, SeriesCNN1D, INPUT_KIND, USES_MARKET, build_model, parameter_table, print_parameter_count

__version__ = "2026-11-16.1"   # bump on every change; printed so a stale import is easy to spot
print(f"chartcnn {__version__} loaded from {__file__}")
