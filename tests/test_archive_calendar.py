"""Calendar regressions for the multi-year archive."""
import paths  # noqa: F401  (step folders on sys.path)
import unittest

import pandas as pd
from download_us_equity_archive import schedule, archive_grid, select_sessions


class ArchiveCalendarTests(unittest.TestCase):
    def test_half_day_and_full_day(self):
        result = schedule('2016-11-24', '2016-11-28')
        self.assertEqual(result.date.tolist(), ['2016-11-25', '2016-11-28'])
        self.assertEqual(result.minutes.tolist(), [210, 390])
        self.assertEqual(result.market_close.dt.tz_convert('America/New_York').dt.hour.tolist(), [13, 16])

    def test_extraordinary_closures(self):
        for day in ['2018-12-05', '2025-01-09']:
            self.assertTrue(schedule(day, day).empty)

    def test_half_day_grid_keeps_unmapped_active_members_missing(self):
        day = '2016-11-25'
        universe = pd.DataFrame([
            dict(date=day,permno=1,ticker='A',indices='SP500',identifier_status='date_effective'),
            dict(date=day,permno=2,ticker='B',indices='SP500',identifier_status='date_effective')])
        mapping = pd.DataFrame([dict(date=day,permno=1,ticker='A',sym_root='A',sym_suffix='',match_lvl=1)])
        bars = pd.DataFrame([dict(sym_root='A',sym_suffix='',bucket=0,open=10.,high=12.,low=9.,close=11.,volume=100,trade_count=2,vwap=10.5)])
        result, quality = archive_grid(day,universe,mapping,bars,5,210)
        self.assertEqual(len(result),84)
        self.assertEqual(result.bar_end.max().strftime('%H:%M'),'13:00')
        missing = result[result.permno.eq(2)]
        self.assertTrue(missing.close.isna().all())
        self.assertTrue(missing.status.eq('no_symbol_mapping').all())
        self.assertEqual(quality['unmapped_members'],1)

    def test_dst_changes_utc_not_local_open(self):
        result = schedule('2024-03-08', '2024-03-11')
        self.assertEqual(result.market_open.dt.hour.tolist(), [14, 13])
        self.assertEqual(result.minutes.tolist(), [390, 390])

    def test_select_sessions(self):
        cal = schedule('2016-11-21', '2016-11-30')
        self.assertEqual(len(select_sessions(cal)), len(cal))
        self.assertEqual(select_sessions(cal, limit_days=2).date.tolist(), cal.date[:2].tolist())
        self.assertEqual(select_sessions(cal, days='2016-11-28, 2016-11-21').date.tolist(),
                         ['2016-11-21', '2016-11-28'])
        with self.assertRaises(ValueError):
            select_sessions(cal, days='2016-11-24')   # Thanksgiving: not a session


if __name__ == '__main__':
    unittest.main()
