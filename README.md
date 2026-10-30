# ImageTrading-CV-Project

## 美股动态股票池

`build_dynamic_universe.py` 构建 **S&P 500 ∪ NASDAQ 100 ∪ Dow Jones Industrial Average 30** 的每日证券池。输入起止日期（均包含），输出该日历史有效的 ticker 列表。三个指数重叠的证券仅保留一次，不把 500+100+30 简单相加；同一公司的不同股份类别保留为不同证券。

### 安装与运行

Python 3.9+，在你们有 WRDS 权限的本机环境或 WRDS JupyterHub 运行：

```bash
python -m pip install wrds
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --inspect
python build_dynamic_universe.py --username YOUR_WRDS_USERNAME --start 2020-01-01 --end 2020-12-31
```

身份验证交给 WRDS 官方 Python 客户端完成，不在源码中保存密码。`--inspect` 只读查询元数据：列出候选表字段、NASDAQ/道指指数名称与 ID，以及可见的 TAQ 等相关表。报告只说明可见性，不能证明完整订阅或数据质量；脚本实际读取时还会检查权限。

指数名称不能唯一识别时，根据 `--inspect` 中的**准确指数名称**选择其 `index_id`（保留前导零），再运行：

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --ndx100-id VERIFIED_NDX_GVKEYX --djia30-id VERIFIED_DJIA_GVKEYX
```

以上 ID 是占位符，不能直接使用。不要选 NASDAQ Composite、NASDAQ 金融指数或道指运输指数。脚本不猜测固定 ID，也不拿 QQQ、DIA 的持仓近似替代成分股。

默认包含**每个自然日**（周末沿用该自然日有效的成员关系）。训练通常使用交易日模式：

```bash
python build_dynamic_universe.py --start 2020-01-01 --end 2020-12-31 --days trading
```

交易日来自 CRSP `dsi` 中实际出现的日期，不使用简单的周一到周五日历；请求超出可用日历范围时会拒绝执行，包括结束日已超过数据库最新日期的情况。

### 输出与 Python 使用

默认写入本项目的 `data/universe/`；重复运行会覆盖同名输出，建议每个研究区间使用独立的 `--output` 目录。

- `daily_tickers.json`：`{"2020-01-02": ["AAPL", "..."], ...}`，每天一个去重、排序的列表。
- `daily_universe.csv`：`date,permno,ticker,indices`，每个日期、PERMNO 一行；`indices` 用 `|` 连接所属指数。
- `manifest.json`：来源表、指数 ID、请求区间、生成时间，以及每天各指数和并集的证券数，供人工审计。

```python
import json
from pathlib import Path

# 在项目根目录运行
pools = json.loads(Path('data/universe/daily_tickers.json').read_text())
tickers = pools['2020-01-02']
```

所有默认生成路径位于 `ImageTrading-CV-Project`；原 `ImageTrading` 项目无需修改。

### 数据来源与约束

1. **S&P 500**：优先 `crsp_a_indexes.dsp500list`，兼容 `crsp.dsp500list`；要求 `permno/start/ending`。
2. **NASDAQ 100、DJIA 30**：尝试 `comp.idxcst_his`，兼容 `comp_na_daily_all.idxcst_his`；使用 `gvkey/iid/gvkeyx/from/thru`。指数名称由 `idx_index` 解析，要求存在 `gvkeyx`（或 `gvkey`）和 `conm`（或 `conmx`）。**这些只是可检查的适配入口，不保证你们账号包含这两个指数的历史记录。**
3. Compustat 成员按 `gvkey+iid` 经历史 CCM `ccmxpf_linktable` 匹配 PERMNO，限定证券级链接、有效日期和 `LC/LU/LS` 链接类型。多重映射报错，不能随意挑一条或用公司当前主证券替代。
4. PERMNO 按 CRSP `dsenames` / `msenames` 的 `namedt/nameendt` 匹配**当时 ticker**，不使用当前 `comp.security.tic` 回填历史。标点、股份类别不自动转换。
5. 所有日期区间均采用闭区间；NULL 结束日期作为持续有效，NULL 成分开始日期不回填。对于另一套端点约定或字段结构，必须先核实字典再添加适配器。

可用 `--sp500-table`、`--members-table`、`--headers-table`、`--links-table`、`--names-table`、`--calendar-table` 覆盖表名。只允许 `schema.table` 格式，并校验必需字段。如果仅有新版 CRSP CIZ 表，脚本会提示字段/表不匹配，**目前没有实现 CIZ 自动转换**；应依据账号的实际字典添加适配器，不能仅重命名表。

脚本在写入前检查每个日期三个指数都非空、每个成员都能映射历史 ticker、映射无歧义。**非空检查不能证明源数据完整或新鲜**：历史记录可能有修订、供应商更新可能滞后，NULL 结束日也可能是陈旧记录。必须核实 WRDS 数据更新日期、各指数历史覆盖范围，并检查 `manifest.json` 的人数/突变。人数不硬编码为 500/100/30，因为指数公司数与证券数并不总相等。

WRDS 账号不等于拥有全部底层库权限。WRDS 明确说明历史 S&P 500 成分查询需要 CRSP 订阅。Compustat 的 S&P 指数成分授权有过调整，因此不能假定 `idxcst_his` 能提供所有指数。缺失任一指数、某天无历史记录、链接或 ticker 缺失时，脚本拒绝生成本次新输出；需先解决订阅/覆盖问题，不会用当前成分回填，也不会静默降级成仅 S&P 500。若输出目录已有旧文件，失败不会删除旧文件，应检查 manifest，避免把旧结果误认为本次输出。

这里得到的是**按历史生效日重建的成员池**，并非原始公告发布时点的数据库。回测仍需遵守信号产生/下单时点和公告可得性。成员身份不代表当天可交易；停牌、价格可得性和流动性筛选应作为下一层处理，不能用未来退市信息筛掉过去成员。

### 与高频数据的关系

这一步不下载高频行情，不依赖你们尚未确认的 TAQ 字段。后续应先确认是否订阅 NYSE TAQ、可用年份、逐笔成交/报价还是聚合 bar，以及时间戳、交易条件、撤销更正等字段。**CRSP 历史 ticker 不是可以直接用于任意 TAQ 表的通用 join key**；需要按日期和股份类别映射 TAQ symbol root/suffix 等实际字段。保留 PERMNO 是为后续证券识别与审计，不代表 TAQ 天然包含 PERMNO。

### 验证情况

```bash
python -m unittest discover -s tests -v
```

离线测试覆盖去重、跨指数归属、历史改名、周末日期保留、缺失指数/映射、链接歧义、ticker 冲突、输出及 SQL 标识符校验。**尚未连接你们的 WRDS 账号，未核实实际表权限、字段、指数 ID 和历史覆盖，也未执行真实下载。** 首次运行应先 `--inspect`，再取一个包含已知调仓/改名日期的小区间，对照 WRDS 页面核验。

参考：
- [WRDS：历史 S&P 500 成分下载及 CRSP 订阅要求](https://wrds-www.wharton.upenn.edu/pages/grid-items/downloading-the-sp-500-constituents/)
- [WRDS：S&P 500 历史成员变化示例](https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/)
- [CRSP：官方 CCM 文档（通过 WRDS）](https://wrds-www.wharton.upenn.edu/documents/415/CRSPAccess_Software_Guide_CCM_part.pdf)
