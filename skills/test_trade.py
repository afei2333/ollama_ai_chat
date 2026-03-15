"""
定投回测脚本 (DCA Backtest)
支持两种定投方式：
  1. fixed_amount  - 每期固定金额买入
  2. fixed_shares  - 每期固定份额买入

数据来源：直接调用 get_stock_k_data(symbol, start, end) 获取 DataFrame，
列包含：date, open, close, high, low, volume, turnover,
        amplitude, pct_change, change, turnover_rate

直接运行：
  python dca_backtest.py
"""

import pandas as pd
import numpy as np
from skills.get_stock_k_data import get_stock_k_data

# ─────────────────────────────────────────────
# 数据预处理（对 DataFrame 做统一整理）
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """对 get_stock_k_data 返回的 DataFrame 做标准化处理。"""
    df = df.copy()
    # 统一列名
    df.columns = df.columns.str.strip().str.lower()

    if "date" not in df.columns:
        raise ValueError("DataFrame 中未找到 'date' 列。")
    if "close" not in df.columns:
        raise ValueError("DataFrame 中未找到 'close' 列。")

    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 定投日期筛选
# ─────────────────────────────────────────────

def get_invest_dates(df: pd.DataFrame, freq: str) -> pd.Series:
    """
    根据频率从数据中选取定投交易日（取实际存在的交易日）。

    freq:
      daily   - 每个交易日
      weekly  - 每周第一个交易日
      monthly - 每月第一个交易日
    """
    freq = freq.lower()
    if freq == "daily":
        return df["date"]

    if freq == "weekly":
        df2 = df.copy()
        iso = df2["date"].dt.isocalendar()
        df2["_yw"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
        idx = df2.groupby("_yw")["date"].idxmin()
        return df.loc[sorted(idx.values), "date"]

    if freq == "monthly":
        df2 = df.copy()
        df2["_ym"] = df2["date"].dt.to_period("M")
        idx = df2.groupby("_ym")["date"].idxmin()
        return df.loc[sorted(idx.values), "date"]

    raise ValueError(f"不支持的频率: {freq}，请使用 daily / weekly / monthly")


# ─────────────────────────────────────────────
# 核心回测
# ─────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    mode: str,
    amount: float = None,
    shares: float = None,
    freq: str = "monthly",
) -> dict:
    """
    执行定投回测。

    Parameters
    ----------
    df      : 价格数据 DataFrame
    mode    : 'fixed_amount' | 'fixed_shares'
    amount  : 每期投入金额（fixed_amount 模式）
    shares  : 每期买入份额（fixed_shares 模式）
    freq    : 'daily' | 'weekly' | 'monthly'

    Returns
    -------
    dict 包含回测汇总指标及每期明细 DataFrame
    """
    if mode == "fixed_amount" and (amount is None or amount <= 0):
        raise ValueError("fixed_amount 模式需要提供正数 --amount 参数")
    if mode == "fixed_shares" and (shares is None or shares <= 0):
        raise ValueError("fixed_shares 模式需要提供正数 --shares 参数")

    df = prepare_data(df)
    invest_dates = get_invest_dates(df, freq)
    price_map = dict(zip(df["date"], df["close"]))

    records = []
    total_cost   = 0.0
    total_shares = 0.0

    for d in invest_dates:
        price = price_map.get(d)
        if price is None or price <= 0:
            continue

        if mode == "fixed_amount":
            invested = amount
            bought   = amount / price
        else:  # fixed_shares
            bought   = shares
            invested = shares * price

        total_cost   += invested
        total_shares += bought

        avg_cost   = total_cost / total_shares
        mkt_value  = total_shares * price
        unrealized = mkt_value - total_cost
        pnl_pct    = unrealized / total_cost * 100

        records.append({
            "date":            d,
            "price":           round(price, 4),
            "invested":        round(invested, 4),
            "bought_shares":   round(bought, 6),
            "total_cost":      round(total_cost, 4),
            "total_shares":    round(total_shares, 6),
            "avg_cost":        round(avg_cost, 4),
            "market_value":    round(mkt_value, 4),
            "unrealized_pnl":  round(unrealized, 4),
            "pnl_pct(%)":      round(pnl_pct, 4),
        })

    if not records:
        raise ValueError("没有找到有效的定投记录，请检查数据和参数。")

    detail_df = pd.DataFrame(records)

    # ── 汇总指标 ────────────────────────────────
    final        = records[-1]
    final_value  = final["market_value"]
    total_invest = final["total_cost"]
    avg_cost     = final["avg_cost"]

    total_return = (final_value - total_invest) / total_invest * 100

    start_date = records[0]["date"]
    end_date   = records[-1]["date"]
    days       = (end_date - start_date).days
    years      = max(days / 365.25, 1e-9)
    annualized_return = ((final_value / total_invest) ** (1 / years) - 1) * 100

    # 最大回撤（基于市值高水位）
    mv     = detail_df["market_value"].values
    peak   = np.maximum.accumulate(mv)
    dd     = (mv - peak) / peak * 100
    max_drawdown = float(dd.min())

    # 最大回撤（基于收益率高水位）
    pnl_arr   = detail_df["pnl_pct(%)"].values
    peak_pnl  = np.maximum.accumulate(pnl_arr)
    dd_pnl    = pnl_arr - peak_pnl
    max_drawdown_pnl = float(dd_pnl.min())

    return {
        "mode":                    mode,
        "freq":                    freq,
        "start_date":              start_date.strftime("%Y-%m-%d"),
        "end_date":                end_date.strftime("%Y-%m-%d"),
        "invest_periods":          len(records),
        "total_invest":            round(total_invest, 4),
        "final_value":             round(final_value, 4),
        "total_return(%)":         round(total_return, 4),
        "annualized_return(%)":    round(annualized_return, 4),
        "max_drawdown(%)":         round(max_drawdown, 4),
        "max_drawdown_pnl(%)":     round(max_drawdown_pnl, 4),
        "avg_cost":                round(avg_cost, 4),
        "total_shares":            round(final["total_shares"], 6),
        "detail":                  detail_df,
    }


# ─────────────────────────────────────────────
# 打印结果
# ─────────────────────────────────────────────

def print_result(result: dict):
    sep = "═" * 52
    mode_zh = "固定金额" if result["mode"] == "fixed_amount" else "固定份额"
    freq_zh = {"daily": "每日", "weekly": "每周", "monthly": "每月"}.get(result["freq"], result["freq"])

    print(f"\n{sep}")
    print("    📊  定投回测结果汇总")
    print(sep)
    print(f"  定投模式     : {mode_zh}  ({result['mode']})")
    print(f"  定投频率     : {freq_zh}")
    print(f"  回测区间     : {result['start_date']}  →  {result['end_date']}")
    print(f"  定投期数     : {result['invest_periods']} 期")
    print(sep)
    print(f"  累计投入     : {result['total_invest']:>15,.4f}")
    print(f"  最终市值     : {result['final_value']:>15,.4f}")
    print(f"  总收益率     : {result['total_return(%)']:>+14.2f}%")
    print(f"  年化收益率   : {result['annualized_return(%)']:>+14.2f}%")
    print(f"  最大回撤     : {result['max_drawdown(%)']:>14.2f}%   (基于市值高水位)")
    print(f"  最大回撤     : {result['max_drawdown_pnl(%)']:>14.2f}%   (基于收益率高水位)")
    print(f"  平均持仓成本 : {result['avg_cost']:>15,.4f}")
    print(f"  总持有份额   : {result['total_shares']:>15,.4f}")
    print(sep)


if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────
    # 【修改这里】配置你的回测参数
    # ──────────────────────────────────────────────────────────────
    SYMBOL     = "159552"   # 标的代码，如 sh600036、sz000001
    START_DATE = "2022-12-01" # 回测开始日期
    END_DATE   = "2026-03-14" # 回测结束日期

    # 定投模式: "fixed_amount"（固定金额）或 "fixed_shares"（固定份额）
    MODE   = "fixed_amount"
    AMOUNT = 1000    # 每期投入金额（MODE = fixed_amount 时生效）
    SHARES = 100     # 每期买入份额（MODE = fixed_shares 时生效）

    # 定投频率: "daily" | "weekly" | "monthly"
    FREQ = "monthly"

    # 明细输出路径（None 则不保存）
    OUTPUT_CSV = f"{SYMBOL}_dca_detail.csv"
    # ──────────────────────────────────────────────────────────────

    # 1. 获取数据
    print(f"\n  📡 获取数据: {SYMBOL}  {START_DATE} → {END_DATE}")
    df_raw = get_stock_k_data(SYMBOL, start_date=START_DATE, end_date=END_DATE)
    print(f"  数据行数: {len(df_raw)}")

    # 2. 执行回测（fixed_amount 示例）
    result = run_backtest(
        df=df_raw,
        mode=MODE,
        amount=AMOUNT,   # fixed_amount 模式使用
        shares=SHARES,   # fixed_shares 模式使用
        freq=FREQ,
    )
    print_result(result)

    # 3. 保存明细
    if OUTPUT_CSV:
        result["detail"].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"  📁 每期明细已保存至: {OUTPUT_CSV}\n")

    # ──────────────────────────────────────────────────────────────
    # 对比示例：同一数据跑固定份额模式
    # ──────────────────────────────────────────────────────────────
    result2 = run_backtest(
        df=df_raw,
        mode="fixed_shares",
        shares=SHARES,
        freq=FREQ,
    )
    print_result(result2)