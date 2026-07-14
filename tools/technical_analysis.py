#!/usr/bin/env python3
"""Fetch adjusted OHLCV data and calculate common technical indicators."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


def chart_url(symbol: str, range_: str, interval: str) -> str:
    query = urlencode({"range": range_, "interval": interval, "events": "div,splits"})
    return (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(symbol, safe='')}?{query}"
    )


def fetch_chart(symbol: str, range_: str, interval: str) -> dict:
    url = chart_url(symbol, range_, interval)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        chart = json.loads(response.read().decode("utf-8"))["chart"]
    if chart.get("error"):
        raise ValueError(chart["error"].get("description", "market data request failed"))
    if not chart.get("result"):
        raise ValueError(f"no market data found for {symbol}")
    return chart["result"][0]


def item(values: list | None, index: int):
    return values[index] if values and index < len(values) else None


def market_timezone(meta: dict):
    try:
        return ZoneInfo(meta.get("exchangeTimezoneName", "UTC"))
    except Exception:
        return timezone.utc


def build_rows(result: dict, interval: str) -> tuple[list[dict], bool]:
    meta = result["meta"]
    timestamps = result.get("timestamp", [])
    quote_data = result.get("indicators", {}).get("quote", [{}])[0]
    adjusted = result.get("indicators", {}).get("adjclose", [{}])[0].get(
        "adjclose", []
    )
    exchange_timezone = market_timezone(meta)

    rows = []
    for index, timestamp in enumerate(timestamps):
        raw_close = item(quote_data.get("close"), index)
        raw_open = item(quote_data.get("open"), index)
        raw_high = item(quote_data.get("high"), index)
        raw_low = item(quote_data.get("low"), index)
        if None in (raw_open, raw_high, raw_low, raw_close) or raw_close == 0:
            continue
        adjusted_close = item(adjusted, index) or raw_close
        factor = adjusted_close / raw_close
        rows.append(
            {
                "timestamp": timestamp,
                "date": datetime.fromtimestamp(timestamp, timezone.utc)
                .astimezone(exchange_timezone)
                .date()
                .isoformat(),
                "open": raw_open * factor,
                "high": raw_high * factor,
                "low": raw_low * factor,
                "close": adjusted_close,
                "raw_close": raw_close,
                "adjustment_factor": factor,
                "volume": item(quote_data.get("volume"), index),
            }
        )

    dropped = False
    while rows and is_incomplete(rows[-1]["timestamp"], meta, interval):
        rows.pop()
        dropped = True
    return rows, dropped


def is_incomplete(timestamp: int, meta: dict, interval: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    if timestamp >= now:
        return True

    exchange_timezone = market_timezone(meta)
    latest = datetime.fromtimestamp(timestamp, exchange_timezone)
    current = datetime.fromtimestamp(now, exchange_timezone)
    if interval.endswith("wk"):
        return latest.isocalendar()[:2] == current.isocalendar()[:2]
    if interval.endswith("mo"):
        return (latest.year, latest.month) == (current.year, current.month)

    regular = meta.get("currentTradingPeriod", {}).get("regular", {})
    start, end = regular.get("start"), regular.get("end")
    if start and end and start <= timestamp < end:
        return now < end

    seconds = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
    }.get(interval)
    return bool(seconds and timestamp + seconds > now)


def sma(values: list[float], period: int) -> float | None:
    return statistics.fmean(values[-period:]) if len(values) >= period else None


def ema_series(values: list[float], period: int) -> list[float | None]:
    if len(values) < period:
        return [None] * len(values)
    result: list[float | None] = [None] * (period - 1)
    current = statistics.fmean(values[:period])
    result.append(current)
    alpha = 2 / (period + 1)
    for value in values[period:]:
        current = alpha * value + (1 - alpha) * current
        result.append(current)
    return result


def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) <= period:
        return None
    changes = [current - previous for previous, current in zip(values, values[1:])]
    gains = [max(change, 0) for change in changes]
    losses = [max(-change, 0) for change in changes]
    average_gain = statistics.fmean(gains[:period])
    average_loss = statistics.fmean(losses[:period])
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period
    if average_loss == 0:
        return 100.0 if average_gain else 50.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def macd(values: list[float]) -> tuple[float | None, float | None, float | None]:
    fast = ema_series(values, 12)
    slow = ema_series(values, 26)
    line = [
        fast_value - slow_value
        for fast_value, slow_value in zip(fast, slow)
        if fast_value is not None and slow_value is not None
    ]
    signal = ema_series(line, 9)
    if not line or not signal or signal[-1] is None:
        return None, None, None
    return line[-1], signal[-1], line[-1] - signal[-1]


def atr(rows: list[dict], period: int = 14) -> float | None:
    if len(rows) < period:
        return None
    true_ranges = []
    previous_close = None
    for row in rows:
        candidates = [row["high"] - row["low"]]
        if previous_close is not None:
            candidates.extend(
                [abs(row["high"] - previous_close), abs(row["low"] - previous_close)]
            )
        true_ranges.append(max(candidates))
        previous_close = row["close"]
    current = statistics.fmean(true_ranges[:period])
    for value in true_ranges[period:]:
        current = (current * (period - 1) + value) / period
    return current


def percent_change(values: list[float], bars: int) -> float | None:
    if len(values) <= bars or values[-bars - 1] == 0:
        return None
    return (values[-1] / values[-bars - 1] - 1) * 100


def rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def annual_bar_count(interval: str, instrument_type: str | None) -> int | None:
    if interval == "1d":
        return 365 if instrument_type == "CRYPTOCURRENCY" else 252
    return {"5d": 52, "1wk": 52, "1mo": 12, "3mo": 4}.get(interval)


def level(rows: list[dict], period: int, field: str, function) -> float | None:
    return function(row[field] for row in rows[-period:]) if len(rows) >= period else None


def analyze(result: dict, rows: list[dict], range_: str, interval: str, dropped: bool) -> dict:
    if len(rows) < 2:
        raise ValueError("fewer than two completed price bars were returned")

    meta = result["meta"]
    closes = [row["close"] for row in rows]
    latest = rows[-1]
    calendar_days_behind = None
    if not interval.endswith(("wk", "mo")):
        calendar_days_behind = (
            datetime.now(market_timezone(meta)).date()
            - datetime.fromisoformat(latest["date"]).date()
        ).days
    averages = {period: sma(closes, period) for period in (20, 50, 200)}
    macd_line, macd_signal, macd_histogram = macd(closes)
    atr_14 = atr(rows)

    if all(averages.values()) and latest["close"] > averages[20] > averages[50] > averages[200]:
        trend = "bullish"
    elif all(averages.values()) and latest["close"] < averages[20] < averages[50] < averages[200]:
        trend = "bearish"
    else:
        trend = "mixed"

    band_values = closes[-20:] if len(closes) >= 20 else []
    band_middle = statistics.fmean(band_values) if band_values else None
    band_deviation = statistics.pstdev(band_values) if band_values else None
    volumes = [row["volume"] for row in rows[-20:] if row["volume"] and row["volume"] > 0]
    year_bars = annual_bar_count(interval, meta.get("instrumentType"))
    lookback_52w = rows[-year_bars:] if year_bars and len(rows) >= year_bars else []
    range_low = min((row["low"] for row in lookback_52w), default=None)
    range_high = max((row["high"] for row in lookback_52w), default=None)
    range_position = (
        (latest["close"] - range_low) / (range_high - range_low) * 100
        if range_low is not None and range_high is not None and range_high != range_low
        else None
    )

    limitations = []
    if len(rows) < 200:
        limitations.append("sma_200_unavailable")
    if not volumes or not latest["volume"]:
        limitations.append("volume_unavailable")
    if not lookback_52w:
        limitations.append("range_52w_unavailable")

    return {
        "identity": {
            "symbol": meta.get("symbol"),
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
            "instrument_type": meta.get("instrumentType"),
            "currency": meta.get("currency"),
            "timezone": meta.get("exchangeTimezoneName"),
        },
        "data": {
            "primary_source": "Yahoo Finance Chart API",
            "source_url": chart_url(meta.get("symbol", ""), range_, interval),
            "range": range_,
            "interval": interval,
            "adjustment_method": "OHLC scaled by Yahoo adjusted-close/raw-close factor",
            "completed_bars": len(rows),
            "start": rows[0]["date"],
            "cutoff": latest["date"],
            "bar_label_semantics": (
                "period_start" if interval.endswith(("wk", "mo")) else "trading_date"
            ),
            "calendar_days_behind": calendar_days_behind,
            "partial_bar_dropped": dropped,
            "limitations": limitations,
        },
        "price": {
            "close": rounded(latest["close"]),
            "raw_close": rounded(latest["raw_close"]),
            "adjusted_close": rounded(latest["close"]),
            "adjustment_factor": rounded(latest["adjustment_factor"]),
            "return_1_bar_pct": rounded(percent_change(closes, 1)),
            "return_5_bar_pct": rounded(percent_change(closes, 5)),
            "return_20_bar_pct": rounded(percent_change(closes, 20)),
            "return_60_bar_pct": rounded(percent_change(closes, 60)),
            "return_252_bar_pct": rounded(percent_change(closes, 252)),
            "range_52w_bars": year_bars,
            "range_52w_low": rounded(range_low),
            "range_52w_high": rounded(range_high),
            "range_52w_position_pct": rounded(range_position),
        },
        "trend": {
            "classification": trend,
            "sma_20": rounded(averages[20]),
            "sma_50": rounded(averages[50]),
            "sma_200": rounded(averages[200]),
        },
        "momentum": {
            "rsi_14": rounded(rsi(closes)),
            "macd_12_26": rounded(macd_line),
            "macd_signal_9": rounded(macd_signal),
            "macd_histogram": rounded(macd_histogram),
        },
        "volatility": {
            "atr_14": rounded(atr_14),
            "atr_14_pct": rounded(atr_14 / latest["close"] * 100) if atr_14 else None,
            "bollinger_middle_20": rounded(band_middle),
            "bollinger_upper_20_2": rounded(
                band_middle + 2 * band_deviation
                if band_middle is not None and band_deviation is not None
                else None
            ),
            "bollinger_lower_20_2": rounded(
                band_middle - 2 * band_deviation
                if band_middle is not None and band_deviation is not None
                else None
            ),
        },
        "volume": {
            "latest": latest["volume"] if volumes else None,
            "average_20": rounded(statistics.fmean(volumes)) if volumes else None,
            "ratio_to_average_20": rounded(
                latest["volume"] / statistics.fmean(volumes)
                if volumes and latest["volume"] and statistics.fmean(volumes)
                else None
            ),
        },
        "levels": {
            "low_20": rounded(level(rows, 20, "low", min)),
            "high_20": rounded(level(rows, 20, "high", max)),
            "low_60": rounded(level(rows, 60, "low", min)),
            "high_60": rounded(level(rows, 60, "high", max)),
        },
    }


def self_check() -> None:
    assert sma([1.0, 2.0, 3.0], 2) == 2.5
    assert ema_series([1.0, 2.0, 3.0], 2)[-1] == 2.5
    assert rsi([float(value) for value in range(1, 17)]) == 100.0
    assert annual_bar_count("1wk", "EQUITY") == 52
    hk_week_start = datetime(2026, 7, 12, 16, tzinfo=timezone.utc).timestamp()
    tuesday = datetime(2026, 7, 14, 3, tzinfo=timezone.utc).timestamp()
    assert is_incomplete(
        hk_week_start,
        {"exchangeTimezoneName": "Asia/Hong_Kong"},
        "1wk",
        now=tuesday,
    )
    rows = [
        {"high": value + 1.0, "low": value - 1.0, "close": value}
        for value in range(1, 17)
    ]
    assert atr(rows) == 2.0
    print("technical_analysis self-check: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", help="Yahoo Finance symbol, such as AAPL or 0700.HK")
    parser.add_argument("--range", dest="range_", default="2y", help="history range (default: 2y)")
    parser.add_argument("--interval", default="1d", help="bar interval (default: 1d)")
    parser.add_argument("--self-check", action="store_true", help="run deterministic indicator checks")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.symbol:
        parser.error("symbol is required unless --self-check is used")

    try:
        result = fetch_chart(args.symbol, args.range_, args.interval)
        rows, dropped = build_rows(result, args.interval)
        print(
            json.dumps(
                analyze(result, rows, args.range_, args.interval, dropped),
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as error:
        print(f"technical_analysis: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
