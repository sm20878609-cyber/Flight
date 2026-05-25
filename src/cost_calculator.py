"""
src/cost_calculator.py
成本計算模組：計算每個航班的真實總成本，包含行李費、選位費、餐點費、付款手續費與時間成本。
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_baggage_cost(df: pd.DataFrame, needs_baggage: bool, baggage_count: int) -> pd.Series:
    """
    計算托運行李費用。

    Args:
        df: 航班資料
        needs_baggage: 使用者是否需要托運行李
        baggage_count: 托運行李件數

    Returns:
        每個航班的行李費用 Series
    """
    if not needs_baggage or baggage_count == 0:
        return pd.Series(0.0, index=df.index)
    return df["baggage_fee"] * baggage_count


def calculate_seat_cost(df: pd.DataFrame, needs_seat: bool) -> pd.Series:
    """
    計算選位費用。

    Args:
        df: 航班資料
        needs_seat: 使用者是否需要選位

    Returns:
        每個航班的選位費用 Series
    """
    if not needs_seat:
        return pd.Series(0.0, index=df.index)
    return df["seat_selection_fee"].copy()


def calculate_meal_cost(df: pd.DataFrame, needs_meal: bool) -> pd.Series:
    """
    計算餐點費用。

    Args:
        df: 航班資料
        needs_meal: 使用者是否需要餐點

    Returns:
        每個航班的餐點費用 Series
    """
    if not needs_meal:
        return pd.Series(0.0, index=df.index)
    return df["meal_fee"].copy()


def calculate_time_cost(df: pd.DataFrame, hourly_value: float) -> pd.Series:
    """
    計算時間成本：飛行時間（小時）× 使用者每小時願意支付的金額。

    Args:
        df: 航班資料
        hourly_value: 使用者每節省 1 小時願意支付的金額

    Returns:
        每個航班的時間成本 Series
    """
    hours = df["duration_minutes"] / 60.0
    return hours * hourly_value


def calculate_all_costs(
    df: pd.DataFrame,
    needs_baggage: bool,
    baggage_count: int,
    needs_seat: bool,
    needs_meal: bool,
    hourly_value: float
) -> pd.DataFrame:
    """
    計算所有航班的完整成本，包含真實總成本與調整後總成本。

    Args:
        df: 原始航班 DataFrame
        needs_baggage: 是否需要托運行李
        baggage_count: 托運行李件數
        needs_seat: 是否需要選位
        needs_meal: 是否需要餐點
        hourly_value: 每小時時間成本（元）

    Returns:
        加入各費用欄位後的 DataFrame
    """
    result = df.copy()

    # 計算各項費用
    result["calculated_baggage_fee"] = calculate_baggage_cost(df, needs_baggage, baggage_count)
    result["calculated_seat_fee"] = calculate_seat_cost(df, needs_seat)
    result["calculated_meal_fee"] = calculate_meal_cost(df, needs_meal)
    result["time_cost"] = calculate_time_cost(df, hourly_value)

    # 付款手續費直接採用欄位值（廉航通常有額外手續費）
    result["calculated_payment_fee"] = result["payment_fee"].copy()

    # 真實總成本 = 基本票價 + 行李費 + 選位費 + 餐點費 + 付款手續費
    result["true_total_cost"] = (
        result["base_price"]
        + result["calculated_baggage_fee"]
        + result["calculated_seat_fee"]
        + result["calculated_meal_fee"]
        + result["calculated_payment_fee"]
    )

    # 調整後總成本 = 真實總成本 + 時間成本（用於 ML 推薦）
    result["adjusted_total_cost"] = result["true_total_cost"] + result["time_cost"]

    return result


def get_cost_summary(row: pd.Series) -> Dict[str, float]:
    """
    取得單一航班的成本摘要字典。

    Args:
        row: 航班資料列

    Returns:
        成本各項明細的字典
    """
    return {
        "基本票價": float(row.get("base_price", 0)),
        "行李費": float(row.get("calculated_baggage_fee", 0)),
        "選位費": float(row.get("calculated_seat_fee", 0)),
        "餐點費": float(row.get("calculated_meal_fee", 0)),
        "付款手續費": float(row.get("calculated_payment_fee", 0)),
        "時間成本": float(row.get("time_cost", 0)),
        "真實總成本": float(row.get("true_total_cost", 0)),
        "調整後總成本": float(row.get("adjusted_total_cost", 0)),
    }
