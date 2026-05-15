"""Open-Meteo API fetch and eging condition scoring."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, datetime

import pandas as pd
import streamlit as st


def tide_score_from_tide_range(tide_range_m: float) -> tuple[float, str]:
    """
    その日の海面高度（潮汐込みモデル）から求めた潮差に基づき潮スコアを返す。
    tide_range_m は同日の hourly sea level の最大値と最小値の差（メートル）。
    """
    ref_range_m = 1.85
    normalized = max(0.0, min(1.0, tide_range_m / ref_range_m))
    score = 45.0 + (normalized * 55.0)

    if normalized >= 0.82:
        tide_type = "大潮寄り"
    elif normalized >= 0.62:
        tide_type = "中潮寄り"
    elif normalized >= 0.42:
        tide_type = "小潮寄り"
    else:
        tide_type = "長潮/若潮寄り"
    return score, tide_type


def get_rank(total_score: float) -> str:
    """Convert total score (0-100) into rank label."""
    if total_score >= 85:
        return "S"
    if total_score >= 73:
        return "A"
    if total_score >= 60:
        return "B"
    if total_score >= 48:
        return "C"
    return "D"


def rank_color(rank: str) -> str:
    """Return marker color for each rank."""
    return {
        "S": "red",
        "A": "orange",
        "B": "green",
        "C": "blue",
        "D": "gray",
    }.get(rank, "blue")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_hourly(  # pylint: disable=too-many-locals
    target_coords: list[float], target_day: date
) -> pd.DataFrame:
    """Fetch one day hourly weather/marine values from Open-Meteo."""
    lat, lon = target_coords
    day_text = target_day.isoformat()
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "start_date": day_text,
        "end_date": day_text,
        "hourly": "wind_speed_10m,pressure_msl",
        "windspeed_unit": "ms",
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "start_date": day_text,
        "end_date": day_text,
        "hourly": "wave_height,sea_surface_temperature,sea_level_height_msl",
        "cell_selection": "sea",
    }

    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"{urllib.parse.urlencode(weather_params)}"
    )
    marine_url = (
        "https://marine-api.open-meteo.com/v1/marine?"
        f"{urllib.parse.urlencode(marine_params)}"
    )
    with urllib.request.urlopen(weather_url, timeout=15) as response:
        weather_data = json.loads(response.read().decode("utf-8"))
    with urllib.request.urlopen(marine_url, timeout=15) as response:
        marine_data = json.loads(response.read().decode("utf-8"))

    weather_hourly = weather_data.get("hourly")
    marine_hourly = marine_data.get("hourly")
    if not weather_hourly or not marine_hourly:
        raise ValueError("時間別データを取得できませんでした。")

    weather_df = pd.DataFrame(
        {
            "time": weather_hourly["time"],
            "wind_mps": weather_hourly["wind_speed_10m"],
            "pressure_hpa": weather_hourly["pressure_msl"],
        }
    )
    marine_df = pd.DataFrame(
        {
            "time": marine_hourly["time"],
            "wave_m": marine_hourly["wave_height"],
            "water_temp": marine_hourly["sea_surface_temperature"],
            "sea_level_m": marine_hourly["sea_level_height_msl"],
        }
    )
    merged = weather_df.merge(marine_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])
    return merged.dropna(
        subset=["wind_mps", "wave_m", "water_temp", "pressure_hpa", "sea_level_m"]
    )


CATCH_WEATHER_WIND_UNIT_KEY = "_wind_unit"


def normalize_catch_weather_wind(weather: dict) -> tuple[dict, bool]:
    """
    釣果ログの weather を m/s 基準に揃える。
    旧データは Open-Meteo の km/h を wind_mps 名で保存していたため ÷3.6 する。
    """
    if not isinstance(weather, dict):
        return {}, False
    if weather.get(CATCH_WEATHER_WIND_UNIT_KEY) == "ms":
        return weather, False

    updated = dict(weather)
    wind = updated.get("wind_mps")
    if wind is not None:
        try:
            updated["wind_mps"] = round(float(wind) / 3.6, 1)
        except (TypeError, ValueError):
            pass
    updated[CATCH_WEATHER_WIND_UNIT_KEY] = "ms"
    return updated, True


def get_weather_snapshot(target_coords: list[float], target_dt: datetime) -> dict:
    """Get nearest-hour weather snapshot for record registration."""
    hourly = fetch_open_meteo_hourly(target_coords, target_dt.date())
    if hourly.empty:
        raise ValueError("気象スナップショットを取得できませんでした。")
    nearest_idx = (hourly["time"] - pd.Timestamp(target_dt)).abs().idxmin()
    nearest = hourly.loc[nearest_idx]
    return {
        "wind_mps": round(float(nearest["wind_mps"]), 1),
        "wave_m": round(float(nearest["wave_m"]), 2),
        "water_temp": round(float(nearest["water_temp"]), 1),
        "pressure_hpa": round(float(nearest["pressure_hpa"]), 1),
        "sea_level_m": round(float(nearest["sea_level_m"]), 3),
        CATCH_WEATHER_WIND_UNIT_KEY: "ms",
    }


def evaluate_from_catch_records(
    location_name: str, today_data: dict, record_list: list[dict]
) -> tuple[str, str]:
    """Evaluate today's fishability based on past catch-condition similarity."""
    target_records = [entry for entry in record_list if entry["location"] == location_name]
    if len(target_records) < 2:
        return "データ不足", "釣果ログが2件以上あると実績ベース評価が有効になります。"

    similarities = []
    for record in target_records:
        weather = record.get("weather", {})
        distance = (
            abs(today_data["wind_mps"] - weather.get("wind_mps", today_data["wind_mps"])) * 1.2
            + abs(today_data["wave_m"] - weather.get("wave_m", today_data["wave_m"])) * 8.0
            + abs(
                today_data["water_temp"]
                - weather.get("water_temp", today_data["water_temp"])
            )
            * 1.1
            + abs(
                today_data["pressure_hpa"]
                - weather.get("pressure_hpa", today_data["pressure_hpa"])
            )
            * 0.15
        )
        score = max(0.0, 100 - distance * 4.2)
        similarities.append(score)

    avg_similarity = sum(similarities) / len(similarities)
    if avg_similarity >= 70:
        return "実績一致: 高", "過去の釣果が出た気象条件にかなり近いです。"
    if avg_similarity >= 52:
        return "実績一致: 中", "過去の釣果条件に部分的に近いです。"
    return "実績一致: 低", "過去の釣果時コンディションとの差が大きめです。"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_daily(  # pylint: disable=too-many-locals
    target_coords: list[float],
) -> pd.DataFrame:
    """
    Open-Meteoの無料APIから、7日分の気象/海況データを取得する。
    """
    lat, lon = target_coords
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "forecast_days": 7,
        "hourly": "wind_speed_10m,pressure_msl",
        "windspeed_unit": "ms",
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "forecast_days": 7,
        "hourly": "wave_height,sea_surface_temperature,sea_level_height_msl",
        "cell_selection": "sea",
    }

    weather_query = urllib.parse.urlencode(weather_params)
    marine_query = urllib.parse.urlencode(marine_params)
    weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_query}"
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?{marine_query}"

    with urllib.request.urlopen(weather_url, timeout=15) as response:
        weather_data = json.loads(response.read().decode("utf-8"))
    with urllib.request.urlopen(marine_url, timeout=15) as response:
        marine_data = json.loads(response.read().decode("utf-8"))

    weather_hourly = weather_data.get("hourly")
    marine_hourly = marine_data.get("hourly")
    if not weather_hourly or not marine_hourly:
        raise ValueError("Open-Meteoのレスポンス形式が想定と異なります。")

    weather_df = pd.DataFrame(
        {
            "time": weather_hourly["time"],
            "wind_mps": weather_hourly["wind_speed_10m"],
            "pressure_hpa": weather_hourly["pressure_msl"],
        }
    )
    marine_df = pd.DataFrame(
        {
            "time": marine_hourly["time"],
            "wave_m": marine_hourly["wave_height"],
            "water_temp": marine_hourly["sea_surface_temperature"],
            "sea_level_m": marine_hourly["sea_level_height_msl"],
        }
    )

    merged = weather_df.merge(marine_df, on="time", how="inner")
    if merged.empty:
        raise ValueError("気象データと海況データの時刻が一致しません。")

    merged["time"] = pd.to_datetime(merged["time"])
    merged["date"] = merged["time"].dt.date
    merged = merged.dropna(
        subset=["wind_mps", "wave_m", "water_temp", "pressure_hpa", "sea_level_m"]
    )
    if merged.empty:
        raise ValueError("有効な天候データが取得できませんでした。")

    def _tide_range_m(series: pd.Series) -> float:
        return float(series.max() - series.min())

    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            wind_mps=("wind_mps", "mean"),
            wave_m=("wave_m", "mean"),
            water_temp=("water_temp", "mean"),
            pressure_hpa=("pressure_hpa", "mean"),
            tide_range_m=("sea_level_m", _tide_range_m),
        )
        .sort_values("date")
        .head(7)
    )
    return daily


def evaluate_eging_condition(  # pylint: disable=too-many-locals
    location_name: str, target_date: date, weather_row: pd.Series
) -> dict:
    """
    エギング向け総合判定:
    - 潮の効きやすさ（Open-Meteo 海面高度モデルからの日次潮差）
    - 風（弱いほど高評価）
    - 波（低いほど高評価）
    - 水温（16-24度を高評価）
    - 気圧安定度（急低下を避ける）
    """
    tide_range_m = float(weather_row["tide_range_m"])
    tide_score, tide_type = tide_score_from_tide_range(tide_range_m)

    wind_mps = float(weather_row["wind_mps"])
    wind_score = max(0.0, 100 - (wind_mps * 9.5))

    wave_m = float(weather_row["wave_m"])
    wave_score = max(0.0, 100 - (wave_m * 38))

    water_temp = float(weather_row["water_temp"])
    temp_diff = abs(water_temp - 20.0)
    temp_score = max(0.0, 100 - (temp_diff * 8.5))

    pressure_hpa = float(weather_row["pressure_hpa"])
    pressure_score = max(0.0, 100 - (abs(1016 - pressure_hpa) * 3.5))

    total_score = (
        tide_score * 0.34
        + wind_score * 0.26
        + wave_score * 0.18
        + temp_score * 0.14
        + pressure_score * 0.08
    )

    rank = get_rank(total_score)

    return {
        "date": target_date,
        "location": location_name,
        "rank": rank,
        "total_score": round(total_score, 1),
        "tide_type": tide_type,
        "wind_mps": round(wind_mps, 1),
        "wave_m": round(wave_m, 1),
        "water_temp": round(water_temp, 1),
        "pressure_hpa": round(pressure_hpa, 1),
        "detail": {
            "潮": round(tide_score, 1),
            "風": round(wind_score, 1),
            "波": round(wave_score, 1),
            "水温": round(temp_score, 1),
            "気圧": round(pressure_score, 1),
        },
    }


def weekly_forecast(
    location_name: str,
    locations: dict[str, list[float]],
    days: int = 7,
) -> list[dict]:
    """Build daily eging forecast for one location."""
    location_coords = locations[location_name]
    daily_weather = fetch_open_meteo_daily(location_coords)
    results = []
    for _, wrow in daily_weather.head(days).iterrows():
        target = wrow["date"]
        results.append(evaluate_eging_condition(location_name, target, wrow))
    return results
