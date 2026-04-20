from datetime import date, timedelta
import json
import urllib.parse
import urllib.request

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="鹿児島エギング指数", layout="wide")
st.title("鹿児島エギング指数マップ 🎣")
st.caption("鹿児島の主要ポイントを対象に、エギング向けの釣りやすさを独自ロジックで判定します。")

# 9地点の座標
locations = {
    "鹿児島港": [31.583, 130.566],
    "谷山港": [31.527, 130.547],
    "指宿": [31.238, 130.641],
    "枕崎": [31.267, 130.298],
    "甑島": [31.850, 129.800],
    "種子島": [30.730, 130.998],
    "屋久島": [30.344, 130.511],
    "柏原海岸": [31.312, 131.023],
    "佐多岬": [30.994, 130.660],
}


def moon_age(target_date: date) -> float:
    """
    簡易月齢（0-29.53）を計算する。
    既知の新月日を基準に周回で求める。
    """
    known_new_moon = date(2000, 1, 6)
    synodic_month = 29.53058867
    diff_days = (target_date - known_new_moon).days
    return diff_days % synodic_month


def tide_score_from_moon(target_date: date) -> tuple[float, str]:
    """
    月齢から潮の動きやすさを近似し、エギング向けの潮スコアを返す。
    大潮寄りを高めに評価。
    """
    age = moon_age(target_date)
    # 新月/満月(0, 14.77, 29.53)付近で潮が動きやすい
    distance_to_spring = min(abs(age - 0), abs(age - 14.765), abs(age - 29.53))
    normalized = max(0.0, min(1.0, 1 - (distance_to_spring / 7.5)))
    score = 45 + (normalized * 55)

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
    return {
        "S": "red",
        "A": "orange",
        "B": "green",
        "C": "blue",
        "D": "gray",
    }.get(rank, "blue")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_daily(coords: list[float]) -> pd.DataFrame:
    """
    Open-Meteoの無料APIから、7日分の気象/海況データを取得する。
    """
    lat, lon = coords
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "forecast_days": 7,
        "hourly": "wind_speed_10m,pressure_msl",
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "forecast_days": 7,
        "hourly": "wave_height,sea_surface_temperature",
    }

    weather_url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(weather_params)}"
    marine_url = f"https://marine-api.open-meteo.com/v1/marine?{urllib.parse.urlencode(marine_params)}"

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
        }
    )

    merged = weather_df.merge(marine_df, on="time", how="inner")
    if merged.empty:
        raise ValueError("気象データと海況データの時刻が一致しません。")

    merged["time"] = pd.to_datetime(merged["time"])
    merged["date"] = merged["time"].dt.date
    merged = merged.dropna(subset=["wind_mps", "wave_m", "water_temp", "pressure_hpa"])
    if merged.empty:
        raise ValueError("有効な天候データが取得できませんでした。")

    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            wind_mps=("wind_mps", "mean"),
            wave_m=("wave_m", "mean"),
            water_temp=("water_temp", "mean"),
            pressure_hpa=("pressure_hpa", "mean"),
        )
        .sort_values("date")
        .head(7)
    )
    return daily


def evaluate_eging_condition(location_name: str, target_date: date, weather_row: pd.Series) -> dict:
    """
    エギング向け総合判定:
    - 潮の効きやすさ（月齢近似）
    - 風（弱いほど高評価）
    - 波（低いほど高評価）
    - 水温（16-24度を高評価）
    - 気圧安定度（急低下を避ける）
    """
    tide_score, tide_type = tide_score_from_moon(target_date)

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


def weekly_forecast(location_name: str, days: int = 7) -> list[dict]:
    coords = locations[location_name]
    daily_weather = fetch_open_meteo_daily(coords)
    results = []
    for _, row in daily_weather.head(days).iterrows():
        target = row["date"]
        results.append(evaluate_eging_condition(location_name, target, row))
    return results


today = date.today()
selected_location = st.sidebar.selectbox("表示するポイント", list(locations.keys()))
try:
    forecast = weekly_forecast(selected_location, days=7)
except Exception as error:
    st.error(
        "Open-Meteoから天候データを取得できませんでした。"
        "時間をおいて再試行してください。"
    )
    st.exception(error)
    st.stop()

today_result = next((item for item in forecast if item["date"] == today), forecast[0])

col_left, col_right = st.columns([1.4, 1.0])

with col_left:
    # 地図作成
    m = folium.Map(location=[31.3, 130.6], zoom_start=9)

    # ピンを立てる（今日のランク表示）
    for name, coords in locations.items():
        try:
            location_forecast = weekly_forecast(name, days=7)
            eval_today = next(
                (item for item in location_forecast if item["date"] == today),
                location_forecast[0],
            )
        except Exception:
            # 一部ポイントの取得失敗時もUI全体は維持する
            continue
        popup_text = (
            f"{name}<br>"
            f"本日ランク: {eval_today['rank']} ({eval_today['total_score']}点)<br>"
            f"潮: {eval_today['tide_type']} / 風: {eval_today['wind_mps']}m/s / 波: {eval_today['wave_m']}m"
        )
        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{name} | {eval_today['rank']}",
            icon=folium.Icon(color=rank_color(eval_today["rank"]), icon="info-sign"),
        ).add_to(m)

    st.subheader("鹿児島エギング指数マップ（本日ランク付き）")
    st_folium(m, width=900, height=620)

with col_right:
    st.subheader(f"{selected_location} の本日評価")
    st.metric("総合ランク", today_result["rank"])
    st.metric("総合スコア", f"{today_result['total_score']} / 100")
    st.write(
        f"潮: **{today_result['tide_type']}** / 風: **{today_result['wind_mps']} m/s** / "
        f"波: **{today_result['wave_m']} m** / 水温: **{today_result['water_temp']} ℃**"
    )

    detail_df = pd.DataFrame(
        [{"項目": key, "スコア": value} for key, value in today_result["detail"].items()]
    )
    st.caption("評価内訳（エギング専用ロジック）")
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.subheader("本日から1週間の予測")
forecast_df = pd.DataFrame(
    [
        {
            "日付": item["date"].strftime("%m/%d"),
            "ポイント": item["location"],
            "ランク": item["rank"],
            "総合スコア": item["total_score"],
            "潮傾向": item["tide_type"],
            "風(m/s)": item["wind_mps"],
            "波(m)": item["wave_m"],
            "水温(℃)": item["water_temp"],
            "気圧(hPa)": item["pressure_hpa"],
        }
        for item in forecast
    ]
)
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.caption(
    "※ 風速・気圧は Open-Meteo Forecast API、波高・海面水温は Open-Meteo Marine API の"
    "無料予報データを使用しています。実釣前に最新情報を再確認してください。"
)