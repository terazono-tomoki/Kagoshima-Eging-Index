"""Kagoshima eging condition dashboard using Open-Meteo forecast data."""

from datetime import date, datetime, time
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="鹿児島エギング指数", layout="wide")
st.title("鹿児島エギング指数マップ 🎣")
st.caption("鹿児島の主要ポイントを対象に、エギング向けの釣りやすさを独自ロジックで判定します。")

# 対象地点の座標（2地点のみ）
locations = {
    "東風泊": [31.074, 130.783],
    "佐多岬": [30.994, 130.660],
}

RECORDS_FILE = Path("catch_records.json")
IMAGE_DIR = Path("catch_images")
RECORDS_SECTION_PASSWORD = st.secrets.get("records_section_password")


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


def load_catch_records() -> list[dict]:
    """Load catch records from local JSON file."""
    if not RECORDS_FILE.exists():
        return []
    with RECORDS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_catch_records(record_list: list[dict]) -> None:
    """Save catch records into local JSON file."""
    with RECORDS_FILE.open("w", encoding="utf-8") as file:
        json.dump(record_list, file, ensure_ascii=False, indent=2)


def save_uploaded_image(uploaded_file) -> str | None:
    """Save uploaded squid image and return relative path."""
    if uploaded_file is None:
        return None
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    file_name = f"{uuid.uuid4().hex}{suffix}"
    file_path = IMAGE_DIR / file_name
    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(file_path.as_posix())


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_hourly(target_coords: list[float], target_day: date) -> pd.DataFrame:
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
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "start_date": day_text,
        "end_date": day_text,
        "hourly": "wave_height,sea_surface_temperature",
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
        }
    )
    merged = weather_df.merge(marine_df, on="time", how="inner")
    merged["time"] = pd.to_datetime(merged["time"])
    return merged.dropna(subset=["wind_mps", "wave_m", "water_temp", "pressure_hpa"])


def get_weather_snapshot(location_name: str, target_dt: datetime) -> dict:
    """Get nearest-hour weather snapshot for record registration."""
    hourly = fetch_open_meteo_hourly(locations[location_name], target_dt.date())
    if hourly.empty:
        raise ValueError("気象スナップショットを取得できませんでした。")
    nearest_idx = (hourly["time"] - pd.Timestamp(target_dt)).abs().idxmin()
    nearest = hourly.loc[nearest_idx]
    return {
        "wind_mps": round(float(nearest["wind_mps"]), 1),
        "wave_m": round(float(nearest["wave_m"]), 2),
        "water_temp": round(float(nearest["water_temp"]), 1),
        "pressure_hpa": round(float(nearest["pressure_hpa"]), 1),
    }


def evaluate_from_catch_records(
    location_name: str, today_data: dict, record_list: list[dict]
) -> tuple[str, str]:
    """Evaluate today's fishability based on past catch-condition similarity."""
    target_records = [item for item in record_list if item["location"] == location_name]
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
def fetch_open_meteo_daily(target_coords: list[float]) -> pd.DataFrame:
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
    }
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Tokyo",
        "forecast_days": 7,
        "hourly": "wave_height,sea_surface_temperature",
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
    """Build daily eging forecast for one location."""
    location_coords = locations[location_name]
    daily_weather = fetch_open_meteo_daily(location_coords)
    results = []
    for _, row in daily_weather.head(days).iterrows():
        target = row["date"]
        results.append(evaluate_eging_condition(location_name, target, row))
    return results


today = date.today()
location_options = list(locations.keys())
st.sidebar.radio(
    "表示するポイント",
    location_options,
    key="point_selector",
)
current_point = st.session_state["point_selector"]
st.sidebar.caption(f"現在の選択: {current_point}")

col_left, col_right = st.columns([1.4, 1.0])

with col_left:
    # 地図作成
    m = folium.Map(location=[31.3, 130.6], zoom_start=9)

    # ピンを立てる（今日のランク表示）
    for name, point_coords in locations.items():
        try:
            location_forecast = weekly_forecast(name, days=7)
            eval_today = next(
                (item for item in location_forecast if item["date"] == today),
                location_forecast[0],
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            # 一部ポイントの取得失敗時もUI全体は維持する
            continue
        popup_text = (
            f"{name}<br>"
            f"本日ランク: {eval_today['rank']} ({eval_today['total_score']}点)<br>"
            f"潮: {eval_today['tide_type']} / "
            f"風: {eval_today['wind_mps']}m/s / 波: {eval_today['wave_m']}m"
        )
        folium.Marker(
            location=point_coords,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{name} | {eval_today['rank']}",
            icon=folium.Icon(color=rank_color(eval_today["rank"]), icon="info-sign"),
        ).add_to(m)

    st.subheader("鹿児島エギング指数マップ（本日ランク付き）")
    st_folium(m, width=900, height=620)

try:
    forecast = weekly_forecast(current_point, days=7)
except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
    st.error(
        "Open-Meteoから天候データを取得できませんでした。"
        "時間をおいて再試行してください。"
    )
    st.exception(error)
    st.stop()

today_result = next((item for item in forecast if item["date"] == today), forecast[0])

with col_right:
    st.subheader("指定したポイントの評価")
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

st.divider()
st.subheader("釣果ログ（写真 + 日時 + 気象）")
if not RECORDS_SECTION_PASSWORD:
    st.warning(
        "記録欄パスワードが未設定です。"
        ".streamlit/secrets.toml に records_section_password を設定してください。"
    )
elif not st.session_state.get("records_auth_unlocked"):
    st.caption("釣果の閲覧・保存にはパスワードが必要です。")
    with st.form("records_auth_form"):
        gate_pw = st.text_input("パスワード", type="password")
        gate_submit = st.form_submit_button("ログイン")
    if gate_submit:
        if gate_pw == str(RECORDS_SECTION_PASSWORD):
            st.session_state.records_auth_unlocked = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
else:
    if st.button("ログアウト（記録欄を隠す）"):
        st.session_state.records_auth_unlocked = False
        st.rerun()

    record_items = load_catch_records()
    record_eval_label, record_eval_text = evaluate_from_catch_records(
        current_point, today_result, record_items
    )
    st.info(f"釣果ログ実績評価: {record_eval_label} - {record_eval_text}")

    with st.form("catch_log_form", clear_on_submit=True):
        record_col1, record_col2 = st.columns(2)
        with record_col1:
            catch_location = st.selectbox(
                "釣れたポイント", list(locations.keys()), key="catch_location"
            )
            catch_date = st.date_input("釣れた日", value=today, key="catch_date")
            catch_time = st.time_input("釣れた時刻", value=time(20, 0), key="catch_time")
        with record_col2:
            squid_size = st.number_input(
                "胴長(cm)", min_value=5.0, max_value=70.0, value=20.0, step=0.5
            )
            squid_count = st.number_input(
                "杯数", min_value=1, max_value=30, value=1, step=1
            )
            memo = st.text_area("メモ", placeholder="ヒットエギ・レンジ・潮位など")
        squid_photo = st.file_uploader(
            "イカ写真をアップロード", type=["jpg", "jpeg", "png", "webp"]
        )
        submit_record = st.form_submit_button("釣果ログを保存")

    if submit_record:
        catch_datetime = datetime.combine(catch_date, catch_time)
        try:
            weather_snapshot = get_weather_snapshot(catch_location, catch_datetime)
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            weather_snapshot = {
                "wind_mps": None,
                "wave_m": None,
                "water_temp": None,
                "pressure_hpa": None,
            }
        photo_path = save_uploaded_image(squid_photo)
        record_items.append(
            {
                "id": uuid.uuid4().hex,
                "location": catch_location,
                "datetime": catch_datetime.isoformat(timespec="minutes"),
                "size_cm": float(squid_size),
                "count": int(squid_count),
                "memo": memo.strip(),
                "photo_path": photo_path,
                "weather": weather_snapshot,
            }
        )
        save_catch_records(record_items)
        st.success("釣果ログを保存しました。")
        st.rerun()

    if record_items:
        sorted_records = sorted(record_items, key=lambda item: item["datetime"], reverse=True)
        history_df = pd.DataFrame(
            [
                {
                    "日時": item["datetime"].replace("T", " "),
                    "ポイント": item["location"],
                    "杯数": item["count"],
                    "胴長(cm)": item["size_cm"],
                    "風(m/s)": item["weather"].get("wind_mps"),
                    "波(m)": item["weather"].get("wave_m"),
                    "水温(℃)": item["weather"].get("water_temp"),
                    "気圧(hPa)": item["weather"].get("pressure_hpa"),
                    "メモ": item["memo"],
                }
                for item in sorted_records
            ]
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.caption("最新の釣果写真")
        photo_cols = st.columns(3)
        photo_idx = 0
        for item in sorted_records:
            if not item.get("photo_path"):
                continue
            photo_file = Path(item["photo_path"])
            if not photo_file.exists():
                continue
            with photo_cols[photo_idx % 3]:
                st.image(
                    str(photo_file),
                    caption=f"{item['location']} {item['datetime'].replace('T', ' ')}",
                )
            photo_idx += 1
