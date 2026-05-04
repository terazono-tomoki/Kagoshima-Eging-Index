"""Kagoshima eging condition dashboard using Open-Meteo (incl. marine sea level / tides)."""

from datetime import date, datetime, time
import json
from pathlib import Path
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# DB・画像はカレントディレクトリではなくこのファイルと同じフォルダに固定する
_APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="鹿児島エギング指数", layout="wide")
st.title("鹿児島エギング指数マップ 🎣")
st.caption("選択したポイントを対象に、エギング向けの釣りやすさを独自ロジックで判定します。")

# 対象地点の座標（2地点のみ）
locations = {
    "東風泊": [31.074, 130.783],
    "佐多岬": [30.994, 130.660],
}

RECORDS_DB = _APP_ROOT / "catch_records.db"
# sqlite3.connect に渡す（Path のままだと環境によって相性問題が出ることがある）
RECORDS_DB_ABS = str(RECORDS_DB.resolve())
IMAGE_DIR = _APP_ROOT / "catch_images"
RECORDS_SECTION_PASSWORD = st.secrets.get("records_section_password")


def tide_score_from_tide_range(tide_range_m: float) -> tuple[float, str]:
    """
    その日の海面高度（潮汐込みモデル）から求めた潮差に基づき潮スコアを返す。
    tide_range_m は同日の hourly sea level の最大値と最小値の差（メートル）。
    """
    # 潮差が大きいほど「大潮に近い」動きとみなす（閾値はおおよそ西南海域向けの目安）
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


def _init_catch_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catch_records (
            id TEXT PRIMARY KEY NOT NULL,
            location TEXT NOT NULL,
            datetime TEXT NOT NULL,
            size_cm REAL NOT NULL,
            count INTEGER NOT NULL,
            memo TEXT NOT NULL DEFAULT '',
            photo_path TEXT,
            weather_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )


def _ensure_catch_db() -> None:
    """Create SQLite schema for catch records if needed."""
    conn = sqlite3.connect(RECORDS_DB_ABS)
    try:
        _init_catch_db(conn)
        conn.commit()
    finally:
        conn.close()


def load_catch_records() -> list[dict]:
    """Load catch records from local SQLite database."""
    _ensure_catch_db()
    conn = sqlite3.connect(RECORDS_DB_ABS)
    try:
        cur = conn.execute(
            """
            SELECT id, location, datetime, size_cm, count, memo, photo_path, weather_json
            FROM catch_records
            ORDER BY datetime ASC
            """
        )
        result: list[dict] = []
        for db_row in cur.fetchall():
            rid_val, loc_val, dt_val, size_val, cnt_val, memo_val, path_val, weather_json = db_row
            try:
                weather = json.loads(weather_json) if weather_json else {}
            except json.JSONDecodeError:
                weather = {}
            if not isinstance(weather, dict):
                weather = {}
            result.append(
                {
                    "id": rid_val,
                    "location": loc_val,
                    "datetime": dt_val,
                    "size_cm": size_val,
                    "count": cnt_val,
                    "memo": memo_val or "",
                    "photo_path": path_val,
                    "weather": weather,
                }
            )
        return result
    finally:
        conn.close()


def count_catch_records() -> int:
    """Return number of rows in catch_records (for UI when the section is locked)."""
    _ensure_catch_db()
    conn = sqlite3.connect(RECORDS_DB_ABS)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM catch_records")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def save_catch_records(record_list: list[dict]) -> None:
    """Replace all catch records in SQLite (same contract as the former JSON file)."""
    _ensure_catch_db()
    conn = sqlite3.connect(RECORDS_DB_ABS)
    try:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM catch_records")
        for stored in record_list:
            row_id = stored.get("id") or uuid.uuid4().hex
            weather = stored.get("weather")
            if not isinstance(weather, dict):
                weather = {}
            conn.execute(
                """
                INSERT INTO catch_records
                (id, location, datetime, size_cm, count, memo, photo_path, weather_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    stored["location"],
                    stored["datetime"],
                    float(stored["size_cm"]),
                    int(stored["count"]),
                    stored.get("memo", "") or "",
                    stored.get("photo_path"),
                    json.dumps(weather, ensure_ascii=False),
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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


def _photo_path_after_edit(uploaded, delete_flag: bool, previous: dict) -> str | None:
    """Resolve stored photo path after the user edits or removes the image."""
    if uploaded:
        new_path = save_uploaded_image(uploaded)
        old_p = previous.get("photo_path")
        if old_p:
            op = Path(old_p)
            if op.exists():
                op.unlink()
        return new_path
    if delete_flag:
        old_p = previous.get("photo_path")
        if old_p:
            op = Path(old_p)
            if op.exists():
                op.unlink()
        return None
    return previous.get("photo_path")


def parse_record_datetime(record: dict) -> datetime:
    """Parse ISO datetime from a catch record; fallback if missing or invalid."""
    raw = record.get("datetime")
    if not raw:
        return datetime.combine(date.today(), time(20, 0))
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return datetime.combine(date.today(), time(20, 0))


def index_of_record_in_store(store: list[dict], record: dict) -> int:
    """Find index of a record in the persisted list (by id, identity, or datetime+location)."""
    rec_id = record.get("id")
    if rec_id:
        for i, entry in enumerate(store):
            if entry.get("id") == rec_id:
                return i
    for i, entry in enumerate(store):
        if entry is record:
            return i
    dt_key = record.get("datetime")
    loc_key = record.get("location")
    if dt_key is not None and loc_key is not None:
        for i, entry in enumerate(store):
            if entry.get("datetime") == dt_key and entry.get("location") == loc_key:
                return i
    return -1


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
        "sea_level_m": round(float(nearest["sea_level_m"]), 3),
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


def evaluate_eging_condition(location_name: str, target_date: date, weather_row: pd.Series) -> dict:
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


def weekly_forecast(location_name: str, days: int = 7) -> list[dict]:
    """Build daily eging forecast for one location."""
    location_coords = locations[location_name]
    daily_weather = fetch_open_meteo_daily(location_coords)
    results = []
    for _, wrow in daily_weather.head(days).iterrows():
        target = wrow["date"]
        results.append(evaluate_eging_condition(location_name, target, wrow))
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
                (fc for fc in location_forecast if fc["date"] == today),
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

today_result = next((fc for fc in forecast if fc["date"] == today), forecast[0])

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
            "日付": fc["date"].strftime("%m/%d"),
            "ポイント": fc["location"],
            "ランク": fc["rank"],
            "総合スコア": fc["total_score"],
            "潮傾向": fc["tide_type"],
            "風(m/s)": fc["wind_mps"],
            "波(m)": fc["wave_m"],
            "水温(℃)": fc["water_temp"],
            "気圧(hPa)": fc["pressure_hpa"],
        }
        for fc in forecast
    ]
)
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.caption(
    "※ 風速・気圧は Open-Meteo Forecast API、波高・海面水温・海面高度（潮汐を含むモデル）は "
    "Open-Meteo Marine API の無料予報データを使用しています。実釣前に最新情報を再確認してください。"
)

st.divider()
st.subheader("釣果ログ（写真 + 日時 + 気象）")
if not RECORDS_SECTION_PASSWORD:
    st.warning(
        "記録欄パスワードが未設定です。"
        ".streamlit/secrets.toml に records_section_password を設定してください。"
    )
    _n_saved = count_catch_records()
    if _n_saved:
        st.info(
            f"SQLite（{RECORDS_DB.name}）には釣果が **{_n_saved} 件**保存されています。"
            "一覧や編集を表示するには、上記のとおりパスワードを設定し、記録欄にログインしてください。"
        )
elif not st.session_state.get("records_auth_unlocked"):
    st.caption("釣果の閲覧・保存にはパスワードが必要です。")
    _n_saved = count_catch_records()
    if _n_saved:
        st.info(
            f"保存済みの釣果は **{_n_saved} 件**です。"
            "下のフォームでログインすると一覧・編集・削除ができます。"
        )
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
                "sea_level_m": None,
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
        try:
            save_catch_records(record_items)
        except OSError as exc:
            st.error("ファイルへ書き込めませんでした（権限・同期・ロックの可能性）。")
            st.exception(exc)
        except sqlite3.Error as exc:
            st.error("SQLiteへの保存に失敗しました。")
            st.exception(exc)
        else:
            st.success("釣果ログを保存しました。")
            st.rerun()

    if record_items:
        sorted_records = sorted(record_items, key=lambda log: log["datetime"], reverse=True)

        with st.expander("ログ編集", expanded=False):
            with st.form("edit_record_form"):
                edit_idx = st.selectbox(
                    "編集するログ",
                    options=list(range(len(sorted_records))),
                    format_func=lambda idx: (
                        f"{sorted_records[idx]['datetime'].replace('T', ' ')} | "
                        f"{sorted_records[idx]['location']} | "
                        f"{sorted_records[idx].get('count', '?')}杯"
                    ),
                )
                rec_before = sorted_records[edit_idx]
                dt_parsed = parse_record_datetime(rec_before)
                loc_keys = list(locations.keys())
                loc_index = (
                    loc_keys.index(rec_before["location"])
                    if rec_before.get("location") in locations
                    else 0
                )
                ed_location = st.selectbox(
                    "釣れたポイント", loc_keys, index=loc_index, key=f"ed_loc_{edit_idx}"
                )
                ed_date = st.date_input(
                    "釣れた日", value=dt_parsed.date(), key=f"ed_date_{edit_idx}"
                )
                ed_time = st.time_input(
                    "釣れた時刻", value=dt_parsed.time(), key=f"ed_time_{edit_idx}"
                )
                ed_size = st.number_input(
                    "胴長(cm)",
                    min_value=5.0,
                    max_value=70.0,
                    value=float(rec_before.get("size_cm", 20.0)),
                    step=0.5,
                    key=f"ed_size_{edit_idx}",
                )
                ed_count = st.number_input(
                    "杯数",
                    min_value=1,
                    max_value=30,
                    value=int(rec_before.get("count", 1)),
                    step=1,
                    key=f"ed_count_{edit_idx}",
                )
                ed_memo = st.text_area(
                    "メモ",
                    value=rec_before.get("memo", ""),
                    placeholder="ヒットエギ・レンジ・潮位など",
                    key=f"ed_memo_{edit_idx}",
                )
                existing_photo = rec_before.get("photo_path")
                if existing_photo and Path(existing_photo).exists():
                    st.caption(f"現在の写真: {existing_photo}")
                ed_photo = st.file_uploader(
                    "新しい写真に差し替え（未選択ならそのまま）",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"ed_photo_{edit_idx}",
                )
                delete_photo_on_edit = st.checkbox(
                    "写真を削除する（ファイルも削除）",
                    value=False,
                    key=f"ed_delphoto_{edit_idx}",
                )
                submit_edit = st.form_submit_button("この内容で更新")

            if submit_edit:
                store_idx = index_of_record_in_store(record_items, rec_before)
                if store_idx < 0:
                    st.error("更新対象のログが見つかりませんでした。")
                else:
                    catch_dt = datetime.combine(ed_date, ed_time)
                    try:
                        weather_snapshot = get_weather_snapshot(ed_location, catch_dt)
                    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                        weather_snapshot = {
                            "wind_mps": None,
                            "wave_m": None,
                            "water_temp": None,
                            "pressure_hpa": None,
                            "sea_level_m": None,
                        }
                    old = record_items[store_idx]
                    photo_stored = _photo_path_after_edit(
                        ed_photo, delete_photo_on_edit, old
                    )
                    record_id = old.get("id") or uuid.uuid4().hex
                    record_items[store_idx] = {
                        "id": record_id,
                        "location": ed_location,
                        "datetime": catch_dt.isoformat(timespec="minutes"),
                        "size_cm": float(ed_size),
                        "count": int(ed_count),
                        "memo": ed_memo.strip(),
                        "photo_path": photo_stored,
                        "weather": weather_snapshot,
                    }
                    try:
                        save_catch_records(record_items)
                    except OSError as exc:
                        st.error("ファイルへ書き込めませんでした。")
                        st.exception(exc)
                    except sqlite3.Error as exc:
                        st.error("SQLiteへの保存に失敗しました。")
                        st.exception(exc)
                    else:
                        st.success("ログを更新しました。")
                        st.rerun()

        with st.expander("ログ削除", expanded=False):
            with st.form("delete_record_form"):
                delete_idx = st.selectbox(
                    "削除するログ",
                    options=list(range(len(sorted_records))),
                    format_func=lambda idx: (
                        f"{sorted_records[idx]['datetime'].replace('T', ' ')} | "
                        f"{sorted_records[idx]['location']} | "
                        f"{sorted_records[idx]['count']}杯"
                    ),
                )
                delete_photo_file = st.checkbox(
                    "このログの写真ファイルも削除する", value=True
                )
                submit_delete = st.form_submit_button("選択したログを削除")

            if submit_delete:
                target_record = sorted_records[delete_idx]
                target_photo_path = target_record.get("photo_path")
                try:
                    record_items.remove(target_record)
                except ValueError:
                    st.error("削除対象のログが見つかりませんでした。")
                else:
                    if delete_photo_file and target_photo_path:
                        photo_file = Path(target_photo_path)
                        if photo_file.exists():
                            photo_file.unlink()
                    try:
                        save_catch_records(record_items)
                    except OSError as exc:
                        st.error("ファイルへ書き込めませんでした。")
                        st.exception(exc)
                    except sqlite3.Error as exc:
                        st.error("SQLiteへの保存に失敗しました。")
                        st.exception(exc)
                    else:
                        st.success("ログを削除しました。")
                        st.rerun()

        filter_col1, filter_col2 = st.columns([1.1, 1.4])
        with filter_col1:
            date_filter_enabled = st.checkbox("日付で絞り込む", value=False)
        with filter_col2:
            filter_date = st.date_input("表示する日付", value=today, disabled=not date_filter_enabled)

        if date_filter_enabled:
            displayed_records = []
            for log in sorted_records:
                try:
                    log_date = datetime.fromisoformat(log["datetime"]).date()
                except (TypeError, ValueError):
                    continue
                if log_date == filter_date:
                    displayed_records.append(log)
        else:
            displayed_records = sorted_records

        if not displayed_records:
            st.info("指定日の釣果ログはありません。")
        else:
            history_df = pd.DataFrame(
                [
                    {
                        "日時": hist["datetime"].replace("T", " "),
                        "ポイント": hist["location"],
                        "杯数": hist["count"],
                        "胴長(cm)": hist["size_cm"],
                        "風(m/s)": hist["weather"].get("wind_mps"),
                        "波(m)": hist["weather"].get("wave_m"),
                        "水温(℃)": hist["weather"].get("water_temp"),
                        "気圧(hPa)": hist["weather"].get("pressure_hpa"),
                        "海面高度(m)": hist["weather"].get("sea_level_m"),
                        "メモ": hist["memo"],
                    }
                    for hist in displayed_records
                ]
            )
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            st.caption("最新の釣果写真")
            photo_cols = st.columns(3)
            photo_idx = 0
            for hist in displayed_records:
                if not hist.get("photo_path"):
                    continue
                photo_file = Path(hist["photo_path"])
                if not photo_file.exists():
                    continue
                with photo_cols[photo_idx % 3]:
                    st.image(
                        str(photo_file),
                        caption=f"{hist['location']} {hist['datetime'].replace('T', ' ')}",
                    )
                photo_idx += 1
