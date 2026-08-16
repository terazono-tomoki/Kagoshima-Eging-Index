"""Kagoshima eging condition dashboard using Open-Meteo (incl. marine sea level / tides)."""

from datetime import date, datetime, time
import json
import math
import mimetypes
from pathlib import Path
import statistics
import urllib.error
import urllib.parse
import urllib.request
import uuid

import boto3
import folium
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st
from streamlit_folium import st_folium

from eging_forecast import (
    CATCH_WEATHER_WIND_UNIT_KEY,
    fetch_open_meteo_hourly,
    get_weather_snapshot,
    normalize_catch_weather_wind,
    rank_color,
    weekly_forecast,
)

_APP_ROOT = Path(__file__).resolve().parent


def format_wind_display(wind_mps: float | None) -> str:
    """風速を km/h 主表示・m/s 併記で返す（内部計算は m/s のまま）。"""
    if wind_mps is None:
        return "-"
    mps = float(wind_mps)
    kmh = int(round(mps * 3.6))
    mps_text = f"{mps:g}" if mps != int(mps) else str(int(mps))
    return f"{kmh} km/h（{mps_text} m/s）"


def format_point_conditions_markdown(result: dict) -> str:
    """本日評価パネル用の潮・風・波・水温サマリー（Markdown）。"""
    tide = result["tide_type"]
    wind = format_wind_display(result["wind_mps"])
    wave = result["wave_m"]
    water_temp = result["water_temp"]
    return (
        f"潮: **{tide}** / 風: **{wind}** / "
        f"波: **{wave} m** / 水温: **{water_temp} ℃**"
    )


def _weather_metric_for_suggest(value, fallback: float) -> float:
    """気象値を float にする。欠損・不正値は fallback を使う。"""
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number):
        return fallback
    return number


def _catch_weather_values(
    location_name: str, record_list: list[dict], field: str
) -> list[float]:
    """指定ポイント・指標について、釣果ログに保存された気象値を集める。"""
    values = []
    for record in record_list:
        if not isinstance(record, dict) or record.get("location") != location_name:
            continue
        weather = record.get("weather")
        if not isinstance(weather, dict):
            continue
        raw = weather.get(field)
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


FAVORABLE_CONDITION_FIELDS = ("wind_mps", "wave_m", "water_temp", "pressure_hpa")


def derive_favorable_conditions(
    location_name: str, record_list: list[dict]
) -> dict[str, dict] | None:
    """
    釣果ログから、そのポイントで釣れた時の気象条件（平均・目安レンジ）を導出する。
    指標ごとにサンプルが2件未満なら、その指標は結果から除く。
    全指標が2件未満なら None（分析に使えるデータがない）。
    """
    if not isinstance(record_list, list):
        return None

    stats: dict[str, dict] = {}
    for field in FAVORABLE_CONDITION_FIELDS:
        values = _catch_weather_values(location_name, record_list, field)
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values)
        low = mean - stdev
        if field in ("wind_mps", "wave_m"):
            low = max(0.0, low)
        stats[field] = {
            "mean": mean,
            "low": low,
            "high": mean + stdev,
            "n": len(values),
        }

    return stats or None


def suggest_best_hours(
    location_name: str,
    coords: list[float],
    target_day: date,
    record_list: list[dict],
) -> list[dict]:
    """
    釣果ログとの気象類似度をもとに、指定日の時間帯ごとの釣れやすさスコアを返す。
    釣果ログが2件未満、または時間別データが取得できない場合は空リスト。
    """
    target_records = [
        entry
        for entry in record_list
        if isinstance(entry, dict)
        and entry.get("location") == location_name
        and isinstance(entry.get("weather"), dict)
    ]
    if len(target_records) < 2:
        return []

    hourly = fetch_open_meteo_hourly(coords, target_day)
    if hourly.empty:
        return []

    results = []
    for _, hour_row in hourly.iterrows():
        hour_metrics = {
            "wind_mps": float(hour_row["wind_mps"]),
            "wave_m": float(hour_row["wave_m"]),
            "water_temp": float(hour_row["water_temp"]),
            "pressure_hpa": float(hour_row["pressure_hpa"]),
        }
        similarities = []
        for record in target_records:
            weather = record["weather"]
            distance = (
                abs(
                    hour_metrics["wind_mps"]
                    - _weather_metric_for_suggest(
                        weather.get("wind_mps"), hour_metrics["wind_mps"]
                    )
                )
                * 1.2
                + abs(
                    hour_metrics["wave_m"]
                    - _weather_metric_for_suggest(
                        weather.get("wave_m"), hour_metrics["wave_m"]
                    )
                )
                * 8.0
                + abs(
                    hour_metrics["water_temp"]
                    - _weather_metric_for_suggest(
                        weather.get("water_temp"), hour_metrics["water_temp"]
                    )
                )
                * 1.1
                + abs(
                    hour_metrics["pressure_hpa"]
                    - _weather_metric_for_suggest(
                        weather.get("pressure_hpa"), hour_metrics["pressure_hpa"]
                    )
                )
                * 0.15
            )
            similarities.append(max(0.0, 100 - distance * 4.2))

        results.append(
            {
                "time": hour_row["time"],
                "score": round(sum(similarities) / len(similarities), 1),
                "wind_mps": round(hour_metrics["wind_mps"], 1),
                "wave_m": round(hour_metrics["wave_m"], 2),
                "water_temp": round(hour_metrics["water_temp"], 1),
                "pressure_hpa": round(hour_metrics["pressure_hpa"], 1),
            }
        )
    return results


def summarize_best_windows(
    hourly_scores: list[dict], top_fraction: float = 0.3
) -> list[dict]:
    """
    時間帯別スコアの上位帯を連続する時間ごとにまとめ、おすすめ時間帯を返す。
    スコアの高い順に並べる。
    """
    if not hourly_scores:
        return []

    ranked = sorted(hourly_scores, key=lambda row: row["score"], reverse=True)
    cutoff_count = max(1, round(len(hourly_scores) * top_fraction))
    threshold = ranked[cutoff_count - 1]["score"]

    windows: list[list[dict]] = []
    current: list[dict] = []
    for row in hourly_scores:
        if row["score"] >= threshold:
            current.append(row)
        else:
            if current:
                windows.append(current)
                current = []
    if current:
        windows.append(current)

    summarized = [
        {
            "start": window[0]["time"],
            "end": window[-1]["time"],
            "avg_score": round(sum(r["score"] for r in window) / len(window), 1),
        }
        for window in windows
    ]
    summarized.sort(key=lambda w: w["avg_score"], reverse=True)
    return summarized


st.set_page_config(page_title="鹿児島エギング指数", layout="wide")
st.title("鹿児島エギング指数マップ 🎣")
st.caption("選択したポイントを対象に、エギング向けの釣りやすさを独自ロジックで判定します。")

locations = {
    "東風泊": [31.341081, 131.068983],
    "佐多岬": [30.994, 130.660],
}


def _require_postgres_database_url() -> str:
    """Secrets から PostgreSQL の接続 URL を返す。無ければエラーを表示してアプリを停止する。"""
    keys = ("catch_records_database_url", "DATABASE_URL")
    for key in keys:
        raw = st.secrets.get(key)
        if raw is None:
            continue
        url = str(raw).strip()
        if not url:
            continue
        lo = url.lower()
        if key == "DATABASE_URL" and not (
            lo.startswith("postgresql://") or lo.startswith("postgres://")
        ):
            continue
        return url
    st.error(
        "釣果ログには **PostgreSQL** が必要です。\n\n"
        ".streamlit/secrets.toml に **catch_records_database_url** または "
        "**DATABASE_URL**（`postgresql://` または `postgres://` で始まる URL）を設定してください。"
    )
    st.stop()


POSTGRES_DATABASE_URL = _require_postgres_database_url()

IMAGE_DIR = _APP_ROOT / "catch_images"
RECORDS_SECTION_PASSWORD = st.secrets.get("records_section_password")

_REMOTE_PHOTO_PREFIX = "s3key:"


def _photo_storage_enabled() -> bool:
    """Secrets に S3 互換ストレージの必須項目がそろっているとき True。"""
    b = st.secrets.get("photo_storage_s3_bucket")
    ak = st.secrets.get("photo_storage_s3_access_key")
    sk = st.secrets.get("photo_storage_s3_secret_key")
    return bool(b and ak and sk)


@st.cache_resource
def _s3_photo_client():
    endpoint_raw = st.secrets.get("photo_storage_s3_endpoint_url")
    endpoint = str(endpoint_raw).strip() if endpoint_raw else ""
    region_raw = st.secrets.get("photo_storage_s3_region")
    region = str(region_raw).strip() if region_raw else "auto"
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=st.secrets["photo_storage_s3_access_key"],
        aws_secret_access_key=st.secrets["photo_storage_s3_secret_key"],
        region_name=region or "auto",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def delete_stored_photo(stored: str | None) -> None:
    """ローカルファイルまたはリモートオブジェクトを削除する（失敗しても例外は握りつぶす）。"""
    if not stored:
        return
    if stored.startswith(_REMOTE_PHOTO_PREFIX):
        if not _photo_storage_enabled():
            return
        key = stored[len(_REMOTE_PHOTO_PREFIX) :]
        bucket = str(st.secrets["photo_storage_s3_bucket"]).strip()
        try:
            _s3_photo_client().delete_object(Bucket=bucket, Key=key)
        except ClientError:
            pass
        return
    if stored.startswith(("http://", "https://")):
        return
    p = Path(stored)
    if p.exists():
        p.unlink()


def photo_display_url(stored: str | None) -> str | None:
    """st.image に渡すパスまたは HTTPS URL。表示できないときは None。"""
    if not stored:
        return None
    if stored.startswith(("http://", "https://")):
        return stored
    if stored.startswith(_REMOTE_PHOTO_PREFIX):
        if not _photo_storage_enabled():
            return None
        key = stored[len(_REMOTE_PHOTO_PREFIX) :]
        bucket = str(st.secrets["photo_storage_s3_bucket"]).strip()
        pub_raw = st.secrets.get("photo_storage_public_base_url")
        pub = str(pub_raw).strip().rstrip("/") if pub_raw else ""
        if pub:
            return f"{pub}/{key}"
        return _s3_photo_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
    p = Path(stored)
    return str(p) if p.exists() else None


def _postgres_url_with_ssl(url: str) -> str:
    """マネージド PostgreSQL で SSL 省略時に sslmode=require を付与（localhost は対象外）。"""
    lo = url.lower()
    if "sslmode=" in lo:
        return url
    if not (lo.startswith("postgresql://") or lo.startswith("postgres://")):
        return url
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        if host in ("localhost", "127.0.0.1", "::1", ""):
            return url
    except ValueError:
        pass
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}sslmode=require"


def catch_records_storage_hint() -> str:
    """UI 向けの保存先の短文。"""
    return "PostgreSQL（Secrets の catch_records_database_url または DATABASE_URL）"


@st.cache_resource
def _catch_records_engine():
    return create_engine(
        _postgres_url_with_ssl(POSTGRES_DATABASE_URL),
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )


_CATCH_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS catch_records (
    id TEXT PRIMARY KEY NOT NULL,
    location TEXT NOT NULL,
    datetime TEXT NOT NULL,
    size_cm DOUBLE PRECISION NOT NULL,
    count INTEGER NOT NULL,
    memo TEXT NOT NULL DEFAULT '',
    photo_path TEXT,
    weather_json TEXT NOT NULL DEFAULT '{}'
)
"""

_UPSERT_CATCH_RECORD_SQL = text(
    """
    INSERT INTO catch_records
    (id, location, datetime, size_cm, count, memo, photo_path, weather_json)
    VALUES (:id, :loc, :dt, :size_cm, :cnt, :memo, :photo_path, :weather_json)
    ON CONFLICT (id) DO UPDATE SET
        location = EXCLUDED.location,
        datetime = EXCLUDED.datetime,
        size_cm = EXCLUDED.size_cm,
        count = EXCLUDED.count,
        memo = EXCLUDED.memo,
        photo_path = EXCLUDED.photo_path,
        weather_json = EXCLUDED.weather_json
    """
)


def _catch_record_sql_params(record: dict) -> dict:
    """Bind parameters for INSERT/UPSERT of one catch record."""
    row_id = record.get("id") or uuid.uuid4().hex
    weather = record.get("weather")
    if not isinstance(weather, dict):
        weather = {}
    return {
        "id": row_id,
        "loc": record["location"],
        "dt": record["datetime"],
        "size_cm": float(record["size_cm"]),
        "cnt": int(record["count"]),
        "memo": record.get("memo", "") or "",
        "photo_path": record.get("photo_path"),
        "weather_json": json.dumps(weather, ensure_ascii=False),
    }


def _ensure_catch_db() -> None:
    """Create schema for catch records if needed (PostgreSQL)."""
    engine = _catch_records_engine()
    with engine.begin() as conn:
        conn.execute(text(_CATCH_RECORDS_DDL))


def _catch_record_from_db_row(db_row) -> tuple[dict, bool]:
    """DB 行を釣果 dict に変換する。天候の正規化があったとき True。"""
    rid_val, loc_val, dt_val, size_val, cnt_val, memo_val, path_val, weather_json = db_row
    try:
        weather = json.loads(weather_json) if weather_json else {}
    except json.JSONDecodeError:
        weather = {}
    if not isinstance(weather, dict):
        weather = {}
    weather, changed = normalize_catch_weather_wind(weather)
    record = {
        "id": rid_val,
        "location": loc_val,
        "datetime": dt_val,
        "size_cm": float(size_val),
        "count": int(cnt_val),
        "memo": memo_val or "",
        "photo_path": path_val,
        "weather": weather,
    }
    return record, changed


def load_catch_records() -> list[dict]:
    """Load catch records from the configured database."""
    _ensure_catch_db()
    engine = _catch_records_engine()
    result: list[dict] = []
    to_persist: list[dict] = []
    with engine.connect() as conn:
        cur = conn.execute(
            text(
                """
                SELECT id, location, datetime, size_cm, count, memo, photo_path, weather_json
                FROM catch_records
                ORDER BY datetime ASC
                """
            )
        )
        for db_row in cur.fetchall():
            record, changed = _catch_record_from_db_row(db_row)
            result.append(record)
            if changed:
                to_persist.append(record)
    for record in to_persist:
        upsert_catch_record(record)
    return result


def count_catch_records() -> int:
    """Return number of rows in catch_records (for UI when the section is locked)."""
    _ensure_catch_db()
    engine = _catch_records_engine()
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM catch_records")).scalar_one()
        return int(n)


def upsert_catch_record(record: dict) -> None:
    """Insert or update a single catch record (no full-table rewrite)."""
    _ensure_catch_db()
    engine = _catch_records_engine()
    params = _catch_record_sql_params(record)
    with engine.begin() as conn:
        conn.execute(_UPSERT_CATCH_RECORD_SQL, params)


def delete_catch_record(row_pk: str) -> None:
    """Delete one catch record by primary key."""
    _ensure_catch_db()
    engine = _catch_records_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM catch_records WHERE id = :id"), {"id": row_pk})


def save_uploaded_image(uploaded_file) -> str | None:
    """Save uploaded squid image locally or to S3-compatible storage; return stored reference."""
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    file_name = f"{uuid.uuid4().hex}{suffix}"

    if _photo_storage_enabled():
        key = f"catch_images/{file_name}"
        body = uploaded_file.getbuffer().tobytes()
        ctype = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        bucket = str(st.secrets["photo_storage_s3_bucket"]).strip()
        _s3_photo_client().put_object(
            Bucket=bucket, Key=key, Body=body, ContentType=ctype
        )
        return f"{_REMOTE_PHOTO_PREFIX}{key}"

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = IMAGE_DIR / file_name
    with file_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(file_path.as_posix())


def _photo_path_after_edit(uploaded, delete_flag: bool, previous: dict) -> str | None:
    """Resolve stored photo path after the user edits or removes the image."""
    if uploaded:
        new_path = save_uploaded_image(uploaded)
        delete_stored_photo(previous.get("photo_path"))
        return new_path
    if delete_flag:
        delete_stored_photo(previous.get("photo_path"))
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


today = date.today()
location_options = list(locations.keys())
if "point_selector" not in st.session_state:
    st.session_state.point_selector = location_options[0]

tab_forecast, tab_catch, tab_suggest = st.tabs(
    ["エギング指数・予測", "釣果ログ", "釣れる時間帯提案"]
)

with tab_forecast:
    st.radio(
        "表示するポイント",
        location_options,
        key="point_selector",
        horizontal=True,
    )
    current_point = st.session_state["point_selector"]
    st.caption(f"現在の選択: {current_point}")

    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        m = folium.Map(location=[31.3, 130.6], zoom_start=9)

        for name, point_coords in locations.items():
            try:
                location_forecast = weekly_forecast(name, locations, days=7)
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
                f"風: {format_wind_display(eval_today['wind_mps'])} / "
                f"波: {eval_today['wave_m']}m"
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
        forecast = weekly_forecast(current_point, locations, days=7)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
        st.error(
            "Open-Meteoから天候データを取得できませんでした。"
            "時間をおいて再試行してください。"
        )
        st.exception(error)
    else:
        today_result = next((fc for fc in forecast if fc["date"] == today), forecast[0])

        with col_right:
            st.subheader("指定したポイントの評価")
            st.metric("総合ランク", today_result["rank"])
            st.metric("総合スコア", f"{today_result['total_score']} / 100")
            st.write(format_point_conditions_markdown(today_result))

            detail_df = pd.DataFrame(
                [
                    {"項目": key, "スコア": value}
                    for key, value in today_result["detail"].items()
                ]
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
                    "風": format_wind_display(fc["wind_mps"]),
                    "波(m)": fc["wave_m"],
                    "水温(℃)": fc["water_temp"],
                    "気圧(hPa)": fc["pressure_hpa"],
                }
                for fc in forecast
            ]
        )
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.caption(
            "※ 風速は地上10m・気圧は Open-Meteo Forecast API（風速は m/s で取得し km/h も併記）、"
            "波高・海面水温・海面高度（潮汐を含むモデル）は Open-Meteo Marine API の無料予報です。"
            "実釣前に最新情報を再確認してください。"
        )

with tab_catch:
    st.subheader("釣果ログ（写真 + 日時 + 気象）")
    catch_eval_point = st.session_state.get("point_selector", location_options[0])

    if not RECORDS_SECTION_PASSWORD:
        st.warning(
            "記録欄パスワードが未設定です。"
            ".streamlit/secrets.toml に records_section_password を設定してください。"
        )
        _n_saved = count_catch_records()
        if _n_saved:
            st.info(
                f"{catch_records_storage_hint()} には釣果が **{_n_saved} 件**保存されています。"
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

        save_notice = st.session_state.pop("catch_save_notice", None)
        if save_notice:
            st.success(save_notice)

        record_items = load_catch_records()
        st.caption(
            "釣果ログをもとにした「釣れやすい気象条件」と「本日の釣れやすい時間帯」は"
            "「釣れる時間帯提案」タブでご覧いただけます。"
        )
        if not _photo_storage_enabled():
            st.caption(
                "釣果の「行データ」は PostgreSQL に保存されています。"
                "写真はローカルフォルダのみのため、Streamlit Cloud では再デプロイ後に画像が失われます。"
                "photo_storage_s3_* を設定すると写真も永続化できます。"
            )

        with st.form("catch_log_form", clear_on_submit=True):
            record_col1, record_col2 = st.columns(2)
            with record_col1:
                catch_location = st.selectbox(
                    "釣れたポイント", list(locations.keys()), key="catch_location"
                )
                catch_date = st.date_input("釣れた日", value=today, key="catch_date")
                catch_time = st.time_input(
                    "釣れた時刻", value=time(20, 0), key="catch_time"
                )
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
                weather_snapshot = get_weather_snapshot(
                    locations[catch_location], catch_datetime
                )
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                weather_snapshot = {
                    "wind_mps": None,
                    "wave_m": None,
                    "water_temp": None,
                    "pressure_hpa": None,
                    "sea_level_m": None,
                    CATCH_WEATHER_WIND_UNIT_KEY: "ms",
                }
            try:
                photo_path = save_uploaded_image(squid_photo)
            except ClientError as exc:
                st.error("オブジェクトストレージへの写真アップロードに失敗しました。")
                st.exception(exc)
                st.stop()
            new_record = {
                "id": uuid.uuid4().hex,
                "location": catch_location,
                "datetime": catch_datetime.isoformat(timespec="minutes"),
                "size_cm": float(squid_size),
                "count": int(squid_count),
                "memo": memo.strip(),
                "photo_path": photo_path,
                "weather": weather_snapshot,
            }
            try:
                upsert_catch_record(new_record)
            except SQLAlchemyError as exc:
                st.error("データベースへの保存に失敗しました。")
                st.exception(exc)
            else:
                st.session_state["catch_save_notice"] = (
                    "釣果ログを保存しました。"
                    f"（気象: 風 {format_wind_display(weather_snapshot.get('wind_mps'))} / "
                    f"波 {weather_snapshot.get('wave_m')} m / "
                    f"水温 {weather_snapshot.get('water_temp')} ℃）"
                )
                st.rerun()

        if record_items:
            sorted_records = sorted(
                record_items, key=lambda log: log["datetime"], reverse=True
            )

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
                        "釣れたポイント",
                        loc_keys,
                        index=loc_index,
                        key=f"ed_loc_{edit_idx}",
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
                    existing_preview = photo_display_url(rec_before.get("photo_path"))
                    if existing_preview:
                        st.caption("現在の写真")
                        st.image(existing_preview, width=280)
                    ed_photo = st.file_uploader(
                        "新しい写真に差し替え（未選択ならそのまま）",
                        type=["jpg", "jpeg", "png", "webp"],
                        key=f"ed_photo_{edit_idx}",
                    )
                    delete_photo_on_edit = st.checkbox(
                        "写真を削除する（ローカル／オブジェクトストレージからも削除）",
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
                            weather_snapshot = get_weather_snapshot(
                                locations[ed_location], catch_dt
                            )
                        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                            weather_snapshot = {
                                "wind_mps": None,
                                "wave_m": None,
                                "water_temp": None,
                                "pressure_hpa": None,
                                "sea_level_m": None,
                                CATCH_WEATHER_WIND_UNIT_KEY: "ms",
                            }
                        old = record_items[store_idx]
                        try:
                            photo_stored = _photo_path_after_edit(
                                ed_photo, delete_photo_on_edit, old
                            )
                        except ClientError as exc:
                            st.error(
                                "オブジェクトストレージへの写真アップロードに失敗しました。"
                            )
                            st.exception(exc)
                            st.stop()
                        record_id = old.get("id") or uuid.uuid4().hex
                        updated = {
                            "id": record_id,
                            "location": ed_location,
                            "datetime": catch_dt.isoformat(timespec="minutes"),
                            "size_cm": float(ed_size),
                            "count": int(ed_count),
                            "memo": ed_memo.strip(),
                            "photo_path": photo_stored,
                            "weather": weather_snapshot,
                        }
                        record_items[store_idx] = updated
                        try:
                            upsert_catch_record(updated)
                        except SQLAlchemyError as exc:
                            st.error("データベースへの保存に失敗しました。")
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
                    rid = target_record.get("id")
                    if not rid:
                        st.error(
                            "削除できません（ログに ID がありません）。"
                            "一覧を再表示してから再度お試しください。"
                        )
                    else:
                        if delete_photo_file and target_photo_path:
                            delete_stored_photo(target_photo_path)
                        try:
                            delete_catch_record(rid)
                        except SQLAlchemyError as exc:
                            st.error("データベースからの削除に失敗しました。")
                            st.exception(exc)
                        else:
                            st.success("ログを削除しました。")
                            st.rerun()

            filter_col1, filter_col2 = st.columns([1.1, 1.4])
            with filter_col1:
                date_filter_enabled = st.checkbox("日付で絞り込む", value=False)
            with filter_col2:
                filter_date = st.date_input(
                    "表示する日付", value=today, disabled=not date_filter_enabled
                )

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
                            "風": format_wind_display(hist["weather"].get("wind_mps")),
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
                    img_src = photo_display_url(hist.get("photo_path"))
                    if not img_src:
                        continue
                    with photo_cols[photo_idx % 3]:
                        st.image(
                            img_src,
                            caption=(
                                f"{hist['location']} "
                                f"{hist['datetime'].replace('T', ' ')}"
                            ),
                        )
                    photo_idx += 1

with tab_suggest:
    st.subheader("釣果ログから探る、釣れる時間帯")
    suggest_point = st.session_state.get("point_selector", location_options[0])
    st.caption(
        f"対象ポイント: {suggest_point}"
        "（「エギング指数・予測」タブのポイント選択と連動しています）"
    )

    if not RECORDS_SECTION_PASSWORD or not st.session_state.get("records_auth_unlocked"):
        st.info(
            "この機能は釣果ログのデータをもとに算出します。"
            "「釣果ログ」タブでログインすると表示されます。"
        )
    else:
        suggest_records = load_catch_records()
        favorable = derive_favorable_conditions(suggest_point, suggest_records)

        if favorable is None:
            st.info(
                f"{suggest_point} の釣果ログが各指標2件以上あると、"
                "釣れやすい条件を分析できます。"
            )
        else:
            st.markdown("#### 過去の釣果から見える「釣れやすい気象条件」")
            field_labels = {
                "wind_mps": "風速 (m/s)",
                "wave_m": "波高 (m)",
                "water_temp": "水温 (℃)",
                "pressure_hpa": "気圧 (hPa)",
            }
            favorable_rows = [
                {
                    "指標": label,
                    "釣果時の平均": round(favorable[field]["mean"], 1),
                    "目安レンジ": (
                        f"{round(favorable[field]['low'], 1)} 〜 "
                        f"{round(favorable[field]['high'], 1)}"
                    ),
                    "サンプル数": favorable[field]["n"],
                }
                for field, label in field_labels.items()
                if field in favorable
            ]
            st.dataframe(
                pd.DataFrame(favorable_rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "目安レンジは釣果ログの平均 ± ばらつき（標準偏差）です。"
                "件数が少ないうちは参考程度にご覧ください。"
            )

            st.markdown("#### 本日の時間帯別 釣れやすさ")
            try:
                hourly_scores = suggest_best_hours(
                    suggest_point, locations[suggest_point], today, suggest_records
                )
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
                st.error("本日の時間別データを取得できませんでした。")
                st.exception(error)
            else:
                if not hourly_scores:
                    st.info("時間帯別の提案に必要な釣果ログが不足しています。")
                else:
                    windows = summarize_best_windows(hourly_scores)
                    if windows:
                        best_window = windows[0]
                        st.success(
                            "本日のおすすめ時間帯: "
                            f"**{best_window['start'].strftime('%H:%M')}〜"
                            f"{best_window['end'].strftime('%H:%M')}**"
                            f"（釣果ログとの類似度 平均 {best_window['avg_score']} 点）"
                        )
                        if len(windows) > 1:
                            others = "、".join(
                                f"{w['start'].strftime('%H:%M')}〜"
                                f"{w['end'].strftime('%H:%M')}（{w['avg_score']}点）"
                                for w in windows[1:3]
                            )
                            st.caption(f"次点の時間帯: {others}")

                    hourly_df = pd.DataFrame(
                        [
                            {
                                "時刻": row["time"].strftime("%H:%M"),
                                "釣れやすさスコア": row["score"],
                                "風": format_wind_display(row["wind_mps"]),
                                "波(m)": row["wave_m"],
                                "水温(℃)": row["water_temp"],
                                "気圧(hPa)": row["pressure_hpa"],
                            }
                            for row in hourly_scores
                        ]
                    )
                    st.line_chart(hourly_df.set_index("時刻")["釣れやすさスコア"])
                    st.dataframe(
                        hourly_df, use_container_width=True, hide_index=True
                    )
                    st.caption(
                        "※ 釣れやすさスコアは、その時間帯の予報が釣果ログの気象条件に"
                        "どれだけ近いかをもとにした目安です。潮汐や時合いなど、"
                        "他の要因も合わせてご判断ください。"
                    )
