"""
Streamlit Dashboard — He thong du bao thoi tiet (WEATHER-5K)
Chay: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import json
import os
import glob

# ── Config ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "WEATHER-5K")
STATIONS_DIR = os.path.join(DATA_DIR, "global_weather_stations")
META_FILE = os.path.join(DATA_DIR, "meta_info.json")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_weather5k")
METRICS_FILE = os.path.join(CKPT_DIR, "metrics.json")

st.set_page_config(
    page_title="Weather Forecast — WEATHER-5K",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        color: white; margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .metric-card h2 { margin: 0; font-size: 2rem; font-weight: 700; }
    .metric-card p { margin: 4px 0 0 0; font-size: 0.85rem; opacity: 0.85; }
    .card-blue { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
    .card-green { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .card-orange { background: linear-gradient(135deg, #f2994a, #f2c94c); }
    .card-red { background: linear-gradient(135deg, #eb3349, #f45c43); }
    .section-title {
        font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #667eea; padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #f8f9fa; border-radius: 10px; padding: 12px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────
@st.cache_data
def load_meta_info():
    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_station_data(station_file):
    fpath = os.path.join(STATIONS_DIR, station_file)
    df = pd.read_csv(fpath, parse_dates=["DATE"])
    df = df[df["MASK"] == "[1 1 1 1 1]"].copy()
    df.rename(columns={
        "DATE": "datetime", "TMP": "Temperature", "DEW": "DewPoint",
        "WND_ANGLE": "WindDirection", "WND_RATE": "WindSpeed", "SLP": "Pressure",
    }, inplace=True)
    return df


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None


try:
    meta_info = load_meta_info()
    metrics = load_metrics()
    station_files = sorted(meta_info.keys())
    total_stations = len(station_files)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛅ Weather Forecast")
    st.markdown("**WEATHER-5K + Spark MLlib**")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(f"- **{total_stations:,}** trạm thời tiết")
    st.markdown(f"- **~87,000** dòng/trạm")
    st.markdown(f"- **2014–2023** (hourly)")
    st.markdown(f"- **5** biến: TMP, DEW, WND, SLP")
    st.markdown("---")
    if metrics:
        st.success("Pipeline đã chạy xong")
        st.markdown(f"Train: **{metrics.get('train_count', '?'):,}**")
        st.markdown(f"Test: **{metrics.get('test_count', '?'):,}**")
    else:
        st.warning("Chưa chạy pipeline")
        st.caption("Chạy: python src/pipeline.py")
    st.markdown("---")
    st.caption("Đồ án: Các công cụ & nền tảng TTNT")


# ── Tabs ─────────────────────────────────────────────────────
tab_overview, tab_eda, tab_clf, tab_reg, tab_pipeline = st.tabs([
    "📊 Tổng quan",
    "📈 Phân tích dữ liệu",
    "🌧️ Classification",
    "🌡️ Regression",
    "⚙️ Pipeline",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: TONG QUAN
# ═══════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("## Tổng quan hệ thống")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card card-blue">
        <h2>{total_stations:,}</h2><p>Trạm thời tiết</p></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card card-green">
        <h2>~493M</h2><p>Tổng quan sát (full)</p></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card card-orange">
        <h2>5</h2><p>Biến thời tiết</p></div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card card-red">
        <h2>6</h2><p>Models</p></div>""", unsafe_allow_html=True)

    # Map tram
    st.markdown('<div class="section-title">Bản đồ trạm thời tiết</div>', unsafe_allow_html=True)

    REGIONS = {
        "🌍 Toàn cầu":      {"lat": (-90, 90),   "lon": (-180, 180), "scope": None},
        "🌏 Châu Á":         {"lat": (-10, 75),   "lon": (25, 180),   "scope": "asia"},
        "🌍 Châu Âu":        {"lat": (35, 72),    "lon": (-25, 50),   "scope": "europe"},
        "🌎 Bắc Mỹ":        {"lat": (10, 85),    "lon": (-170, -50), "scope": "north america"},
        "🌎 Nam Mỹ":         {"lat": (-60, 15),   "lon": (-90, -30),  "scope": "south america"},
        "🌍 Châu Phi":       {"lat": (-35, 38),   "lon": (-20, 55),   "scope": "africa"},
        "🌏 Châu Đại Dương": {"lat": (-50, 5),    "lon": (100, 180),  "scope": None},
    }

    sel_region = st.selectbox("🗺️ Chọn khu vực", list(REGIONS.keys()), index=0)
    region = REGIONS[sel_region]

    map_data = []
    for fname, info in meta_info.items():
        lat = info.get("latitude", 0)
        lon = info.get("longitude", 0)
        if lat == 0 and lon == 0:
            continue
        lat_range = region["lat"]
        lon_range = region["lon"]
        if not (lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]):
            continue
        map_data.append({
            "Station": fname.replace(".csv", ""),
            "latitude": lat,
            "longitude": lon,
            "elevation": info.get("ELEVATION", 0),
            "valid_percent": round(info.get("valid_percent", 0) * 100, 1),
        })
    map_df = pd.DataFrame(map_data)

    if len(map_df) == 0:
        st.info("Không có trạm nào trong khu vực này.")
    else:
        st.caption(f"Hiển thị **{len(map_df):,}** / {total_stations:,} trạm")

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lat=map_df["latitude"],
            lon=map_df["longitude"],
            mode="markers",
            marker=dict(
                size=6,
                color=map_df["valid_percent"],
                colorscale=[
                    [0.0, "#ff4757"],
                    [0.5, "#ffa502"],
                    [0.75, "#2ed573"],
                    [1.0, "#7bed9f"],
                ],
                cmin=50, cmax=100,
                opacity=0.9,
                line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                colorbar=dict(
                    title=dict(text="Valid %", font=dict(color="white", size=12)),
                    tickfont=dict(color="white", size=10),
                    len=0.5, thickness=15,
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    x=1.02,
                ),
            ),
            text=[f"<b>{s}</b><br>Lat: {la:.2f}° | Lon: {lo:.2f}°<br>"
                  f"Elev: {el}m | Valid: {vp}%"
                  for s, la, lo, el, vp in zip(
                      map_df["Station"], map_df["latitude"],
                      map_df["longitude"], map_df["elevation"],
                      map_df["valid_percent"])],
            hoverinfo="text",
        ))

        scope = region["scope"]
        geo_args = dict(
            showcoastlines=True, coastlinecolor="rgba(120,120,160,0.5)",
            coastlinewidth=1,
            showland=True, landcolor="#16213e",
            showocean=True, oceancolor="#0a1128",
            showlakes=True, lakecolor="#0a1128",
            showcountries=True, countrycolor="rgba(100,100,140,0.4)",
            countrywidth=0.5,
            showframe=False,
            projection_type="natural earth",
        )
        if scope:
            geo_args["scope"] = scope

        fig.update_geos(**geo_args)
        fig.update_layout(
            height=550,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Bài toán Classification</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Target:** `Rain_next` — giờ tới có mưa không?
        - **Label:** Dew Point Depression < 2.5°C → Rain = 1
        - **Models:** Logistic Regression, Random Forest, GBT
        - **Metrics:** Accuracy, Precision, Recall, F1, AUC
        """)
    with col2:
        st.markdown('<div class="section-title">Bài toán Regression</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Target:** `Temp_next` — nhiệt độ giờ tới (°C)
        - **Label:** `lead(Temperature, 1)` — shift 1 giờ
        - **Models:** Linear Regression, Random Forest, GBT
        - **Metrics:** RMSE, MAE, R²
        """)


# ═══════════════════════════════════════════════════════════════
# TAB 2: EDA
# ═══════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("## Phân tích dữ liệu (EDA)")

    sel_station = st.selectbox("Chọn trạm (Station ID)", station_files[:200],
                               index=0)

    try:
        sdf = load_station_data(sel_station)
    except Exception as e:
        st.error(f"Lỗi load station: {e}")
        st.stop()

    station_info = meta_info.get(sel_station, {})
    st.markdown(f"**Vị trí:** lat={station_info.get('latitude', '?')}, "
                f"lon={station_info.get('longitude', '?')}, "
                f"elevation={station_info.get('ELEVATION', '?')}m | "
                f"**Valid:** {station_info.get('valid_percent', 0)*100:.1f}% | "
                f"**Rows:** {len(sdf):,}")

    # Row 1: Temp + DewPoint
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="section-title">Nhiệt độ (°C)</div>', unsafe_allow_html=True)
        t = sdf[["datetime", "Temperature"]].dropna()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(t["datetime"], t["Temperature"], alpha=0.3, color="#e74c3c")
        ax.plot(t["datetime"], t["Temperature"], linewidth=0.3, color="#e74c3c")
        ax.set_ylabel("°C", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f'<div class="section-title">Dew Point (°C)</div>', unsafe_allow_html=True)
        d = sdf[["datetime", "DewPoint"]].dropna()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(d["datetime"], d["DewPoint"], alpha=0.3, color="#3498db")
        ax.plot(d["datetime"], d["DewPoint"], linewidth=0.3, color="#3498db")
        ax.set_ylabel("°C", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2: Pressure + Wind
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="section-title">Áp suất (hPa)</div>', unsafe_allow_html=True)
        p = sdf[["datetime", "Pressure"]].dropna()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(p["datetime"], p["Pressure"], linewidth=0.3, color="#f39c12")
        ax.set_ylabel("hPa", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(f'<div class="section-title">Tốc độ gió (m/s)</div>', unsafe_allow_html=True)
        w = sdf[["datetime", "WindSpeed"]].dropna()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(w["datetime"], w["WindSpeed"], linewidth=0.3, color="#2ecc71")
        ax.set_ylabel("m/s", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3: Dew Point Depression distribution
    st.markdown(f'<div class="section-title">Phân bố Dew Point Depression</div>', unsafe_allow_html=True)
    sdf["DPD"] = sdf["Temperature"] - sdf["DewPoint"]
    dpd = sdf["DPD"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(dpd, bins=80, color="#667eea", edgecolor="white", linewidth=0.3)
    ax.axvline(2.5, color="#e74c3c", linestyle="--", linewidth=2,
               label="Rain threshold (2.5°C)")
    rain_pct = (dpd < 2.5).sum() / len(dpd) * 100
    ax.set_xlabel("Temperature - DewPoint (°C)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title(f"Rain ratio: {rain_pct:.1f}% (DPD < 2.5°C)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Row 4: Temp distribution
    st.markdown('<div class="section-title">Phân bố nhiệt độ trạm</div>', unsafe_allow_html=True)
    all_t = sdf["Temperature"].dropna().values
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.hist(all_t, bins=80, color="#667eea", edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(all_t), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(all_t):.1f}°C")
    ax.set_xlabel("°C", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 3: CLASSIFICATION RESULTS
# ═══════════════════════════════════════════════════════════════
with tab_clf:
    st.markdown("## 🌧️ Classification — Dự đoán Rain_next")
    st.markdown("Giờ tiếp theo có mưa không? (Dew Point Depression < 2.5°C)")

    if metrics and "classification" in metrics:
        clf = metrics["classification"]

        best_name = max(clf, key=lambda x: clf[x].get("AUC-ROC", clf[x].get("AUC", 0)))
        best_auc = clf[best_name].get("AUC-ROC", clf[best_name].get("AUC", 0))
        st.success(f"**Best Model (AUC): {best_name}** — AUC = {best_auc:.4f}")

        cols = st.columns(len(clf))
        card_colors = ["card-blue", "card-green", "card-red"]
        for i, (name, m) in enumerate(clf.items()):
            with cols[i]:
                color = card_colors[i % len(card_colors)]
                acc = m.get("Accuracy", 0)
                st.markdown(f"""<div class="metric-card {color}">
                    <h2>{acc:.2%}</h2><p>{name}</p></div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Chi tiết metrics</div>', unsafe_allow_html=True)
        rows = []
        for name, m in clf.items():
            row = {"Model": f"⭐ {name}" if name == best_name else name}
            row.update({k: f"{v:.4f}" for k, v in m.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">So sánh models</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        models = list(clf.keys())
        x = np.arange(len(models))
        w = 0.15
        metric_names = list(list(clf.values())[0].keys())
        cmap = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))
        for i, mn in enumerate(metric_names):
            vals = [clf[m].get(mn, 0) for m in models]
            bars = ax.bar(x + i * w, vals, w, label=mn, color=cmap[i], edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + w * (len(metric_names)-1) / 2)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0.5, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Chưa có kết quả. Chạy pipeline trước.")
        st.code("cd weather_forecast\npython src/pipeline.py", language="bash")


# ═══════════════════════════════════════════════════════════════
# TAB 4: REGRESSION RESULTS
# ═══════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("## 🌡️ Regression — Dự đoán Temp_next")
    st.markdown("Nhiệt độ (°C) ở giờ tiếp theo")

    if metrics and "regression" in metrics:
        reg = metrics["regression"]

        best_name = min(reg, key=lambda x: reg[x].get("RMSE", 999))
        st.success(f"**Best Model (RMSE): {best_name}** — RMSE = {reg[best_name]['RMSE']:.4f}°C")

        cols = st.columns(len(reg))
        card_colors = ["card-blue", "card-green", "card-red"]
        for i, (name, m) in enumerate(reg.items()):
            with cols[i]:
                color = card_colors[i % len(card_colors)]
                rmse = m.get("RMSE", 0)
                st.markdown(f"""<div class="metric-card {color}">
                    <h2>{rmse:.2f}°C</h2><p>{name}<br>RMSE</p></div>""",
                            unsafe_allow_html=True)

        st.markdown('<div class="section-title">Chi tiết metrics</div>', unsafe_allow_html=True)
        rows = []
        for name, m in reg.items():
            row = {"Model": f"⭐ {name}" if name == best_name else name}
            row.update({k: f"{v:.4f}" for k, v in m.items()})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">So sánh models</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        models = list(reg.keys())
        palette = ["#667eea", "#2ecc71", "#e74c3c"]

        for idx, (metric, title, better) in enumerate([
            ("RMSE", "RMSE (°C)", "lower"),
            ("MAE", "MAE (°C)", "lower"),
            ("R2", "R²", "higher"),
        ]):
            vals = [reg[m].get(metric, 0) for m in models]
            bars = axes[idx].bar(models, vals, color=palette, edgecolor="white", linewidth=1.5)
            for bar, val in zip(bars, vals):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            axes[idx].set_title(title, fontsize=12, fontweight="600")
            axes[idx].spines[["top", "right"]].set_visible(False)
            axes[idx].grid(axis="y", alpha=0.2)
            best_idx = vals.index(min(vals)) if better == "lower" else vals.index(max(vals))
            bars[best_idx].set_edgecolor("#f39c12")
            bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Chưa có kết quả. Chạy pipeline trước.")
        st.code("cd weather_forecast\npython src/pipeline.py", language="bash")


# ═══════════════════════════════════════════════════════════════
# TAB 5: PIPELINE
# ═══════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("## ⚙️ Pipeline Architecture")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        DATA INGESTION                               │
    │  WEATHER-5K: 5,672 station CSVs (2014–2023, hourly)                │
    │  Columns: DATE, TMP, DEW, WND_ANGLE, WND_RATE, SLP, MASK          │
    │                    ↓ Sample N stations (default 50)                 │
    │                    ↓ Filter MASK = [1 1 1 1 1]                     │
    │                    ↓ Union all → ~4.3M rows × 7 columns            │
    └─────────────────────────┬───────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      PREPROCESSING                                  │
    │  ✓ Rain label (DewPoint Depression < 2.5°C → Rain = 1)            │
    │  ✓ Targets: Rain_next, Temp_next (lead 1 hour)                    │
    │  ✓ Fill missing (median)                                           │
    └─────────────────────────┬───────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   FEATURE ENGINEERING                               │
    │  ✓ Time: Hour, Month, DayOfWeek + sin/cos encoding                │
    │  ✓ Lag: Temp 1/3/6/24h, DewPoint/Pressure/Wind 1h, Rain 1/3h    │
    │  ✓ Rolling: avg 6h/24h, Rain_sum 6h/24h                          │
    │  ✓ Derived: TempChange, DewPointChange, PressureChange            │
    └─────────────────────────┬───────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     ML PIPELINE (Spark MLlib)                       │
    │  StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler │
    │                              ↓                                     │
    │  Classification          Regression                                │
    │  ├─ LogisticRegression   ├─ LinearRegression                      │
    │  ├─ RandomForest         ├─ RandomForest                          │
    │  └─ GBTClassifier        └─ GBTRegressor                          │
    └─────────────────────────┬───────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      EVALUATION                                     │
    │  Classification: Accuracy, Precision, Recall, F1, AUC-ROC         │
    │  Regression:     RMSE, MAE, R²                                    │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Feature Engineering</div>', unsafe_allow_html=True)
        features = {
            "Loại": ["Numeric", "Time", "Lag", "Rolling", "Derived", "Categorical"],
            "Features": [
                "Temperature, DewPoint, Pressure, WindSpeed, WindDirection",
                "Hour, Month, DayOfWeek, DayOfYear + sin/cos",
                "Temp 1/3/6/24h, DewPoint/Pressure/Wind 1h, Rain 1/3h",
                "Temp/DewPoint/Pressure avg 6h/24h, Rain_sum 6h/24h",
                "TempChange, DewPointChange, PressureChange, TempDewPoint",
                "StationID (OneHotEncoded)",
            ],
            "Số lượng": [5, 8, 9, 6, 4, 1],
        }
        st.dataframe(pd.DataFrame(features), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Checkpoint</div>', unsafe_allow_html=True)
        ckpts = {
            "File": [
                "processed_data.parquet",
                "train.parquet",
                "test.parquet",
                "models/clf_*",
                "models/reg_*",
                "metrics.json",
            ],
            "Nội dung": [
                "Data sau preprocessing + FE",
                "Train split (80%)",
                "Test split (20%)",
                "3 classification models",
                "3 regression models",
                "Kết quả đánh giá (cho dashboard)",
            ],
            "Status": [
                "✅" if os.path.exists(os.path.join(CKPT_DIR, "processed_data.parquet")) else "❌",
                "✅" if os.path.exists(os.path.join(CKPT_DIR, "train.parquet")) else "❌",
                "✅" if os.path.exists(os.path.join(CKPT_DIR, "test.parquet")) else "❌",
                "✅" if os.path.exists(os.path.join(CKPT_DIR, "models")) else "❌",
                "✅" if os.path.exists(os.path.join(CKPT_DIR, "models")) else "❌",
                "✅" if os.path.exists(METRICS_FILE) else "❌",
            ],
        }
        st.dataframe(pd.DataFrame(ckpts), use_container_width=True, hide_index=True)
