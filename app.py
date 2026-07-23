import calendar
from datetime import date, datetime
from pathlib import Path
import pytz
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    METEO_TERSEDIA = True
except ImportError:
    METEO_TERSEDIA = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CabaiCast Analytics",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_ss("dark_mode", True)
_ss("data_page", 0)
_ss("pred_result", None)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (identical to original)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
LAT, LON = -6.9175, 107.6191

COL_IKLIM = [
    "Suhu Max (°C)", "Suhu Min (°C)", "Suhu Mean (°C)", "Curah Hujan (mm)",
    "Durasi Hujan (jam)", "Kecepatan Angin Maks (km/h)",
    "Radiasi Matahari (MJ/m²)", "Evapotranspirasi ET0 (mm)",
]
NAMA_BULAN = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}
MOMENTUM_COLS = [
    "lag_2", "lag_3", "lag_5", "lag_7", "lag_14", "lag_30",
    "delta_1_3", "delta_1_7", "delta_7_14", "delta_7_30", "momentum_5",
    "bulan", "tahun", "hari_index",
]

# ══════════════════════════════════════════════════════════════════════════════
# THEME  (light / dark)
# ══════════════════════════════════════════════════════════════════════════════
_LIGHT = dict(
    bg="#F7F7FB", sidebar="#FFFFFF", card="#FFFFFF", card2="#EFECFD",
    text="#1F2937", muted="#6B7280", border="#E5E7EB",
    shadow="rgba(0,0,0,0.06)", primary="#7C5CFC", secondary="#A78BFA",
    pri_rgb="124,92,252", pri_light="rgba(124,92,252,0.08)",
    green="#059669", red="#DC2626", grid="#F3F4F6",
    plot_tpl="plotly_white", inp_bg="#F7F7FB",
)
_HYBRID = dict(
    bg="#F6F5FB", sidebar="#14122B", card="#FFFFFF", card2="#F0EEFA",
    text="#1E1B33", muted="#6B7280", border="#ECEAF5",
    shadow="rgba(0,0,0,0.06)", primary="#6D5CE0", secondary="#9B8AFA",
    pri_rgb="109,92,224", pri_light="rgba(109,92,224,0.10)",
    green="#22C55E", red="#EF4444", grid="#F0EEFA",
    plot_tpl="plotly_white", inp_bg="#F6F5FB",
    blue="#3B82F6", orange="#F97316",
    sb_text="#F9FAFB", sb_muted="#9CA3AF", sb_border="#2D2A45",
)

def T():
    return _HYBRID

# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    t = T()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
*, *::before, *::after {{
    box-sizing: border-box;
}}
/* ── Global ─────────────────────────────────────── */
.stApp {{ background-color: {t['bg']} !important; }}
.main .block-container {{
    padding: 0 2rem 3rem 2rem !important;
    max-width: 100% !important;
}}

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {t['sidebar']} !important;
    border-right: 1px solid {t['sb_border']} !important;
    transition: margin-left 0.35s ease, width 0.35s ease, transform 0.35s ease !important;
}}

/* ── Fade blur di tepi atas/bawah sidebar saat scroll ── */
[data-testid="stSidebar"] > div:first-child {{
    position: relative;
}}
[data-testid="stSidebar"]::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 40px;
    background: linear-gradient(to bottom, {t['sidebar']} 0%, transparent 100%);
    pointer-events: none;
    z-index: 10;
}}
[data-testid="stSidebar"]::after {{
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 40px;
    background: linear-gradient(to top, {t['sidebar']} 0%, transparent 100%);
    pointer-events: none;
    z-index: 10;
}}

/* Nav radio → pill menu */
[data-testid="stHeader"] {{
    background: transparent !important;
}}
[data-testid="stSidebar"] .stRadio > label {{ display: none !important; }}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {{ gap: 0 !important; }}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {{
    display: flex !important; align-items: center !important;
    padding: 10px 16px !important; border-radius: 12px !important;
    margin: 1px 8px !important; cursor: pointer !important;
    transition: all 0.18s ease !important; color: {t['sb_muted']} !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    background: transparent !important; border: none !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {{
    background: {t['pri_light']} !important; color: {t['primary']} !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {{
    background: {t['primary']} !important; color: #FFFFFF !important;
}}

/* ── Sembunyikan bulatan radio indicator (fix tampilan deploy) ── */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {{
    display: none !important;
}}
[data-testid="stSidebar"] .stRadio label svg,
[data-testid="stSidebar"] .stRadio label input[type="radio"] {{
    display: none !important;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > span:first-of-type + span {{
    display: none !important;
}}
/* ── Cards ────────────────────────────────────────── */
.card {{
    background: {t['card']}; border-radius: 22px;
    padding: 1.5rem; border: 1px solid {t['border']};
    box-shadow: 0 2px 20px {t['shadow']};
    transition: all 0.22s ease; margin-bottom: 1rem;
}}
div[class*="st-key-card_"] {{
    background: {t['card']}; border-radius: 22px !important;
    padding: 1.5rem !important; border: 1px solid {t['border']};
    box-shadow: 0 2px 20px {t['shadow']};
    margin-bottom: 1rem;
}}
div[class*="st-key-card_"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent; border: none; box-shadow: none; padding: 0;
}}
.card-sm {{
    background: {t['card']}; border-radius: 16px;
    padding: 1rem 1.2rem; border: 1px solid {t['border']};
    box-shadow: 0 2px 12px {t['shadow']}; transition: all 0.22s ease;
}}
.card-sm:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba({t['pri_rgb']},0.15);
}}

/* ── Metric Cards ─────────────────────────────────── */
.m-card {{
    background: {t['card']}; border-radius: 20px;
    padding: 1.1rem 1.3rem; border: 1px solid {t['border']};
    box-shadow: 0 2px 16px {t['shadow']};
    display: flex; align-items: center; gap: 0.9rem;
    transition: all 0.22s ease; height: 88px;
}}
.m-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba({t['pri_rgb']},0.18);
    border-color: rgba({t['pri_rgb']},0.35);
}}
.m-icon {{
    width: 48px; height: 48px;
    background: rgba({t['pri_rgb']},0.12); border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}}
.m-val {{
    font-size: 1.45rem; font-weight: 800; color: {t['text']};
    letter-spacing: -0.5px; line-height: 1.15;
}}
.m-lbl {{
    font-size: 0.72rem; font-weight: 600; color: {t['muted']};
    text-transform: uppercase; letter-spacing: 0.7px; margin-top: 2px;
}}

/* ── Sidebar Logo ─────────────────────────────────── */
.sb-logo {{
    padding: 1.2rem 1.2rem 0.8rem;
    border-bottom: 1px solid {t['sb_border']};
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 10px;
}}
.sb-logo-icon {{
    width: 36px; height: 36px;
    background: #6D5CE0;
    border-radius: 10px; display: flex;
    align-items: center; justify-content: center; font-size: 1.1rem;
    flex-shrink: 0;
}}
.sb-logo-name {{ font-size: 1.15rem; font-weight: 700; color: {t['sb_text']}; }}
.sb-logo-sub {{ font-size: 0.85rem; color: {t['sb_muted']}; }}
.sb-sect-lbl {{
    font-size: 0.62rem; font-weight: 700; color: {t['sb_muted']};
    text-transform: uppercase; letter-spacing: 1.3px;
    padding: 0 1.4rem; margin: 0.8rem 0 0.2rem 0; display: block;
}}
.sb-div {{ height: 1px; background: {t['sb_border']}; margin: 0.7rem 1.2rem; }}

/* ── Typography ───────────────────────────────────── */
.sec-title {{
    font-size: 0.95rem; font-weight: 700;
    color: {t['text']}; letter-spacing: -0.3px;
}}
.sec-badge {{
    font-size: 0.67rem; font-weight: 700; color: {t['primary']};
    background: {t['pri_light']}; padding: 3px 9px; border-radius: 20px;
}}

/* ── Model Summary Item ───────────────────────────── */
.mi {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid {t['border']};
}}
.mi:last-child {{ border-bottom: none; }}
.mi-lbl {{
    display: flex; align-items: center; gap: 7px;
    font-size: 0.82rem; color: {t['muted']}; font-weight: 500;
}}
.mi-icon {{
    width: 24px; height: 24px; background: {t['pri_light']}; border-radius: 7px;
    display: flex; align-items: center; justify-content: center; font-size: 0.75rem;
}}
.mi-val {{ font-size: 0.875rem; font-weight: 700; color: {t['text']}; }}

/* ── Header ───────────────────────────────────────── */
.hdr-date {{
    font-size: 0.78rem; color: {t['text']}; font-weight: 700;
    padding: 7px 13px; background: {t['card']};
    border-radius: 11px; border: 1px solid {t['border']};
    display: inline-flex; align-items: center; width: fit-content;
    margin-top: 1.5rem;
}}
.hdr-avatar {{
    width: 34px; height: 34px;
    background: #6D5CE0;
    border-radius: 11px; display: flex;
    align-items: center; justify-content: center;
    color: white; font-size: 0.8rem; font-weight: 700;
    margin-top: 1.5rem;
}}

/* ── Buttons ──────────────────────────────────────── */
div[data-testid="stButton"] > button {{
    background: #6D5CE0 !important;
    color: #FFF !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba({t['pri_rgb']},0.35) !important;
    transition: all 0.25s ease !important;
}}
div[data-testid="stButton"] > button:hover {{
    box-shadow: 0 8px 22px rgba({t['pri_rgb']},0.52) !important;
    transform: translateY(-2px) !important;
}}

div[data-testid="stDownloadButton"] > button {{
    background: #6D5CE0 !important;
    color: #FFF !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba({t['pri_rgb']},0.35) !important;
    transition: all 0.25s ease !important;
}}
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:active,
div[data-testid="stDownloadButton"] > button:focus {{
    background: #6D5CE0 !important;
    color: #FFF !important;
    box-shadow: 0 8px 22px rgba({t['pri_rgb']},0.52) !important;
}}

/* ── Inputs ───────────────────────────────────────── */
.stTextInput [data-baseweb="input"], .stNumberInput [data-baseweb="input"] {{
    background-color: {t['inp_bg']} !important;
    border: 2px solid #D1CCE8 !important;
    border-radius: 12px !important;
}}
.stTextInput [data-baseweb="base-input"] {{
    background-color: {t['inp_bg']} !important;
    border: none !important;
}}
.stTextInput input, .stNumberInput input,
.stSelectbox input, .stDateInput input {{
    color: var(--text-color) !important;
    background-color: transparent !important;
}}
.stTextInput input::placeholder {{
    color: {t['muted']} !important;
    opacity: 1 !important;
}}
.stTextInput svg {{
    color: {t['muted']} !important;
}}

.stSelectbox label, .stTextInput label, .stNumberInput label {{
    color: {t['muted']} !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}}

/* ── Table ────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border-radius: 14px !important; overflow: hidden !important;
    border: 1px solid {t['border']} !important;
}}

/* ── Insight box ──────────────────────────────────── */
.insight {{
    background: {t['pri_light']}; border-left: 3px solid #7C5CFC;
    border-radius: 0 14px 14px 0; padding: 0.9rem 1.1rem; margin: 0.8rem 0;
}}
.insight-ttl {{
    font-size: 0.67rem; font-weight: 700; color: #7C5CFC;
    text-transform: uppercase; letter-spacing: 1.1px; margin-bottom: 4px;
}}
.insight-txt {{ font-size: 0.855rem; color: {t['text']}; line-height: 1.6; }}

/* ── Result gradient card ─────────────────────────── */
.result-card {{
    background: #6D5CE0;
    border-radius: 20px; padding: 1.8rem; color: white; text-align: center;
}}
.result-val {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -1.5px; }}
.result-lbl {{ font-size: 0.85rem; opacity: 0.85; margin-top: 5px; }}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {t['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {t['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #7C5CFC; }}

/* ── Divider ──────────────────────────────────────── */
hr {{ border-color: {t['border']} !important; margin: 1.2rem 0 !important; }}

/* ── Element toolbar (fullscreen/download icons on charts) ── */
[data-testid="stElementToolbar"] {{
    background: transparent !important;
    box-shadow: none !important;
}}

/* ── FORCE sidebar text visibility regardless of Streamlit light/dark theme ── */
[data-testid="stSidebar"] * {{
    color: {t['sb_text']} !important;
}}
[data-testid="stSidebar"] .sb-logo-sub,
[data-testid="stSidebar"] .sb-sect-lbl,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {{
    color: {t['sb_muted']} !important;
}}
/* radio pill labels tetap ikut aturan hover/checked yang sudah ada */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {{
    color: {t['sb_muted']} !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {{
    color: {t['primary']} !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {{
    color: #FFFFFF !important;
}}
/* input/selectbox di dalam sidebar (kalau ada) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] * {{
    color: {t['sb_text']} !important;
}}
[data-testid="stSidebar"] svg {{
    stroke: {t['sb_text']} !important;
}}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def metric_card(icon: str, value: str, label: str, accent: str = None) -> str:
    icon_html = f'<div class="m-icon">{icon}</div>' if icon else ""
    dot = f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{accent};margin-right:5px;"></span>' if accent else ""
    return f'<div class="m-card">{icon_html}<div><div class="m-val">{value}</div><div class="m-lbl">{dot}{label}</div></div></div>'


def model_item(icon: str, label: str, value: str) -> str:
    icon_html = f'<div class="mi-icon">{icon}</div>' if icon else ""
    return f'<div class="mi"><div class="mi-lbl">{icon_html}{label}</div><div class="mi-val">{value}</div></div>'


def sec_header(title: str, badge: str = None) -> str:
    b = f'<span class="sec-badge">{badge}</span>' if badge else ""
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.9rem;">'
        f'<span class="sec-title">{title}</span>{b}</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
CHART_CFG = {"scrollZoom": False, "displaylogo": False, "displayModeBar": False}


def _base_layout(t: dict, height: int = 380, **kw) -> dict:
    return dict(
        template=t["plot_tpl"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Inter, sans-serif", color=t["text"], size=11),
        margin=dict(l=0, r=8, t=10, b=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=t["card"], bordercolor=t["border"],
            font=dict(family="Inter", color=t["text"], size=12),
        ),
        legend=dict(
            orientation="h", y=1.04, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=t["text"], size=11),
        ),
        **kw,
    )


def line_chart(traces: list, t: dict, height: int = 380, slider: bool = False):
    fig = go.Figure()
    for tr in traces:
        fig.add_trace(go.Scatter(
            x=tr["x"], y=tr["y"],
            mode=tr.get("mode", "lines"),
            name=tr["name"],
            line=dict(color=tr["color"], width=tr.get("width", 2.5), dash=tr.get("dash", "solid")),
            fill="tozeroy" if tr.get("fill") else None,
            fillcolor=f"rgba({tr.get('fill_rgb','124,92,252')},0.08)" if tr.get("fill") else None,
            hovertemplate="<b>Rp %{y:,.0f}</b><br>%{x|%d %b %Y}<extra></extra>",
        ))

    layout = _base_layout(t, height=height)
    layout["margin"] = dict(l=0, r=8, t=60 if slider else 10, b=0)

    xaxis_cfg = dict(
        showgrid=False, linecolor=t["border"],
        showspikes=True, spikecolor="rgba(124,92,252,0.3)", spikethickness=1,
        tickfont=dict(color=t["text"]),
    )
    if slider:
        xaxis_cfg["rangeslider"] = dict(visible=True, thickness=0.06, bgcolor=t["card2"], bordercolor=t["border"])
        xaxis_cfg["rangeselector"] = dict(
            buttons=[
                dict(count=3, label="3B", step="month", stepmode="backward"),
                dict(count=6, label="6B", step="month", stepmode="backward"),
                dict(count=1, label="1T", step="year", stepmode="backward"),
                dict(step="all", label="Semua"),
            ],
            bgcolor=t["card2"], activecolor="#7C5CFC",
            bordercolor=t["border"], borderwidth=1,
            font=dict(color=t["text"], size=11), x=0, y=1.35,
        )
        layout["legend"]["y"] = 1.16
    else:
        xaxis_cfg["rangeslider"] = dict(visible=False)

    layout.update(
        xaxis=xaxis_cfg,
        yaxis=dict(
            showgrid=True, gridcolor=t["grid"], gridwidth=1,
            zeroline=False, tickprefix="Rp ", tickformat=",.0f", linecolor=t["border"],
    tickfont=dict(color=t["text"]),
        ),
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL & DATA  (logic identical to original)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = CatBoostRegressor()
    model.load_model(str(BASE_DIR / "model_catboost_cabai.cbm"))
    scaler = joblib.load(BASE_DIR / "scaler_cabai.pkl")
    fitur  = joblib.load(BASE_DIR / "fitur_cabai.pkl")
    return model, scaler, fitur


@st.cache_data
def load_dataset():
    df = pd.read_excel(
        BASE_DIR / "dataset harga cabai rawit pasar tradisional dan iklim kota bandung.xlsx"
    )
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    return df.sort_values("Tanggal").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS  (identical to original — TIDAK DIUBAH)
# ══════════════════════════════════════════════════════════════════════════════
def ambil_lag(buffer, n):
    return buffer[-n] if len(buffer) >= n else buffer[0]

def rata_rata_bergerak(buffer, window):
    return float(np.mean(buffer[-window:]))

def total_bergerak(buffer, window):
    return float(np.sum(buffer[-window:]))

def add_lags(df, series, lags, prefix):
    for l in lags:
        df[f"{prefix}_{l}"] = series.shift(l)

def add_rolling(df, series, windows, prefix, kind="mean"):
    label = "rolling" if kind == "mean" else "cum"
    for w in windows:
        df[f"{prefix}_{label}_{w}"] = (
            series.rolling(w).mean() if kind == "mean" else series.rolling(w).sum()
        )

def build_features(df):
    """Replikasi persis feature engineering dari notebook (bagian Data Preparation)."""
    df = df.copy()
    harga = df["Harga Cabai Rawit (Rp/kg)"]
    add_lags(df, harga, [1, 2, 3, 5, 7, 14, 30], "lag")

    deltas = {
        "delta_1_3": (1, 3), "delta_1_7": (1, 7),
        "delta_7_14": (7, 14), "delta_7_30": (7, 30), "momentum_5": (1, 5),
    }
    df = df.assign(**{nama: df[f"lag_{a}"] - df[f"lag_{b}"] for nama, (a, b) in deltas.items()})

    add_rolling(df, df["Suhu Min (°C)"],    [7, 14, 30], "SuhuMin",     "mean")
    add_rolling(df, df["Curah Hujan (mm)"], [7, 14, 30], "CurahHujan",  "mean")
    add_lags(df, df["Suhu Min (°C)"], [7, 14, 21, 30, 45, 60], "SuhuMin_lag")
    add_rolling(df, df["Curah Hujan (mm)"], [14, 30], "CurahHujan", "sum")

    df["bulan"]      = df["Tanggal"].dt.month
    df["tahun"]      = df["Tanggal"].dt.year
    df["hari_index"] = (df["Tanggal"] - df["Tanggal"].min()).dt.days
    df["target_delta"] = harga - df["lag_1"]
    return df.dropna().reset_index(drop=True)


@st.cache_data
def get_model_metrics(_model, _scaler, fitur, df):
    """Evaluasi model di 20% data uji (kronologis), identik dengan notebook."""
    df_model     = build_features(df)
    X            = df_model[fitur]
    lag1         = df_model["lag_1"]
    harga_aktual = df_model["Harga Cabai Rawit (Rp/kg)"]

    split_idx       = int(len(df_model) * 0.8)
    X_test_scaled   = _scaler.transform(X.iloc[split_idx:])
    delta_pred      = _model.predict(X_test_scaled)
    y_pred          = lag1.iloc[split_idx:].values + delta_pred
    y_true          = harga_aktual.iloc[split_idx:].values

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100
    tanggal_test = df_model["Tanggal"].iloc[split_idx:].values
    return mae, rmse, r2, mape, tanggal_test, y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# AMBIL DATA IKLIM  (identical to original — TIDAK DIUBAH)
# ══════════════════════════════════════════════════════════════════════════════
def ambil_data_iklim(df, bulan, tahun):
    last_day   = calendar.monthrange(tahun, bulan)[1]
    start_date = date(tahun, bulan, 1)
    end_date   = date(tahun, bulan, last_day)
    tanggal_range = pd.date_range(str(start_date), str(end_date), freq="D")

    df_bulan_ini = df[df["Tanggal"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if len(df_bulan_ini) == len(tanggal_range) and df_bulan_ini[COL_IKLIM].notna().all().all():
        df_iklim = df_bulan_ini[["Tanggal"] + COL_IKLIM].reset_index(drop=True)
        df_iklim["bulan"] = df_iklim["Tanggal"].dt.month
        return df_iklim, "Data Historis (dataset)", start_date, end_date

    if METEO_TERSEDIA:
        try:
            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                if end_date < date.today()
                else "https://api.open-meteo.com/v1/forecast"
            )
            params = {
                "latitude": LAT, "longitude": LON, "timezone": "Asia/Jakarta",
                "start_date": str(start_date), "end_date": str(end_date),
                "daily": [
                    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                    "precipitation_sum", "precipitation_hours", "wind_speed_10m_max",
                    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
                ],
            }
            session = requests_cache.CachedSession(".cache", expire_after=3600)
            client  = openmeteo_requests.Client(session=retry(session, retries=5, backoff_factor=0.2))
            daily   = client.weather_api(url, params=params)[0].Daily()
            df_iklim = pd.DataFrame({
                "Tanggal": tanggal_range,
                **{k: daily.Variables(i).ValuesAsNumpy() for i, k in enumerate(COL_IKLIM)},
            })
            df_iklim["bulan"] = df_iklim["Tanggal"].dt.month
            return df_iklim, "API Real (Open-Meteo)", start_date, end_date
        except Exception:
            pass

    df_hist = df[df["Tanggal"].dt.month == bulan]
    rata_per_tanggal = df_hist.groupby(df_hist["Tanggal"].dt.day)[COL_IKLIM].mean()
    rata_per_tahun   = df_hist.groupby(df_hist["Tanggal"].dt.year)[COL_IKLIM].mean()

    def hitung_offset(kolom):
        s = rata_per_tahun[kolom].dropna()
        if len(s) < 2:
            return 0.0
        return np.polyfit(s.index, s.values, 1)[0] * (tahun - np.mean(s.index))

    offset = {kolom: hitung_offset(kolom) for kolom in COL_IKLIM}
    rata_disesuaikan = rata_per_tanggal + pd.Series(offset)

    df_iklim = pd.DataFrame({"hari": tanggal_range.day})
    df_iklim = df_iklim.merge(
        rata_disesuaikan.reset_index().rename(columns={"Tanggal": "hari"}),
        on="hari", how="left",
    )
    df_iklim = df_iklim.fillna(rata_disesuaikan.mean()).drop(columns="hari")
    df_iklim["Tanggal"] = tanggal_range
    df_iklim["bulan"]   = bulan
    return df_iklim, "Estimasi Historis + Tren Tahunan", start_date, end_date


# ══════════════════════════════════════════════════════════════════════════════
# LOOP FORECASTING HARIAN  (identical to original — TIDAK DIUBAH)
# ══════════════════════════════════════════════════════════════════════════════
def prediksi_bulanan(df, model, scaler, fitur, bulan, tahun):
    df_iklim, sumber, start_date, end_date = ambil_data_iklim(df, bulan, tahun)
    tanggal_awal = df["Tanggal"].min()
    df_iklim["tahun"]      = df_iklim["Tanggal"].dt.year
    df_iklim["hari_index"] = (df_iklim["Tanggal"] - tanggal_awal).dt.days

    mode = "Validasi" if df["Tanggal"].between(str(start_date), str(end_date)).any() else "Proyeksi"

    aktual_dict = df.set_index(df["Tanggal"].dt.date)["Harga Cabai Rawit (Rp/kg)"].to_dict()
    harga_buf   = list(df[df["Tanggal"] < pd.Timestamp(start_date)]["Harga Cabai Rawit (Rp/kg)"].values)
    suhu_buf    = list(df[df["Tanggal"] < pd.Timestamp(start_date)]["Suhu Min (°C)"].values)
    hujan_buf   = list(df[df["Tanggal"] < pd.Timestamp(start_date)]["Curah Hujan (mm)"].values)

    def ambil_harga(tanggal, hari_ke_belakang):
        tanggal_target = (pd.Timestamp(tanggal) - pd.Timedelta(hari_ke_belakang, "D")).date()
        return aktual_dict.get(tanggal_target, ambil_lag(harga_buf, hari_ke_belakang))

    rows, pred_list = [], []
    for _, row in df_iklim.iterrows():
        tgl    = row["Tanggal"].date()
        aktual = aktual_dict.get(tgl)

        lag_vals = {l: ambil_harga(tgl, l) for l in [1, 2, 3, 5, 7, 14, 30]}
        suhu_buf.append(row["Suhu Min (°C)"])
        hujan_buf.append(row["Curah Hujan (mm)"])

        fitur_hari_ini = {
            **{f"lag_{l}": lag_vals[l] for l in [2, 3, 5, 7, 14, 30]},
            "delta_1_3":   lag_vals[1] - lag_vals[3],
            "delta_1_7":   lag_vals[1] - lag_vals[7],
            "delta_7_14":  lag_vals[7] - lag_vals[14],
            "delta_7_30":  lag_vals[7] - lag_vals[30],
            "momentum_5":  lag_vals[1] - lag_vals[5],
            "bulan":       row["bulan"],
            "tahun":       row["tahun"],
            "hari_index":  row["hari_index"],
            **{kolom: row[kolom] for kolom in COL_IKLIM},
            **{f"SuhuMin_rolling_{w}": rata_rata_bergerak(suhu_buf, w) for w in [7, 14, 30]},
            **{f"CurahHujan_rolling_{w}": rata_rata_bergerak(hujan_buf, w) for w in [7, 14, 30]},
            **{f"SuhuMin_lag_{l}": ambil_lag(suhu_buf, l) for l in [7, 14, 21, 30, 45, 60]},
            "CurahHujan_cum_14": total_bergerak(hujan_buf, 14),
            "CurahHujan_cum_30": total_bergerak(hujan_buf, 30),
        }

        X_in       = pd.DataFrame([fitur_hari_ini])[fitur]
        delta_pred = model.predict(scaler.transform(X_in))[0]
        pred       = max(lag_vals[1] + delta_pred, 15_000)
        pred_list.append(pred)
        harga_buf.append(aktual or pred)

        error = round(abs(aktual - pred)) if aktual is not None else None
        rows.append({
            "Tanggal":                 tgl.strftime("%d/%m/%Y"),
            "Harga Aktual (Rp/kg)":   int(aktual) if aktual is not None else None,
            "Harga Prediksi (Rp/kg)": int(pred),
            "Error (Rp)":             error,
        })

    df_iklim            = df_iklim.copy()
    df_iklim["Prediksi"] = pred_list
    return pd.DataFrame(rows), df_iklim, sumber, mode


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard(df, model, scaler, fitur, mae, rmse, r2, mape, tgl_test, y_test, y_pred_all):
    t = T()

    # ── 4 Summary Cards ──────────────────────────────────────────────────────
    harga_terakhir = df["Harga Cabai Rawit (Rp/kg)"].iloc[-1]
    today          = date.today()
    next_m         = today.month % 12 + 1
    next_y         = today.year + (1 if today.month == 12 else 0)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("", f"Rp {harga_terakhir/1000:.1f}K", "Harga Data Terakhir", accent=t["primary"]), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("", f"{NAMA_BULAN[next_m][:3]} {next_y}", "Bulan Prediksi Berikutnya", accent=t["blue"]), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("", f"{len(df):,}", "Total Observasi", accent=t["green"]), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("", f"{r2:.4f}", "Akurasi Model (R²)", accent=t["orange"]), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── 70 : 30 Layout ────────────────────────────────────────────────────────
    col_main, col_side = st.columns([7, 3])

    with col_main:
        with st.container(key="card_1"):
            st.markdown(
                sec_header("Tren Harga Historis", f"{df['Tanggal'].min().year}–{df['Tanggal'].max().year}"),
                unsafe_allow_html=True,
            )
            fig_hist = line_chart(
                [
                    {"x": df["Tanggal"], "y": df["Harga Cabai Rawit (Rp/kg)"],
                     "name": "Harga Aktual", "color": t["primary"], "fill": True},
                    {"x": tgl_test, "y": y_pred_all,
                     "name": "Prediksi Model (Data Uji)", "color": t["secondary"], "dash": "dash"},
                ],
                t, height=360, slider=True,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config=CHART_CFG)

    with col_side:
        with st.container(key="card_2"):
            st.markdown(sec_header("Ringkasan Model"), unsafe_allow_html=True)
            st.markdown(f"""
    {model_item("","Algoritma","CatBoost")}
    {model_item("","Dataset",f"{len(df):,} hari")}
    {model_item("","Fitur",f"{len(fitur)} fitur")}
    {model_item("","MAE",f"Rp {mae:,.0f}")}
    {model_item("","RMSE",f"Rp {rmse:,.0f}")}
    {model_item("","MAPE",f"{mape:.2f}%")}
    {model_item("","R² Score",f"{r2:.4f}")}
    """, unsafe_allow_html=True)
            st.markdown(f"""
    <div style="font-size:0.74rem;color:{t['muted']};padding:10px 0;border-top:1px solid {t['border']};margin-top:6px;">
      <div>{df['Tanggal'].min().strftime('%d %b %Y')} — {df['Tanggal'].max().strftime('%d %b %Y')}</div>
      <div style="margin-top:3px;">Kota Bandung ({abs(LAT):.2f}°S, {LON:.2f}°E)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Bottom 3 Cards ────────────────────────────────────────────────────────
    b1, b2, b3 = st.columns(3)
    importance = model.get_feature_importance()
    feat_df    = pd.DataFrame({"Fitur": fitur, "Importance": importance})

    with b1:
        with st.container(key="card_3"):
            st.markdown(sec_header("Feature Importance", "Top 10"), unsafe_allow_html=True)
            top10  = feat_df.sort_values("Importance").tail(10)
            colors = [t["primary"] if f in MOMENTUM_COLS else t["secondary"] for f in top10["Fitur"]]
            fig_fi = go.Figure(go.Bar(
                x=top10["Importance"], y=top10["Fitur"], orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
            ))
            ly = _base_layout(t, height=290)
            ly.update(xaxis=dict(showgrid=True, gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["text"])),
                      yaxis=dict(showgrid=False, tickfont=dict(size=8, color=t["text"])),
                      showlegend=False)
            fig_fi.update_layout(**ly)
            st.plotly_chart(fig_fi, use_container_width=True, config=CHART_CFG)

    with b2:
        with st.container(key="card_4"):
            st.markdown(sec_header("Distribusi Harga"), unsafe_allow_html=True)
            fig_dist = go.Figure(go.Histogram(
                x=df["Harga Cabai Rawit (Rp/kg)"], nbinsx=40,
                marker=dict(color=t["primary"], opacity=0.8, line=dict(width=0)),
                hovertemplate="Rp %{x:,.0f}<br>%{y} data<extra></extra>",
            ))
            ly2 = _base_layout(t, height=290)
            ly2.update(
                xaxis=dict(tickprefix="Rp ", tickformat=",.0f", showgrid=False, tickfont=dict(color=t["text"])),
                yaxis=dict(showgrid=True, gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["text"])),
                bargap=0.06, showlegend=False,
            )
            fig_dist.update_layout(**ly2)
            st.plotly_chart(fig_dist, use_container_width=True, config=CHART_CFG)

    with b3:
        with st.container(key="card_5"):
            st.markdown(sec_header("Korelasi Fitur"), unsafe_allow_html=True)
            df_feat = build_features(df)
            top_f   = (["Harga Cabai Rawit (Rp/kg)"]
                       + list(feat_df.sort_values("Importance", ascending=False).head(7)["Fitur"]))
            corr = df_feat[[c for c in top_f if c in df_feat.columns]].corr()
            short = [c.replace("Harga Cabai Rawit (Rp/kg)", "Harga")
                      .replace("CurahHujan","CH").replace("SuhuMin","Tmin")
                      .replace("_rolling","_r").replace("_lag","_l") for c in corr.columns]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=short, y=short,
                colorscale=[[0, "#A78BFA"], [0.5, t["card2"]], [1, "#7C5CFC"]],
                zmid=0, showscale=False,
                hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
                text=corr.round(2).values,
                texttemplate="%{text}", textfont=dict(size=7),
            ))
            ly3 = _base_layout(t, height=290)
            ly3.update(xaxis=dict(showgrid=False, tickfont=dict(size=7, color=t["text"])),
                       yaxis=dict(showgrid=False, tickfont=dict(size=7, color=t["text"])))
            fig_corr.update_layout(**ly3)
            st.plotly_chart(fig_corr, use_container_width=True, config=CHART_CFG)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA HISTORIS
# ══════════════════════════════════════════════════════════════════════════════
def page_data_historis(df):
    t = T()
    st.markdown(sec_header("Data Historis", f"{len(df):,} Observasi"), unsafe_allow_html=True)

    # Filter bar
    f1, f2, f3, f4 = st.columns([3, 1, 1, 1])
    with f1:
        search = st.text_input("", placeholder="Cari tanggal (DD/MM/YYYY)…", label_visibility="collapsed")
    with f2:
        tahun_opts = ["Semua Tahun"] + sorted(df["Tanggal"].dt.year.unique().tolist(), reverse=True)
        filter_tahun = st.selectbox("Tahun", tahun_opts, label_visibility="collapsed")
    with f3:
        bulan_opts = ["Semua Bulan"] + [NAMA_BULAN[i] for i in range(1, 13)]
        filter_bulan = st.selectbox("Bulan", bulan_opts, label_visibility="collapsed")
    with f4:
        csv_dl = df.to_csv(index=False).encode("utf-8")
        st.download_button("Unduh CSV", data=csv_dl,
                           file_name="cabaicast_data_historis.csv",
                           mime="text/csv", use_container_width=True)

    # Apply filters
    df_view = df.copy()
    if search:
        df_view = df_view[df_view["Tanggal"].dt.strftime("%d/%m/%Y").str.contains(search)]
    if filter_tahun != "Semua Tahun":
        df_view = df_view[df_view["Tanggal"].dt.year == int(filter_tahun)]
    if filter_bulan != "Semua Bulan":
        bln_n = list(NAMA_BULAN.values()).index(filter_bulan) + 1
        df_view = df_view[df_view["Tanggal"].dt.month == bln_n]

    # Pagination
    rows_per_page = 20
    total_pages   = max(1, (len(df_view) - 1) // rows_per_page + 1)
    if st.session_state.data_page >= total_pages:
        st.session_state.data_page = 0

    start   = st.session_state.data_page * rows_per_page
    df_show = df_view.iloc[start: start + rows_per_page].copy()
    df_show["Tanggal"] = df_show["Tanggal"].dt.strftime("%d %b %Y")
    df_show["Harga Cabai Rawit (Rp/kg)"] = df_show["Harga Cabai Rawit (Rp/kg)"].apply(
        lambda x: f"Rp {x:,.0f}"
    )

    with st.container(key="card_6"):
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=480)

        st.markdown("<br>", unsafe_allow_html=True)
        pa, pb, pc = st.columns([1, 3, 1])
        with pa:
            if st.button("← Sebelumnya", disabled=st.session_state.data_page == 0):
                st.session_state.data_page -= 1; st.rerun()
        with pb:
            st.markdown(
                f'<p style="text-align:center;color:{t["muted"]};font-size:0.82rem;margin-top:0.55rem;">'
                f'Halaman {st.session_state.data_page+1} dari {total_pages} &nbsp;·&nbsp; {len(df_view):,} baris</p>',
                unsafe_allow_html=True,
            )
        with pc:
            if st.button("Berikutnya →", disabled=st.session_state.data_page >= total_pages - 1):
                st.session_state.data_page += 1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI HARGA
# ══════════════════════════════════════════════════════════════════════════════
def page_prediksi(df, model, scaler, fitur):
    t = T()
    st.markdown(sec_header("Prediksi Harga Cabai Rawit"), unsafe_allow_html=True)

    with st.container(key="card_7"):
        st.markdown(
            f'<p style="font-size:0.855rem;color:{t["muted"]};margin-bottom:1.2rem;">'
            f'Pilih bulan dan tahun target untuk melakukan prediksi harga.</p>',
            
            unsafe_allow_html=True,
        )

        fi1, fi2 = st.columns(2)
        with fi1:
            bulan = st.selectbox(
                "Bulan Target",
                list(NAMA_BULAN.keys()),
                format_func=lambda x: NAMA_BULAN[x],
                index=date.today().month - 1,
            )
        with fi2:
            tahun_ini  = date.today().year
            tahun_opts = list(range(2022, tahun_ini + 3))
            tahun      = st.selectbox("Tahun Target", tahun_opts, index=tahun_opts.index(tahun_ini))

        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(f"""
    </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
            run = st.button("Prediksi Harga", use_container_width=True)


    if run:
        with st.spinner("Menjalankan model CatBoost — prediksi harian dalam proses…"):
            df_hasil, df_iklim_pred, sumber, mode = prediksi_bulanan(df, model, scaler, fitur, bulan, tahun)
        st.session_state.pred_result = {
            "df_hasil": df_hasil, "df_iklim": df_iklim_pred,
            "sumber": sumber, "mode": mode,
            "nama": f"{NAMA_BULAN[bulan]} {tahun}",
            "preds": df_hasil["Harga Prediksi (Rp/kg)"],
            "bulan": bulan,
        }

    if st.session_state.pred_result:
        res           = st.session_state.pred_result
        preds         = res["preds"]
        df_hasil      = res["df_hasil"]
        df_iklim_pred = res["df_iklim"]
        mode          = res["mode"]
        nama_b        = res["nama"]
        bln           = res["bulan"]

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # Musim analysis (same logic as original)
        if bln in [10, 11, 12, 1, 2, 3]:
            musim = "Musim Hujan"
            efek  = "tingginya curah hujan yang berpotensi menghambat distribusi dan memicu risiko gagal panen"
        elif bln in [4, 5]:
            musim = "Masa Transisi (Pancaroba)"
            efek  = "perubahan cuaca fluktuatif serta pergeseran pola tanam"
        else:
            musim = "Musim Kemarau"
            efek  = "penurunan curah hujan dan kenaikan suhu yang memengaruhi produktivitas lahan"

        r1, r2c, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
<div class="result-card">
    <div class="result-val">Rp {preds.mean()/1000:.1f}K</div>
    <div class="result-lbl">Rata-rata Proyeksi</div>
</div>""", unsafe_allow_html=True)
        with r2c:
            st.markdown(metric_card("", f"Rp {preds.max()/1000:.0f}K", "Titik Tertinggi"), unsafe_allow_html=True)
        with r3:
            st.markdown(metric_card("", f"Rp {preds.min()/1000:.0f}K", "Titik Terendah"), unsafe_allow_html=True)
        with r4:
            st.markdown(metric_card("", str(len(df_hasil)), "Durasi (Hari)"), unsafe_allow_html=True)

        st.markdown(f"""
<div class="insight" style="margin-top:1rem;">
    <div class="insight-ttl">Analisis Musiman — {nama_b}</div>
    <div class="insight-txt">
        Bulan <b>{NAMA_BULAN[bln]}</b> berada pada fase <b>{musim}</b>. Model CatBoost
        memproyeksikan rata-rata <b>Rp {preds.mean()/1000:.1f}K/kg</b>, mengintegrasikan dampak
        dari {efek}. Sumber data iklim: <b>{res['sumber']}</b>.
    </div>
</div>""", unsafe_allow_html=True)

        with st.container(key="card_8"):
            st.markdown(sec_header(f"Kurva Proyeksi Harian bulan {nama_b}", mode), unsafe_allow_html=True)

            traces_pred = [{
                "x": df_iklim_pred["Tanggal"], "y": df_iklim_pred["Prediksi"],
                "name": "Prediksi Model", "color": t["primary"], "mode": "lines+markers",
            }]
            if mode == "Validasi":
                aktual_vals = df_hasil["Harga Aktual (Rp/kg)"].dropna()
                aktual_tgl  = pd.to_datetime(df_hasil.loc[aktual_vals.index, "Tanggal"], format="%d/%m/%Y")
                traces_pred.append({
                    "x": aktual_tgl, "y": aktual_vals,
                    "name": "Harga Aktual", "color": t["green"], "dash": "dash",
                })
            fig_pred = line_chart(traces_pred, t, height=380)
            st.plotly_chart(fig_pred, use_container_width=True, config=CHART_CFG)

        with st.container(key="card_9"):
            st.markdown(sec_header("Rincian Data Harian"), unsafe_allow_html=True)

            df_tampil = df_hasil.copy()
            if mode == "Validasi":
                df_tampil["Harga Aktual (Rp/kg)"]   = df_tampil["Harga Aktual (Rp/kg)"].apply(
                    lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "–"
                )
                df_tampil["Harga Prediksi (Rp/kg)"] = df_tampil["Harga Prediksi (Rp/kg)"].apply(
                    lambda x: f"Rp {x:,.0f}"
                )
                df_tampil["Error (Rp)"] = df_tampil["Error (Rp)"].apply(
                    lambda x: f"Rp {x:,.0f}" if pd.notna(x) else "–"
                )
            else:
                df_tampil = df_tampil[["Tanggal", "Harga Prediksi (Rp/kg)"]]
                df_tampil["Harga Prediksi (Rp/kg)"] = df_tampil["Harga Prediksi (Rp/kg)"].apply(
                    lambda x: f"Rp {x:,.0f}"
                )

            st.dataframe(df_tampil, hide_index=True, use_container_width=True, height=360)

            csv_pred = df_hasil.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Laporan CSV",
                data=csv_pred,
                file_name=f"prediksi_{nama_b.replace(' ', '_')}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
def page_evaluasi(tgl_test, y_test, y_pred_all, mae, rmse, r2, mape, model, fitur):
    t = T()
    st.markdown(sec_header("Evaluasi Model", "20% Data Uji Kronologis"), unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1: st.markdown(metric_card("", f"{r2:.4f}", "R² Score"), unsafe_allow_html=True)
    with e2: st.markdown(metric_card("", f"Rp {mae:,.0f}", "Mean Absolute Error"), unsafe_allow_html=True)
    with e3: st.markdown(metric_card("", f"Rp {rmse:,.0f}", "Root Mean Square Error"), unsafe_allow_html=True)
    with e4: st.markdown(metric_card("", f"{mape:.2f}%", "MAPE"), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        with st.container(key="card_10"):
            st.markdown(sec_header("Aktual vs Prediksi (Data Uji)"), unsafe_allow_html=True)
            fig_ev = line_chart([
                {"x": tgl_test, "y": y_test,     "name": "Aktual",   "color": t["green"]},
                {"x": tgl_test, "y": y_pred_all, "name": "Prediksi", "color": t["primary"], "dash": "dash"},
            ], t, height=340)
            st.plotly_chart(fig_ev, use_container_width=True, config=CHART_CFG)

    with col_b:
        with st.container(key="card_11"):
            st.markdown(sec_header("Residual Plot"), unsafe_allow_html=True)
            residuals = y_test - y_pred_all
            fig_res   = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_pred_all, y=residuals, mode="markers",
                marker=dict(color=t["primary"], opacity=0.55, size=5),
                hovertemplate="Pred: Rp %{x:,.0f}<br>Residual: Rp %{y:,.0f}<extra></extra>",
                name="Residual",
            ))
            fig_res.add_hline(y=0, line_color=t["red"], line_dash="dash", line_width=1.5)
            ly_r = _base_layout(t, height=340)
            ly_r.update(
                xaxis=dict(tickprefix="Rp ", tickformat=",.0f", showgrid=False, tickfont=dict(color=t["text"])),
                yaxis=dict(tickprefix="Rp ", tickformat=",.0f", showgrid=True,
                        gridcolor=t["grid"], zeroline=False, tickfont=dict(color=t["text"])),
                showlegend=False,
)
            fig_res.update_layout(**ly_r)
            st.plotly_chart(fig_res, use_container_width=True, config=CHART_CFG)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    with st.container(key="card_12"):
        st.markdown(sec_header("Ringkasan Statistik Data Uji"), unsafe_allow_html=True)
        stats_tbl = pd.DataFrame({
            "Metrik": ["Jumlah Sampel Uji", "Rata-rata Aktual", "Rata-rata Prediksi", "MAE", "RMSE", "MAPE", "R² Score"],
            "Nilai": [
                f"{len(y_test):,} hari",
                f"Rp {y_test.mean():,.0f}",
                f"Rp {y_pred_all.mean():,.0f}",
                f"Rp {mae:,.0f}",
                f"Rp {rmse:,.0f}",
                f"{mape:.2f}%",
                f"{r2:.6f}",
            ],
            "Keterangan": [
                "Split kronologis 80/20",
                "Harga rata-rata periode uji",
                "Proyeksi rata-rata periode uji",
                f"Deviasi rata-rata ±Rp {mae:,.0f}",
                f"Kesalahan kuadratik ±Rp {rmse:,.0f}",
                f"Error relatif rata-rata {mape:.1f}%",
                "1.0 = sempurna" if r2 > 0.9 else "Ada ruang peningkatan",
            ],
        })
        st.dataframe(stats_tbl, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
inject_css()

# ── Load artifacts & compute metrics ─────────────────────────────────────────
try:
    model, scaler, fitur = load_artifacts()
    df = load_dataset()
    mae, rmse, r2, mape, tgl_test, y_test, y_pred_all = get_model_metrics(model, scaler, fitur, df)
except Exception as e:
    st.error(
        f"**Gagal memuat model atau data.**\n\n"
        f"Pastikan file berikut ada di direktori yang sama dengan `app.py`:\n"
        f"- `model_catboost_cabai.cbm`\n"
        f"- `scaler_cabai.pkl`\n"
        f"- `fitur_cabai.pkl`\n"
        f"- `dataset harga cabai rawit pasar tradisional dan iklim kota bandung.xlsx`\n\n"
        f"**Detail error:** `{e}`"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sb-logo">
    <div class="sb-logo-icon">CC</div>
    <div>
        <div class="sb-logo-name">CabaiCast</div>
        <div class="sb-logo-sub">Analytics</div>
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<span class="sb-sect-lbl">Menu Utama</span>', unsafe_allow_html=True)

    nav = st.radio(
        "Menu Navigasi",
        options=[
            "Dashboard",
            "Data Historis",
            "Prediksi Harga",
            "Evaluasi Model",
        ],
        key="nav_radio",
        label_visibility="collapsed"
    )

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sb-sect-lbl">Sistem</span>', unsafe_allow_html=True)

    t_sb = T()
    st.markdown(
        f'<div style="padding:0.8rem 1.2rem 1rem;font-size:0.71rem;color:{t_sb["sb_muted"]};">'
        f'<div>{len(df):,} observasi · {len(fitur)} fitur</div>'
        f'<div style="margin-top:3px;">{df["Tanggal"].min().year}–{df["Tanggal"].max().year}'
        f' · Kota Bandung</div></div>',
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
wib = pytz.timezone('Asia/Jakarta')
now = datetime.now(wib)

_hari   = {0:"Senin",1:"Selasa",2:"Rabu",3:"Kamis",4:"Jumat",5:"Sabtu",6:"Minggu"}
_bulan  = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
           7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}

# Format tanggal
tgl_str = f"{_hari[now.weekday()]}, {now.day} {_bulan[now.month]} {now.year}"

# Sapaan berdasarkan jam WIB
if now.hour < 11:
    sapaan = "Selamat Pagi"
elif now.hour < 15:
    sapaan = "Selamat Siang"
elif now.hour < 19:
    sapaan = "Selamat Sore"
else:
    sapaan = "Selamat Malam"
t_hdr   = T()

hcol1, hcol2 = st.columns([6, 2])
with hcol1:
    st.markdown(f"""
<div style="padding:1.4rem 0 1.2rem 0;border-bottom:1px solid {t_hdr['border']};margin-bottom:1.6rem;">
    <h1 style="font-size:1.6rem;font-weight:800;color:{t_hdr['text']};margin:0 0 4px 0;letter-spacing:-0.5px;">
    {sapaan}! 👋
</h1>
    <p style="font-size:1rem;color:{t_hdr['muted']};margin:5px 0 0 0;">
        Sistem Prediksi Harga Cabai Rawit Kota Bandung
    </p>
</div>""", unsafe_allow_html=True)
with hcol2:
    st.markdown(f'''<div class="hdr-date" style="display:flex;align-items:center;gap:7px;">
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="{t_hdr['primary']}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="16" rx="3"/><line x1="16" y1="3" x2="16" y2="7"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
    <span>{tgl_str}</span></div>''', unsafe_allow_html=True)

# ── Page Router ───────────────────────────────────────────────────────────────
if nav == "Dashboard":
    page_dashboard(df, model, scaler, fitur, mae, rmse, r2, mape, tgl_test, y_test, y_pred_all)
elif nav == "Data Historis":
    page_data_historis(df)
elif nav == "Prediksi Harga":
    page_prediksi(df, model, scaler, fitur)
elif nav == "Evaluasi Model":
    page_evaluasi(tgl_test, y_test, y_pred_all, mae, rmse, r2, mape, model, fitur)
