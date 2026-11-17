# -*- coding: utf-8 -*-
"""
上海市黄浦区大众餐饮市场分析 · 交互式仪表盘
==============================================
数据来源：高德地图 POI（黄浦区 2,671 家餐饮店铺，人均消费 ≤ 500 元）
运行方式：streamlit run app.py
依赖库：见 requirements.txt

功能：
  - 顶部数据管理模块：双 CSV 上传 + 自动校验 + 状态指示灯 + 智能数据匹配（A/B/C/D 四种情况）
  - 页面 1 市场总览：统计卡片 + 人均消费直方图 + 类型构成饼图 + 营业时长分布
  - 页面 2 商圈对比：动态热力图（人均消费）+ 商圈小提琴/箱线/品类构成
  - 页面 3 业态聚类：四类业态雷达图（可点击高亮）+ 明细表
  - 页面 4 定价策略：价格-评分散点 + 价格滑条 + SHAP 边际贡献
  - 页面 5 空间格局：莫兰散点图 + LISA 集聚图 + 文字解读
"""
import base64
import html as _html_lib
import io
import math
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
import folium.plugins
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================== 品牌配色（已通过可访问性校验） ====================
BRAND_NAVY = "#1B3A5C"     # 主色·深蓝（标题/卡片/UI）
BRAND_GOLD = "#D4A853"     # 强调色·金（高亮/锚点）
SERIES_NAVY = "#2E6DA3"    # 系列用海军蓝
PAGE_BG = "#F2F2F0"
SURFACE = "#FCFCFB"
INK = "#0B0B0B"
INK_2 = "#52514E"
GRID = "#E1E0D9"
AXIS = "#C3C2B7"
FONT = "Microsoft YaHei, Segoe UI, sans-serif"
RED_LINE = "#D03B3B"

# 状态指示灯颜色（status 色，专用于状态语义）
GRAY_DOT = "#9AA0A6"
GREEN_OK = "#0CA30C"
RED_ERR = "#D03B3B"

# 品类色板（按出现频次从高到低固定分配；颜色跟随品类而非排序）
TYPE_COLOR_ORDER = ["#2E6DA3", "#D4A853", "#1BAF7A", "#E34948", "#6250D6"]
TYPE_COLOR_OTHER = "#008300"

# 聚类固定颜色（按业态名称映射，全对可区分性已验证）
CLUSTER_COLORS = {
    "大众高效型": "#2E6DA3",
    "全天候口碑型": "#D4A853",
    "低价低质型": "#E34948",
    "高端精品型": "#1BAF7A",
}
CLUSTER_PREFER = ["大众高效型", "全天候口碑型", "低价低质型", "高端精品型"]

PRICE_RAMP = ["#7FAACD", "#5E94C0", "#3E7DB0", "#1B3A5C"]
HEAT_GRADIENT = {0.2: "#7FAACD", 0.4: "#5E94C0", 0.6: "#3E7DB0", 0.8: "#1B3A5C", 1.0: "#D4A853"}

# 是否启用街道底图瓦片（False=纯白底图，与论文图4-13一致、零网络依赖；
# 需要街道背景时改为 True，瓦片走 高德→CartoDB→OSM 降级链）
TILES_ENABLED = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(os.path.dirname(BASE_DIR), "数据文件"),
    r"E:\期末大作业\数据文件",
]

# ==================== 预置数据加载 ====================

def _find_data_dir():
    for d in DATA_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("找不到数据目录。请把 CSV 数据文件放到 data/ 子目录，"
                             "或确认 E:\\期末大作业\\数据文件 存在。")


@st.cache_data(show_spinner="正在加载预置数据…")
def load_main():
    d = _find_data_dir()
    p = os.path.join(d, "餐饮数据_清洗后_全部.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(show_spinner="正在加载预置数据…")
def load_hours():
    d = _find_data_dir()
    p = os.path.join(d, "餐饮数据_清洗后_有时长.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig")


@st.cache_data(show_spinner="正在加载 SHAP 曲线…")
def load_shap_curve():
    p = os.path.join(BASE_DIR, "data", "shap_price_curve.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p, encoding="utf-8-sig")


# ==================== 上传文件校验 ====================

def _read_csv(raw: bytes):
    last = None
    for enc in ("utf-8-sig", "gbk", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise last or ValueError("无法识别文件编码")


@st.cache_data(show_spinner=False)
def _validate_upload(raw: bytes, kind: str):
    """校验并规整上传的 CSV。返回 (ok, 错误信息, 规整后的DataFrame或None)。"""
    try:
        df = _read_csv(raw)
    except Exception as e:
        return False, f"❌ 文件无法解析：{e}", None
    if df is None or df.empty or len(df.columns) < 3:
        return False, "❌ 文件内容为空或列数异常，请检查CSV格式。", None

    cols = set(df.columns)
    id_cols = {"唯一标识", "店铺名称", "店名", "id", "ID", "name", "Name"}
    type_cols = {"类型_合并", "类型", "类型大类_最终", "category", "Category"}
    if not (cols & id_cols):
        return False, "❌ 文件缺少店铺名称（name）或ID列，请检查CSV格式。", None
    if not (cols & type_cols):
        return False, "❌ 文件缺少“类型”列，请检查CSV格式。", None
    for c in ("人均消费", "综合评分", "经度", "纬度", "商圈"):
        if c not in cols:
            return False, f"❌ 文件缺少“{c}”列，请检查CSV格式。", None

    num = {}
    for c in ("人均消费", "综合评分", "经度", "纬度"):
        num[c] = pd.to_numeric(df[c], errors="coerce")
        if num[c].notna().sum() == 0:
            return False, f"❌ “{c}”列中未检测到有效数值，请确认数据。", None

    price = num["人均消费"]
    bad = price[price.notna() & ((price < 1) | (price > 500))]
    if len(bad):
        return False, f"❌ “人均消费”存在超出 1–500 范围的值（如 {bad.iloc[0]:.1f}），请确认数据。", None
    score = num["综合评分"]
    bad = score[score.notna() & ((score < 0) | (score > 5))]
    if len(bad):
        return False, f"❌ “综合评分”存在超出 0–5 范围的值（如 {bad.iloc[0]:.1f}），请确认数据。", None

    if kind == "hours":
        if "营业时长" not in cols:
            return False, "❌ 文件缺少“营业时长”列，请检查CSV格式。", None
        dur = pd.to_numeric(df["营业时长"], errors="coerce")
        if dur.notna().sum() == 0:
            return False, "❌ “营业时长”列中未检测到有效数值，请确认数据。", None
        bad = dur[dur.notna() & ((dur < 0) | (dur > 24))]
        if len(bad):
            return False, f"❌ “营业时长”存在超出 0–24 范围的值（如 {bad.iloc[0]:.1f}），请确认数据。", None

    out = pd.DataFrame(index=df.index)
    name_col = next((c for c in ("店铺名称", "店名", "name", "Name", "唯一标识", "id", "ID") if c in cols), None)
    out["店铺名称"] = df[name_col].astype(str) if name_col else [f"店铺 {i + 1}" for i in range(len(df))]
    type_col = next((c for c in ("类型_合并", "类型", "类型大类_最终", "category", "Category") if c in cols), None)
    out["类型_合并"] = df[type_col].astype(str)
    for c in ("人均消费", "综合评分", "经度", "纬度"):
        out[c] = num[c]
    out["商圈"] = df["商圈"].astype(str)
    if "区县" in cols:
        out["区县"] = df["区县"].astype(str)
    if kind == "hours":
        out["营业时长"] = pd.to_numeric(df["营业时长"], errors="coerce")

    out = out[out["人均消费"].notna() & out["综合评分"].notna()
              & out["经度"].notna() & out["纬度"].notna()].copy()
    if out.empty:
        return False, "❌ 清洗后没有可用记录，请检查数据。", None
    return True, "", out


# ==================== 数据匹配（A/B/C/D） ====================

def type_palette(df):
    """品类 Top5 + 其他，颜色按频次固定分配。"""
    vc = df["类型_合并"].astype(str).value_counts()
    top = vc.head(5).index.tolist()
    pal = {t: c for t, c in zip(top, TYPE_COLOR_ORDER)}
    pal["其他"] = TYPE_COLOR_OTHER
    return pal, top


def apply_type_display(df):
    pal, _ = type_palette(df)
    df = df.copy()
    df["类型_显示"] = df["类型_合并"].astype(str).apply(lambda x: x if x in pal else "其他")
    return df


def resolve_data():
    """按 A/B/C/D 四种情况解析实际使用的数据集。返回 (df_main, df_hours, mode)。"""
    a = st.session_state.get("up_all")
    h = st.session_state.get("up_hours")
    if a is None and h is None:
        return load_main(), load_hours(), "D"
    if a is not None and h is not None:
        return a, h, "A"
    if a is not None:
        return a, None, "B"
    return h, h, "C"


def filter_huangpu(df):
    """取黄浦区切片（报告口径）。无区县列或没有黄浦区时返回全量副本。"""
    if df is None:
        return None
    if "区县" in df.columns and (df["区县"].astype(str) == "黄浦区").any():
        return df[df["区县"].astype(str) == "黄浦区"].copy()
    return df.copy()


def compute_cluster_summary(hdf):
    """运行时 K-Means（K=4）聚类，返回汇总表。"""
    feats = ["人均消费", "综合评分", "营业时长"]
    data = hdf[feats].dropna().copy()
    if len(data) < 20:
        return None
    sc = StandardScaler().fit_transform(data)
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(sc)
    data["聚类"] = km.labels_
    centers_orig = data.groupby("聚类")[feats].mean()
    m_cost = centers_orig["人均消费"].mean()
    m_score = centers_orig["综合评分"].mean()
    m_dur = centers_orig["营业时长"].mean()

    def label(row):
        hc = row["人均消费"] > m_cost
        hs = row["综合评分"] > m_score
        ld = row["营业时长"] > m_dur
        if hc and hs and not ld:
            return "高端精品型"
        if not hc and hs and ld:
            return "全天候口碑型"
        if not hc and hs and not ld:
            return "大众高效型"
        if not hc and not hs and not ld:
            return "低价低质型"
        return f"{'高' if hc else '低'}消费·{'高' if hs else '低'}评分·{'长' if ld else '短'}时长"

    centers_scaled = pd.DataFrame(km.cluster_centers_, columns=feats)
    rows = []
    for i in range(4):
        n = int((km.labels_ == i).sum())
        rows.append({
            "聚类名称": label(centers_orig.loc[i]),
            "样本量": n,
            "占比%": round(n / len(data) * 100, 1),
            "人均消费均值": round(centers_orig.loc[i, "人均消费"], 2),
            "综合评分均值": round(centers_orig.loc[i, "综合评分"], 2),
            "营业时长均值": round(centers_orig.loc[i, "营业时长"], 2),
            "z人均消费": round(centers_scaled.loc[i, "人均消费"], 3),
            "z综合评分": round(centers_scaled.loc[i, "综合评分"], 3),
            "z营业时长": round(centers_scaled.loc[i, "营业时长"], 3),
        })
    return pd.DataFrame(rows)


def get_cluster_summary_cached(hdf):
    fp = (len(hdf), round(float(hdf["人均消费"].mean()), 2),
          round(float(hdf["综合评分"].mean()), 2),
          round(float(hdf["营业时长"].mean()), 2))
    if st.session_state.get("_cl_fp") != fp:
        st.session_state["_cl_fp"] = fp
        st.session_state["_cl_summary"] = compute_cluster_summary(hdf)
    return st.session_state["_cl_summary"]


def get_shap_data(mode):
    """mode=='D' 用预置数据真实 SHAP 曲线；上传数据用模拟曲线（峰值 ~120 元）。"""
    if mode == "D":
        try:
            return load_shap_curve(), True
        except FileNotFoundError:
            pass
    p = np.linspace(5, 500, 100)
    y = (0.12 * np.exp(-((p - 120) / 62) ** 2)
         + 0.020 * np.log1p(p / 40)
         - 0.12 * np.exp(-p / 30)
         + 0.10 / (1 + np.exp(-(p - 380) / 50)))
    return pd.DataFrame({"price_bin_center": p, "shap_mean": y, "n": [1] * len(p)}), False


# ==================== 样式工具 ====================

def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {PAGE_BG}; }}
    [data-testid="stSidebar"] {{ background-color: {BRAND_NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #F0EFEC; }}
    [data-testid="stSidebar"] .stRadio label p {{ color: #F0EFEC; font-size: 15px; }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background-color: rgba(212,168,83,0.15); }}
    .brand-title {{ font-size: 20px; font-weight: 700; color: {BRAND_GOLD}; line-height: 1.35; }}
    .brand-sub {{ font-size: 12px; color: #C9C7C2; margin-bottom: 18px; }}
    .hint-box {{
        background: #FFF8E1; border: 1px solid #EADDA8; border-radius: 8px;
        padding: 10px 16px; font-size: 14px; color: #6B5D2A; margin-bottom: 10px;
    }}
    .up-head {{
        background: {BRAND_NAVY}; color: #FFFFFF; padding: 8px 14px;
        border-radius: 8px 8px 0 0; font-size: 14px; font-weight: 600;
    }}
    .up-sub {{ display: block; font-size: 11px; color: #C9C7C2; font-weight: 400; margin-top: 2px; }}
    .stat-card {{
        background: {SURFACE}; border: 1px solid {GRID}; border-top: 4px solid {BRAND_GOLD};
        border-radius: 8px; padding: 14px 18px 10px 18px; margin-bottom: 4px;
    }}
    .stat-label {{ font-size: 13px; color: {INK_2}; }}
    .stat-value {{ font-size: 30px; font-weight: 700; color: {BRAND_NAVY}; line-height: 1.25; }}
    .stat-unit {{ font-size: 14px; color: {INK_2}; font-weight: 400; }}
    .stat-note {{ font-size: 12px; color: {INK_2}; margin-top: 2px; }}
    .shap-value {{ font-size: 40px; font-weight: 700; }}
    .section-title {{
        font-size: 17px; font-weight: 700; color: {BRAND_NAVY};
        border-left: 4px solid {BRAND_GOLD}; padding-left: 10px; margin: 6px 0 8px 0;
    }}
    [data-testid="stIFrame"] {{
        display: block;
    }}
    .stElementContainer:has(> [data-testid="stIFrame"]) {{
        margin-left: auto !important; margin-right: auto !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def style_fig(fig, title=None, height=None):
    fig.update_layout(
        font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=52, r=20, t=54, b=44),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False),
        legend=dict(font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=13)),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=16, color=BRAND_NAVY, family=FONT),
                                     x=0.01, xanchor="left"))
    if height:
        fig.update_layout(height=height)
    return fig


def stat_card(label, value, unit="", note=""):
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">{label}</div>
      <div class="stat-value">{value}<span class="stat-unit"> {unit}</span></div>
      <div class="stat-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def src_note(text):
    st.markdown(f'<div style="color:#999999;font-size:12px;margin:-6px 0 10px 2px;">{text}</div>',
                unsafe_allow_html=True)


def placeholder_box(text):
    st.markdown(f"""
    <div style="background:#F2F2F0;border:1.5px dashed #C3C2B7;border-radius:10px;
                padding:64px 24px;text-align:center;color:#898781;font-size:15px;">
      ⚠️ {text}
    </div>
    """, unsafe_allow_html=True)


def render_status(state, loaded_n):
    if state == "ok":
        dot, txt = GREEN_OK, f"已加载（n={loaded_n}条）"
    elif state == "bad":
        dot, txt = RED_ERR, "格式不符"
    else:
        dot, txt = GRAY_DOT, "未上传"
    st.markdown(
        f'<div style="padding:6px 2px 2px 2px;">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{dot};margin-right:6px;"></span>'
        f'<span style="color:{INK_2};font-size:13px;">{txt}</span></div>',
        unsafe_allow_html=True)


# ==================== 数据上传模块（顶部横条） ====================

def render_upload_banner():
    hint_holder = st.empty()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(
            '<div class="up-head">📂 上传全部餐饮数据（CSV）'
            '<span class="up-sub">此文件应包含所有黄浦区餐饮POI记录，不要求有营业时长字段</span></div>',
            unsafe_allow_html=True)
        up = st.file_uploader("选择文件", type=["csv"], key="fup_all",
                              label_visibility="collapsed")
        if up is not None:
            ok, msg, df = _validate_upload(up.getvalue(), "all")
            st.session_state.up_all = df if ok else None
            st.session_state.up_all_state = "ok" if ok else "bad"
            st.session_state.up_all_msg = msg
            if not ok:
                st.error(msg)
        else:
            st.session_state.up_all = None
            st.session_state.up_all_state = "none"
            st.session_state.up_all_msg = ""
        _a = st.session_state.get("up_all")
        render_status(st.session_state.get("up_all_state", "none"),
                      len(_a) if _a is not None else 0)

    with col2:
        st.markdown(
            '<div class="up-head">📂 上传营业时长数据（CSV）'
            '<span class="up-sub">此文件应在全部数据基础上额外包含营业时长字段</span></div>',
            unsafe_allow_html=True)
        up = st.file_uploader("选择文件", type=["csv"], key="fup_hours",
                              label_visibility="collapsed")
        if up is not None:
            ok, msg, df = _validate_upload(up.getvalue(), "hours")
            st.session_state.up_hours = df if ok else None
            st.session_state.up_hours_state = "ok" if ok else "bad"
            st.session_state.up_hours_msg = msg
            if not ok:
                st.error(msg)
        else:
            st.session_state.up_hours = None
            st.session_state.up_hours_state = "none"
            st.session_state.up_hours_msg = ""
        _h = st.session_state.get("up_hours")
        render_status(st.session_state.get("up_hours_state", "none"),
                      len(_h) if _h is not None else 0)

    # 提示框（状态取决于上传情况 A/B/C/D）
    a_ok = st.session_state.get("up_all_state") == "ok"
    h_ok = st.session_state.get("up_hours_state") == "ok"
    if a_ok and h_ok:
        hint = "✅ 数据加载完成，全部分析模块已解锁。"
    elif a_ok:
        hint = (f"⚠️ 当前仅加载了全部数据（n={len(st.session_state['up_all'])}），"
                "营业时长相关分析模块暂不可用。上传第二个文件即可解锁全部功能。")
    elif h_ok:
        hint = (f"ℹ️ 当前仅加载了营业时长数据（n={len(st.session_state['up_hours'])}），"
                "部分统计口径可能与引用全部数据的报告存在细微差异。")
    else:
        hint = ("ℹ️ 当前使用预置示例数据（全部数据 n=7483，其中黄浦区 2671；"
                "营业时长数据 n=5867，其中黄浦区 2137）。"
                "如需分析自己的数据，请点击上方上传按钮替换。")
    with hint_holder.container():
        st.markdown(f'<div class="hint-box">{hint}</div>', unsafe_allow_html=True)


# ==================== 页面 1 · 市场总览 ====================

def page_overview(df, hdf, src_all, src_hours):
    # 行政区选择：查看任意区的市场情况（默认黄浦区，与报告口径一致）
    if "区县" in df.columns:
        districts = df["区县"].astype(str).value_counts().index.tolist()
    else:
        districts = ["全部"]
    default_idx = districts.index("黄浦区") if "黄浦区" in districts else 0
    sel_dist = st.selectbox("选择行政区", districts, index=default_idx,
                            help="切换后本页所有统计与图表同步更新")

    d = df if (sel_dist == "全部") else df[df["区县"].astype(str) == sel_dist]
    dh = None
    if hdf is not None:
        if "区县" in hdf.columns and sel_dist != "全部":
            dh = hdf[hdf["区县"].astype(str) == sel_dist]
        else:
            dh = hdf

    st.markdown(f'<div class="section-title">市场总览 · {sel_dist}大众餐饮市场（n = {len(d):,}）</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("店铺总数", f"{len(d):,}", "家", "高德地图 POI · 人均 ≤ 500 元")
    with c2:
        stat_card("人均消费中位数", f"¥{d['人均消费'].median():.0f}", "",
                  f"均值 ¥{d['人均消费'].mean():.0f} · 中低价位主导")
    with c3:
        stat_card("平均综合评分", f"{d['综合评分'].mean():.2f}", "分",
                  f"标准差 {d['综合评分'].std():.2f} · 整体口碑稳定")

    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        fig = go.Figure(go.Histogram(
            x=d["人均消费"], nbinsx=40,
            marker=dict(color=BRAND_NAVY, line=dict(width=0)),
            hovertemplate="人均消费 %{x:.0f} 元<br>店铺数 %{y}<extra></extra>",
        ))
        med = float(d["人均消费"].median())
        fig.add_vline(x=med, line_color=BRAND_GOLD, line_width=2)
        fig.add_annotation(x=med, yref="paper", y=0.98, text=f"中位数 ¥{med:.0f}",
                           showarrow=False, yanchor="top",
                           font=dict(color=BRAND_GOLD, size=13, family=FONT))
        fig.update_xaxes(range=[0, 500], title_text="人均消费（元）")
        fig.update_yaxes(title_text="店铺数")
        style_fig(fig, title="人均消费分布（20–50 元中低价位主导，右偏长尾）", height=400)
        st.plotly_chart(fig, width="stretch", key="hist")
        src_note(f"数据来源：{src_all} · {sel_dist}（n={len(d)}）")

    with right:
        pal, _ = type_palette(df)
        vc = d["类型_显示"].value_counts()
        order = [t for t in list(pal.keys()) if t in vc.index]
        fig = go.Figure(go.Pie(
            labels=order,
            values=[int(vc[t]) for t in order],
            marker=dict(colors=[pal[t] for t in order],
                        line=dict(color=SURFACE, width=2)),
            textinfo="percent",
            textfont=dict(size=13,
                          color=["#0B0B0B" if pal[t] == "#D4A853" else "#FFFFFF" for t in order]),
            sort=False,
            hovertemplate="%{label}：%{value} 家（%{percent}）<extra></extra>",
        ))
        fig.update_layout(legend=dict(title=dict(text="类型（家）"), y=0.5))
        style_fig(fig, title="餐饮类型构成（中餐 + 饮品甜点合计超 70%）", height=400)
        st.plotly_chart(fig, width="stretch", key="pie")
        src_note(f"数据来源：{src_all} · {sel_dist}（n={len(d)}）")

    if dh is not None and len(dh):
        fig = go.Figure(go.Histogram(
            x=dh["营业时长"].dropna(), nbinsx=30,
            marker=dict(color=SERIES_NAVY, line=dict(width=0)),
            hovertemplate="营业时长 %{x:.1f} 小时<br>店铺数 %{y}<extra></extra>",
        ))
        medh = float(dh["营业时长"].median())
        fig.add_vline(x=medh, line_color=BRAND_GOLD, line_width=2)
        fig.add_annotation(x=medh, yref="paper", y=0.98, text=f"中位数 {medh:.0f} 小时",
                           showarrow=False, yanchor="top",
                           font=dict(color=BRAND_GOLD, size=13, family=FONT))
        fig.update_xaxes(range=[0, 24], title_text="营业时长（小时）")
        fig.update_yaxes(title_text="店铺数")
        style_fig(fig, title="营业时长分布（主流 12–14 小时，全天候业态稀少）", height=380)
        st.plotly_chart(fig, width="stretch", key="hist_hours")
        src_note(f"数据来源：{src_hours} · {sel_dist}（n={len(dh)}）")
    else:
        placeholder_box("此分析需要营业时长数据，请上传包含营业时长字段的CSV文件以解锁。")

    st.caption("数据来源：高德地图 POI 餐饮类目（types=050000），采集于 2026 年 3–4 月；"
               "类型按原始编码合并为 16 类，本页展示前 5 类，其余并入「其他」。")


# ==================== 页面 2 · 商圈对比（动态热力图） ====================

def _price_color(v, q):
    if v <= q[0]:
        return PRICE_RAMP[0]
    if v <= q[1]:
        return PRICE_RAMP[1]
    if v <= q[2]:
        return PRICE_RAMP[2]
    return PRICE_RAMP[3]


# ==================== 自包含地图渲染（不依赖任何外部 CDN） ====================

_TILE_FALLBACK_JS = """<script>
(function(){
  function getMap(){
    for (var k in window) {
      if (k.indexOf('map_') === 0 && window[k] && typeof window[k].addLayer === 'function') return window[k];
    }
    return null;
  }
  function install(){
    var map = getMap();
    if (!map || window.__tilesInstalled) return;
    window.__tilesInstalled = true;
    var providers = [
      {url:'https://webrd0{s}.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}',
       opt:{subdomains:'1234', maxZoom:18}},
      {url:'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
       opt:{subdomains:'abcd', maxZoom:19}},
      {url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
       opt:{maxZoom:19}}
    ];
    var idx = 0, loaded = false, switching = false, layer = null;
    function next(){
      if (loaded || switching || idx >= providers.length) return;
      switching = true;
      var p = providers[idx++];
      if (layer) { try { map.removeLayer(layer); } catch(e) {} }
      try {
        layer = L.tileLayer(p.url, p.opt);
        layer.on('load', function(){ loaded = true; });
        layer.addTo(map);
      } catch(e) { layer = null; }
      setTimeout(function(){ switching = false; next(); }, 6000);
    }
    next();
  }
  if (document.readyState === 'complete') install();
  else window.addEventListener('load', install);
  setTimeout(install, 2500);
})();
</script>
"""

_LEAFLET_B64 = {}


def _asset_b64(fname):
    """读取本地 assets 文件并 base64 编码（缓存）。缺失时返回空串。"""
    if fname not in _LEAFLET_B64:
        p = os.path.join(BASE_DIR, "assets", fname)
        try:
            with open(p, "rb") as f:
                _LEAFLET_B64[fname] = base64.b64encode(f.read()).decode("ascii")
        except OSError:
            _LEAFLET_B64[fname] = ""
    return _LEAFLET_B64[fname]


def _map_html(m):
    """把 folium 地图转换为自包含 HTML：
    - Leaflet JS/CSS 用 data URI 内联（本地 assets/leaflet.js、leaflet.css）
    - 移除 folium 默认引用的被墙 CDN 脚本（jquery/bootstrap/awesome-markers 均未使用）
    - 注入瓦片降级链：高德 → CartoDB → OSM，保证在国内网络下也能显示底图
    """
    h = m._repr_html_()
    m2 = re.search(r'<iframe[^>]*srcdoc="([^"]*)"', h, re.S)
    if not m2:
        return h
    doc = _html_lib.unescape(m2.group(1))

    js_b64 = _asset_b64("leaflet.js")
    css_b64 = _asset_b64("leaflet.css")
    jq_b64 = _asset_b64("jquery.min.js")
    heat_b64 = _asset_b64("leaflet.heat.min.js")
    if js_b64:
        doc = re.sub(
            r'<script src="https://cdn\.jsdelivr\.net/npm/leaflet@[\d.]+/dist/leaflet\.js"></script>',
            '<script src="data:text/javascript;base64,' + js_b64 + '"></script>', doc)
    if css_b64:
        doc = re.sub(
            r'<link rel="stylesheet" href="https://cdn\.jsdelivr\.net/npm/leaflet@[\d.]+/dist/leaflet\.css"/>',
            '<link rel="stylesheet" href="data:text/css;base64,' + css_b64 + '"/>', doc)
    if jq_b64:
        # folium 的弹窗绑定依赖 jQuery（$(...)[0]），必须内联保留
        doc = re.sub(
            r'<script src="https://code\.jquery\.com/[^"]*"></script>',
            '<script src="data:text/javascript;base64,' + jq_b64 + '"></script>', doc)
    if heat_b64:
        # leaflet.heat 插件库：folium 从 jsdelivr-gh CDN 引用（被墙），改为本地内联
        doc = re.sub(
            r'<script src="https://cdn\.jsdelivr\.net/gh/[^"]*leaflet_heat\.min\.js"></script>',
            '<script src="data:text/javascript;base64,' + heat_b64 + '"></script>', doc)
    doc = re.sub(r'<script src="https://cdn\.jsdelivr\.net/npm/bootstrap[^"]*"></script>', '', doc)
    doc = re.sub(r'<script src="https://cdnjs\.cloudflare\.com[^"]*awesome-markers[^"]*"></script>', '', doc)
    doc = re.sub(r'<link rel="stylesheet" href="https://cdnjs\.cloudflare\.com[^"]*awesome-markers[^"]*"/>', '', doc)
    doc = re.sub(r'<link rel="stylesheet" href="https://cdn\.jsdelivr\.net/gh/[^"]*"/>', '', doc)

    # 白色画布背景（无瓦片时不再显示灰色底）+ 缩放范围锁定脚本
    doc = doc.replace("</head>",
                      "<style>html,body{background:#FFFFFF;}"
                      ".leaflet-container{background:#FFFFFF;}</style></head>")
    zoom_clamp = """
<script>
(function(){
  var done = false;
  function clamp(){
    if (done) return;
    var ks = Object.keys(window).filter(function(k){return k.indexOf('map_')===0;});
    var map = ks.length ? window[ks[0]] : null;
    if (!map || !map.getZoom) return;
    done = true;
    var z = map.getZoom();
    map.setMinZoom(Math.max(z - 0.6, map.getMinZoom()));
    map.setMaxZoom(z + 4);
  }
  if (document.readyState === 'complete') clamp();
  else window.addEventListener('load', clamp);
  setTimeout(clamp, 2500);
})();
</script>
"""
    doc = doc.replace("</body>", zoom_clamp + "</body>")
    if TILES_ENABLED:
        doc = doc.replace("</body>", _TILE_FALLBACK_JS + "</body>")
    # 外层文档 body 必须有明确高度，内层 iframe 的 height:100% 才会生效
    # （否则浏览器按 iframe 默认高度 150px 渲染，地图纵向会非常短）
    return ('<style>html,body{height:100%;margin:0;padding:0;}</style>'
            '<iframe srcdoc="' + _html_lib.escape(doc)
            + '" style="width:100%;height:100%;border:none;display:block;"></iframe>')


def _thesis_grid_layer(m, df):
    """论文图4-13同款：完整矩形网格矩阵（按经纬度极值范围划分，含空白格），
    每格按人均消费均值着色；样本少于 3 家的格子保留为白色（矩阵仍是矩形）。"""
    cell = 0.0042  # 约 440m 网格
    lat_min, lat_max = float(df["纬度"].min()), float(df["纬度"].max())
    lng_min, lng_max = float(df["经度"].min()), float(df["经度"].max())
    lats = df["纬度"].values
    lngs = df["经度"].values
    prices = df["人均消费"].values
    # 论文配色（从图4-13提取）：淡黄 → 金黄 → 橙
    ramp = ["#F7F3C4", "#F0E090", "#EFD070", "#F0B050", "#EE9040"]
    bounds = [50, 100, 160, 250]  # 人均消费分档（元）

    def cell_color(mean_p):
        for i, b in enumerate(bounds):
            if mean_p <= b:
                return ramp[i]
        return ramp[-1]

    for lat0 in np.arange(lat_min, lat_max, cell):
        for lng0 in np.arange(lng_min, lng_max, cell):
            mask = (lats >= lat0) & (lats < lat0 + cell) & (lngs >= lng0) & (lngs < lng0 + cell)
            n = int(mask.sum())
            if n == 0:
                continue  # 极值范围外留空（瓦片底色）
            if n < 3:
                # 少于3家：白色占位，保持矩阵矩形
                folium.Rectangle(
                    bounds=[[lat0, lng0],
                            [min(lat0 + cell, lat_max), min(lng0 + cell, lng_max)]],
                    color=None, weight=0, fill=True,
                    fill_color="#FFFFFF", fill_opacity=0.9,
                    tooltip=folium.Tooltip(f"{n} 家（样本过少未纳入分析）"),
                ).add_to(m)
                continue
            mean_p = float(prices[mask].mean())
            folium.Rectangle(
                bounds=[[lat0, lng0],
                        [min(lat0 + cell, lat_max), min(lng0 + cell, lng_max)]],
                color=None, weight=0, fill=True,
                fill_color=cell_color(mean_p), fill_opacity=0.75,
                tooltip=folium.Tooltip(f"{n} 家店铺 · 人均均值 ¥{mean_p:.0f}"),
            ).add_to(m)


def build_map(df, sel):
    """论文图4-13同款热力图矩阵 + 金色商圈高亮。
    画布即图：maxBounds 锁死图边界（viscosity 1.0），minZoom=铺满视图（不能缩出灰边），
    滚轮不缩放地图（避免劫持页面滚动），放大后只能在图内拖动。"""
    clat = float(df["纬度"].mean()) if len(df) else 31.219
    clng = float(df["经度"].mean()) if len(df) else 121.485
    lat_min, lat_max = float(df["纬度"].min()), float(df["纬度"].max())
    lng_min, lng_max = float(df["经度"].min()), float(df["经度"].max())

    pad = 0.0015  # 极小边距（约 160m），图边界即画布边界
    m = folium.Map(
        location=[clat, clng], zoom_start=13, tiles=None, control_scale=True,
        scroll_wheel_zoom=False,
        max_bounds=[[lat_min - pad, lng_min - pad], [lat_max + pad, lng_max + pad]],
        max_bounds_viscosity=1.0,
    )
    sel_df = df[df["商圈"] == sel]

    # 热力图矩阵（论文图4-13同款）
    _thesis_grid_layer(m, df)

    # 选中商圈：金色点位 + 虚线范围圈 + 标签
    for _, r in sel_df.iterrows():
        folium.CircleMarker(
            location=[r["纬度"], r["经度"]], radius=5,
            color=BRAND_NAVY, weight=1.5, fill=True,
            fill_color=BRAND_GOLD, fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>{r['店铺名称']}</b><br>{r['类型_合并']} · {r['商圈']}<br>"
                f"人均 ¥{r['人均消费']:.0f} · 评分 {r['综合评分']}",
                max_width=260),
        ).add_to(m)
    if len(sel_df):
        clat, clng = float(sel_df["纬度"].mean()), float(sel_df["经度"].mean())
        dist_deg = float(np.max(np.hypot(sel_df["纬度"] - clat, sel_df["经度"] - clng)))
        folium.Circle([clat, clng], radius=max(dist_deg * 111320 * 1.15, 500),
                      color=BRAND_GOLD, weight=1.5, dash_array="4 4",
                      fill=True, fill_opacity=0.10).add_to(m)
        folium.Marker(
            [clat, clng],
            icon=folium.DivIcon(html=f"""
                <div style="background:{BRAND_NAVY}; color:{BRAND_GOLD}; font-size:13px;
                            font-weight:700; padding:3px 10px; border-radius:12px;
                            border:1.5px solid {BRAND_GOLD}; white-space:nowrap;">
                  {sel}商圈（{len(sel_df)} 家）</div>"""),
        ).add_to(m)

    legend = f"""
    <div style="position:fixed; bottom:28px; left:10px; z-index:9999;
                background:{SURFACE}; border:1px solid {GRID}; border-radius:6px;
                padding:8px 12px; font-family:{FONT}; font-size:12px; color:{INK_2};">
      <b style="color:{BRAND_NAVY};">网格人均消费均值（元）</b><br>
      <span style="background:linear-gradient(to right,#F7F3C4,#F0E090,#EFD070,#F0B050,#EE9040);
                   display:inline-block;width:110px;height:10px;border-radius:2px;"></span><br>
      ≤50 &nbsp; ≤100 &nbsp; ≤160 &nbsp; ≤250 &nbsp; &gt;250<br>
      <span style="color:#FFFFFF;text-shadow:0 0 2px #888;">□</span> 少于3家的格子（白色）<br>
      <span style="color:{BRAND_GOLD};">●</span> 选中商圈（悬停看店铺）
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))

    # 分析区域铺满视野
    m.fit_bounds([[lat_min, lng_min], [lat_max, lng_max]],
                 padding_top_left=(12, 12), padding_bottom_right=(12, 12))
    return m


def page_district(df, src_all):
    st.markdown('<div class="section-title">商圈对比 · 商圈价格与口碑画像</div>',
                unsafe_allow_html=True)
    opts = [b for b in df["商圈"].value_counts().index.tolist() if b != "未知"]
    if not opts:
        opts = ["全部"]
    default_idx = opts.index("淮海路") if "淮海路" in opts else 0
    sel = st.selectbox("选择商圈", opts, index=default_idx,
                       help="切换后地图会高亮该商圈，下方图表同步更新")
    d = df[df["商圈"] == sel]

    # 画布=图的边界：按黄浦区经纬度极值的墨卡托长宽比计算画布宽度，图居中显示
    lat_min, lat_max = float(df["纬度"].min()), float(df["纬度"].max())
    lng_min, lng_max = float(df["经度"].min()), float(df["经度"].max())
    merc_w = (lng_max - lng_min) * math.cos(math.radians((lat_min + lat_max) / 2))
    aspect = merc_w / max(lat_max - lat_min, 1e-6)
    map_h = 600
    map_w = int(min(960, max(340, map_h * aspect * 1.18)))
    # 画布=图边界：宽度按图长宽比计算，容器居中显示
    components.html(
        _map_html(build_map(df, sel)),
        width=map_w, height=map_h, scrolling=False)
    src_note(f"数据来源：{src_all} · 黄浦区（n={len(df)}）· "
             "矩形网格矩阵（格内人均消费均值，黄→橙），白色=样本少于3家，金色=选中商圈；"
             "用右上角 +/- 缩放，放大后可在图内拖动")

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("店铺数", f"{len(d)}", "家", f"占全区 {len(d)/max(len(df),1)*100:.1f}%")
    with c2:
        stat_card("人均中位数", f"¥{d['人均消费'].median():.0f}", "",
                  f"全区 ¥{df['人均消费'].median():.0f}")
    with c3:
        stat_card("平均评分", f"{d['综合评分'].mean():.2f}", "分",
                  f"全区 {df['综合评分'].mean():.2f}")

    ch1, ch2, ch3 = st.columns(3, gap="medium")
    with ch1:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=df["人均消费"], name=f"全区（{len(df)} 家）",
            line_color="#9AA0A6", fillcolor="rgba(154,160,166,0.25)",
            opacity=0.9, box_visible=True, meanline_visible=True, points=False,
            line_width=1.5, hovertemplate="全区<br>人均 %{y:.0f} 元<extra></extra>"))
        fig.add_trace(go.Violin(
            y=d["人均消费"], name=f"{sel}（{len(d)} 家）",
            line_color=SERIES_NAVY, fillcolor="rgba(46,109,163,0.28)",
            opacity=0.95, box_visible=True, meanline_visible=True, points=False,
            line_width=2.5, hovertemplate=f"{sel}<br>人均 %{{y:.0f}} 元<extra></extra>"))
        fig.update_yaxes(title_text="人均消费（元）", range=[0, 500])
        style_fig(fig, title=f"人均消费小提琴图 · {sel} vs 全区", height=360)
        st.plotly_chart(fig, width="stretch", key="violin")
        src_note(f"数据来源：{src_all}")

    with ch2:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df["综合评分"], name=f"全区（{len(df)} 家）",
            marker_color="#9AA0A6", fillcolor="rgba(154,160,166,0.25)",
            line_color="#9AA0A6", boxpoints="suspectedoutliers",
            marker=dict(size=3, opacity=0.5),
            hovertemplate="全区<br>评分 %{y}<extra></extra>"))
        fig.add_trace(go.Box(
            y=d["综合评分"], name=f"{sel}（{len(d)} 家）",
            marker_color=SERIES_NAVY, fillcolor="rgba(46,109,163,0.28)",
            line_color=SERIES_NAVY, boxpoints="suspectedoutliers",
            marker=dict(size=3, opacity=0.5),
            hovertemplate=f"{sel}<br>评分 %{{y}}<extra></extra>"))
        fig.update_yaxes(title_text="综合评分")
        style_fig(fig, title=f"综合评分箱线图 · {sel} vs 全区", height=360)
        st.plotly_chart(fig, width="stretch", key="box")
        src_note(f"数据来源：{src_all}")

    with ch3:
        pal, _ = type_palette(df)
        vc = d["类型_显示"].value_counts()
        order = sorted([t for t in pal if t in vc.index], key=lambda t: vc[t], reverse=True)
        fig = go.Figure(go.Bar(
            x=[int(vc[t]) for t in order], y=order, orientation="h",
            marker=dict(color=[pal[t] for t in order], line=dict(width=0)),
            text=[str(int(vc[t])) for t in order], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}：%{x} 家<extra></extra>",
        ))
        fig.update_xaxes(title_text="店铺数")
        fig.update_yaxes(title=None)
        style_fig(fig, title=f"品类构成 · {sel}", height=360)
        st.plotly_chart(fig, width="stretch", key="hbar")
        src_note(f"数据来源：{src_all}")


# ==================== 页面 3 · 业态聚类 ====================

CLUSTER_NOTES = {
    "大众高效型": "低价位 × 高评分 × 常规营业时长——以 40–70 元的大众刚需餐饮为主，"
                  "「高质价比」是黄浦区最大基本盘（占 42.9%）。",
    "全天候口碑型": "低价位 × 高评分 × 超长营业时长（均值 16.6 小时）——"
                    "覆盖夜宵/早餐时段的口碑店铺，长营业时间是核心竞争壁垒。",
    "低价低质型": "低人均 × 低评分——价格敏感型客群的快餐小店，"
                  "评分显著低于市场均值（3.95 vs 4.23），存在质量断层。",
    "高端精品型": "高消费（均值 ¥296.6）× 高评分 × 短营业时长——"
                  "精品餐饮，以晚餐时段为主，占 9.4%，客单价是大众型的 4 倍以上。",
}


def page_cluster(hdf, src_hours):
    st.markdown('<div class="section-title">业态聚类 · K-Means 四类店铺画像</div>',
                unsafe_allow_html=True)
    summary = get_cluster_summary_cached(hdf)
    if summary is None or summary.empty:
        placeholder_box("有效样本不足，无法执行聚类分析。请检查营业时长数据的完整性。")
        return
    st.caption("K-Means（K=4，特征：人均消费 / 综合评分 / 营业时长，Z-Score 标准化）。"
               "点击按钮高亮对应业态：")

    names = [n for n in CLUSTER_PREFER if n in summary["聚类名称"].values]
    names += [n for n in summary["聚类名称"] if n not in names]
    if "cluster_sel" not in st.session_state or st.session_state.cluster_sel not in names:
        st.session_state.cluster_sel = names[0]

    cols = st.columns(len(names))
    for i, name in enumerate(names):
        pct = float(summary[summary["聚类名称"] == name]["占比%"].iloc[0])
        cols[i].button(
            f"{name} · {pct:.0f}%",
            use_container_width=True,
            type="primary" if name == st.session_state.cluster_sel else "secondary",
            key=f"btn_{name}",
            on_click=lambda n=name: st.session_state.update(cluster_sel=n),
        )

    sel = st.session_state.cluster_sel

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        cats = ["人均消费", "综合评分", "营业时长"]
        fig = go.Figure()
        for name in names:
            row = summary[summary["聚类名称"] == name].iloc[0]
            vals = [float(row[f"z{c}"]) for c in cats]
            active = (name == sel)
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]], name=name,
                fill="toself", fillcolor=CLUSTER_COLORS.get(name, SERIES_NAVY),
                opacity=0.5 if active else 0.10,
                line=dict(color=CLUSTER_COLORS.get(name, SERIES_NAVY),
                          width=3.5 if active else 1.4),
                hovertemplate="%{theta}：%{r:.2f}（Z 分数）<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[-1.8, 2.9], gridcolor=GRID, linecolor=AXIS,
                               tickfont=dict(size=10)),
                angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(size=13)),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
        )
        style_fig(fig, title="四类业态雷达图（Z-Score 中心）", height=480)
        st.plotly_chart(fig, width="stretch", key="radar")
        src_note(f"数据来源：{src_hours} · 黄浦区（n={len(hdf)}）")

    with right:
        st.markdown(f'<div class="section-title">业态明细 · {sel}</div>',
                    unsafe_allow_html=True)

        def _hl(row):
            bg = "background-color: #F7ECD3; font-weight: 600;" if row["业态"] == sel else ""
            return [bg] * len(row)

        show = summary[["聚类名称", "样本量", "占比%", "人均消费均值", "综合评分均值", "营业时长均值"]].copy()
        show.columns = ["业态", "样本量（家）", "占比（%）", "人均消费（元）", "综合评分（分）", "营业时长（小时）"]
        st.dataframe(show.style.apply(_hl, axis=1), hide_index=True)
        src_note(f"数据来源：{src_hours} · 黄浦区（n={len(hdf)}）")

        st.info(f"**{sel}**：{CLUSTER_NOTES.get(sel, '该业态由数据聚类自动识别，特征见上方明细表。')}")


# ==================== 页面 4 · 定价策略 ====================

def page_pricing(df, hdf, mode, src_all, src_hours):
    st.markdown('<div class="section-title">定价策略 · 价格与口碑的边际贡献</div>',
                unsafe_allow_html=True)
    price = int(st.session_state.get("price_slider", 120))
    curve, is_real = get_shap_data(mode)

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        pal, _ = type_palette(df)
        vc = df["类型_显示"].value_counts()
        order = sorted([t for t in pal if t in vc.index], key=lambda t: vc[t], reverse=True)
        fig = go.Figure()
        for t in order:
            sub = df[df["类型_显示"] == t]
            fig.add_trace(go.Scatter(
                x=sub["人均消费"], y=sub["综合评分"], mode="markers",
                name=f"{t}（{len(sub)} 家）",
                marker=dict(color=pal[t], size=5, opacity=0.5, line=dict(width=0)),
                customdata=np.stack([sub["店铺名称"], sub["类型_合并"]], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                              "人均 ¥%{x:.0f} · 评分 %{y}<extra></extra>",
            ))
        anchor_y = float(df[(df["人均消费"] >= 110) & (df["人均消费"] <= 130)]["综合评分"].mean()) \
            if len(df[(df["人均消费"] >= 110) & (df["人均消费"] <= 130)]) else 4.2
        fig.add_trace(go.Scatter(
            x=[120], y=[anchor_y], mode="markers", name="边际贡献峰值（120 元）",
            marker=dict(symbol="diamond", size=16, color=BRAND_GOLD,
                        line=dict(color=BRAND_NAVY, width=1.5)),
            customdata=[["锚点", "边际贡献峰值"]],
            hovertemplate="<b>边际贡献峰值</b><br>人均 ¥120 · 点击查看注释<extra></extra>",
        ))
        fig.add_vline(x=price, line_color=RED_LINE, line_width=2, line_dash="dash")
        fig.add_annotation(x=price, yref="paper", y=0.99, yanchor="bottom", text=f"¥{price}",
                           showarrow=False, font=dict(color=RED_LINE, size=13, family=FONT))
        fig.add_annotation(x=120, y=anchor_y, text="边际贡献峰值", ax=0, ay=40,
                           showarrow=True, arrowhead=2, arrowcolor=BRAND_GOLD, arrowwidth=2,
                           font=dict(color=BRAND_NAVY, size=12, family=FONT))
        fig.update_xaxes(range=[-15, 508], title_text="人均消费（元）")
        fig.update_yaxes(title_text="综合评分")
        style_fig(fig, title="人均消费 vs 综合评分（按品类着色，点击金色锚点查看注释）", height=440)
        st.plotly_chart(fig, width="stretch", key="scatter_anchor",
                        on_select="rerun", selection_mode="points")
        src_note(f"数据来源：{src_all} · 黄浦区（n={len(df)}）")

        evt = st.session_state.get("scatter_anchor")
        if evt and evt.get("selection", {}).get("points"):
            if any(p.get("curve_number") == len(order) and p.get("point_index") == 0
                   for p in evt["selection"]["points"]):
                st.session_state.anchor_note = True
        if st.session_state.get("anchor_note"):
            st.info("💡 **边际贡献峰值**：人均消费约 120 元处，价格对评分的正向边际贡献达到最大"
                    "——该价位带是「质价比」的甜点区间，低于它价格信号快速衰减，高于它边际回报递减。")

        st.slider("拖动查看不同价格点的 SHAP 边际贡献", 0, 500, 120, key="price_slider")

    with right:
        x = curve["price_bin_center"].values.astype(float)
        y = curve["shap_mean"].values.astype(float)
        val = float(np.interp(min(max(price, x.min()), x.max()), x, y))
        val_color = "#B8860B" if val > 0 else SERIES_NAVY

        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-label">当前价格点 ¥{price} 的 SHAP 边际贡献</div>
          <div class="shap-value" style="color:{val_color};">{val:+.3f}</div>
          <div class="stat-note">对综合评分的贡献（分）· 正值为正向影响</div>
        </div>
        """, unsafe_allow_html=True)

        smooth = pd.Series(y).rolling(3, min_periods=1, center=True).mean().values
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=smooth, mode="lines", name="SHAP 贡献曲线",
            line=dict(color=SERIES_NAVY, width=2.5),
            fill="tozeroy", fillcolor="rgba(46,109,163,0.08)",
            hovertemplate="人均 %{x:.0f} 元<br>SHAP %{y:+.3f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=AXIS, line_width=1)
        fig.add_trace(go.Scatter(
            x=[120], y=[float(np.interp(120, x, smooth))], mode="markers",
            name="边际贡献峰值", marker=dict(symbol="diamond", size=13,
                        color=BRAND_GOLD, line=dict(color=BRAND_NAVY, width=1.5)),
            hovertemplate="<b>边际贡献峰值</b><br>¥120 · SHAP %{y:+.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[price], y=[val], mode="markers", name=f"当前 ¥{price}",
            marker=dict(size=12, color=RED_LINE, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=f"¥{price} · SHAP %{{y:+.3f}}<extra></extra>",
        ))
        fig.update_xaxes(title_text="人均消费（元）")
        fig.update_yaxes(title_text="SHAP 贡献")
        style_fig(fig, title="价格 → SHAP 边际贡献曲线", height=330)
        st.plotly_chart(fig, width="stretch", key="shap_line")
        src_note(f"数据来源：{src_hours} · 黄浦区（n={len(hdf)}）"
                 + ("（分箱均值）" if is_real else "（模拟曲线）"))

        if is_real:
            st.caption("说明：SHAP 值来自随机森林模型（n=1,968，含品类/商圈/营业时长协变量）"
                       "在测试集上的依赖值分箱均值；模型 R²=0.023，曲线用于演示方法与方向，"
                       "量级仅供课堂展示参考。")
        else:
            st.caption("说明：上传数据未重新训练随机森林模型，SHAP 曲线为模拟数据"
                       "（峰值约在 120 元，与既有研究结论一致），仅用于演示交互流程。")


# ==================== 页面 5 · 空间格局 ====================

def page_spatial(df, src_all, mode):
    st.markdown('<div class="section-title">空间格局 · 价格集聚与商圈分化</div>',
                unsafe_allow_html=True)

    moran_path = os.path.join(BASE_DIR, "moran_scatter.png")
    lisa_path = os.path.join(BASE_DIR, "lisa_map.png")
    txt_path = os.path.join(BASE_DIR, "莫兰指数结果.txt")
    i_val = "0.3088"
    if os.path.exists(txt_path):
        try:
            with open(txt_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("莫兰指数 I:"):
                        i_val = line.split(":")[1].strip()
        except Exception:
            pass

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(moran_path):
            st.image(moran_path, width="stretch",
                     caption=f"全局莫兰散点图 · Moran's I = {i_val}（p < 0.001）")
        else:
            st.warning("未找到 moran_scatter.png，请将其放到与本文件相同的目录。")
    with c2:
        if os.path.exists(lisa_path):
            st.image(lisa_path, width="stretch",
                     caption="局部空间自相关（LISA）集聚图 · 显著点 p < 0.05")
        else:
            st.warning("未找到 lisa_map.png，请将其放到与本文件相同的目录。")

    src_note(f"数据来源：{src_all} · 黄浦区（n={len(df)}）")
    if mode != "D":
        st.caption("注：空间统计图基于预置数据（n=2,671）生成；上传数据如需重新计算，"
                   "请运行莫兰指数分析脚本后替换图片。")

    st.markdown(f"""
    **解读（与报告一致）：**

    人均消费呈**显著空间正自相关**（全局莫兰指数 I = {i_val}，Z = 34.4，p < 0.001），
    说明相似价位段的店铺在地理上集聚而非随机分布：

    - **高-高集聚（HH，169 家）**：北京路、城隍庙等核心商圈为高消费集聚区，精品餐饮沿主干道连片分布；
    - **低-低集聚（LL，392 家）**：南部社区（五里桥、制造局路一带）形成低值连绵带，以大众刚需餐饮为主；
    - **低-高（LH，98 家）/ 高-低（HL，57 家）**：出现在商圈边界与大型商业体周边，
      反映「高消费商圈内部的价格洼地」与「低值带中的品质孤岛」两类过渡形态。

    空间结构对选址的启示：核心商圈租金高、客流足，适合高客单精品业态；
    南部低值带竞争以价格为主，适合高翻台的大众餐饮。
    """)


# ==================== 主入口 ====================

def main():
    st.set_page_config(page_title="黄浦区餐饮市场分析仪表盘",
                       page_icon="🍜", layout="wide")
    inject_css()

    st.sidebar.markdown(
        '<div class="brand-title">🍜 黄浦区餐饮市场<br>分析仪表盘</div>'
        '<div class="brand-sub">基于高德地图 POI 数据的<br>价格 · 品类 · 商圈 · 空间分析</div>',
        unsafe_allow_html=True)

    page = st.sidebar.radio(
        "页面导航",
        ["市场总览", "商圈对比", "业态聚类", "定价策略", "空间格局"],
        label_visibility="collapsed")

    render_upload_banner()

    try:
        df_main, df_hours, mode = resolve_data()
    except FileNotFoundError as e:
        st.error(f"⚠️ 数据加载失败：{e}")
        st.stop()
    df_main = apply_type_display(df_main)
    if df_hours is not None:
        df_hours = apply_type_display(df_hours)

    has_hours = df_hours is not None
    # 商圈/聚类/定价/空间分析保持黄浦区口径（与报告一致）；市场总览可切换行政区
    hp_main = filter_huangpu(df_main)
    hp_hours = filter_huangpu(df_hours) if has_hours else None
    src_all = "营业时长数据" if mode == "C" else "全部数据"
    src_hours = "营业时长数据"

    if page == "市场总览":
        page_overview(df_main, df_hours, src_all, src_hours)
    elif page == "商圈对比":
        page_district(hp_main, src_all)
    elif page == "业态聚类":
        if has_hours:
            page_cluster(hp_hours, src_hours)
        else:
            placeholder_box("此分析需要营业时长数据，请上传包含营业时长字段的CSV文件以解锁。")
    elif page == "定价策略":
        if has_hours:
            page_pricing(hp_main, hp_hours, mode, src_all, src_hours)
        else:
            placeholder_box("此分析需要营业时长数据，请上传包含营业时长字段的CSV文件以解锁。")
    else:
        page_spatial(hp_main, src_all, mode)

    st.markdown("---")
    st.caption("数据：高德地图 POI（黄浦区餐饮类目，人均 ≤ 500 元）；"
               "方法：K-Means 聚类、随机森林 + SHAP、全局/局部莫兰指数（KNN k=8）。"
               "图表配色已通过色觉障碍（CVD）可区分性校验。")


if os.environ.get("SMOKE_TEST") == "1":
    # 冒烟测试：裸模式逐页构建 + 校验逻辑自测
    import warnings
    warnings.filterwarnings("ignore")

    df = apply_type_display(load_main())
    hdf = apply_type_display(load_hours())

    # 校验逻辑自测
    bad = _validate_upload(b"a,b,c,d,e,f,g\n1,2,3,4,5,6,7\n", "all")
    assert bad[0] is False and "缺少" in bad[1], "bad csv should fail with missing-col msg"
    good = _validate_upload(open(r"E:\期末大作业\数据文件\餐饮数据_清洗后_全部.csv", "rb").read(), "all")
    assert good[0] is True, "default all csv should pass"
    goodh = _validate_upload(open(r"E:\期末大作业\数据文件\餐饮数据_清洗后_有时长.csv", "rb").read(), "hours")
    assert goodh[0] is True, "default hours csv should pass"
    badh = _validate_upload(good[2].to_csv(index=False).encode("utf-8-sig"), "hours")
    assert badh[0] is False and "营业时长" in badh[1], "hours upload without 营业时长 col should fail"
    print("validation self-tests OK")

    # 聚类与 SHAP
    s = compute_cluster_summary(hdf)
    assert s is not None and len(s) == 4, "cluster summary failed"
    c_real, is_real = get_shap_data("D")
    c_sim, is_sim = get_shap_data("C")
    assert is_real and not is_sim and len(c_real) > 10 and len(c_sim) == 100
    print("cluster + shap OK")

    st.session_state["price_slider"] = 120
    st.session_state["cluster_sel"] = "高端精品型"
    page_overview(df, hdf, "全部数据", "营业时长数据")
    hp = filter_huangpu(df)
    hph = filter_huangpu(hdf)
    assert len(hp) == 2671 and len(hph) == 2137, "黄浦区切片行数不符"
    for d_ in ["北京路", "淮海路", "南浦大桥"]:
        mh = _map_html(build_map(hp, d_))
        assert "data:text/javascript;base64," in mh, "leaflet js not inlined"
        assert "data:text/css;base64," in mh, "leaflet css not inlined"
        assert "cdn.jsdelivr.net/npm/leaflet" not in mh, "CDN js leftover"
        assert "code.jquery.com" not in mh, "CDN jquery leftover"
        assert "leaflet-container{background:#FFFFFF;}" in mh, "white canvas bg missing"
        assert "setMinZoom" in mh, "zoom clamp missing"
        assert "data:text/javascript;base64," + _asset_b64("jquery.min.js")[:20] in mh, "jquery not inlined"
        assert "cdn.jsdelivr.net/gh" not in mh, "gh CDN leftover"
        assert "cdn.jsdelivr.net/npm/leaflet@" not in mh, "leaflet CDN leftover"
        assert '<iframe srcdoc=' in mh
    print("self-contained map OK")
    page_district(hp, "全部数据")
    page_cluster(hph, "营业时长数据")
    page_pricing(hp, hph, "D", "全部数据", "营业时长数据")
    page_spatial(hp, "全部数据", "D")
    print("SMOKE OK: all pages built without exception")
else:
    main()
