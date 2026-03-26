"""
demo.py — Anomaly Detection Dashboard
=======================================
Launch:  python -m streamlit run demo.py
"""

import os, sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colors ────────────────────────────────────────────────────────────────────
C = {
    'AT': '#4285F4',
    'TranAD': '#34A853',
    'anomaly': '#EA4335',
    'price': '#202124',
    'bg': '#FFFFFF',
    'card': '#F8F9FA',
    'border': '#DADCE0',
    'text': '#202124',
    'text2': '#5F6368',
    'accent': '#1A73E8',
}

# ── Light-mode CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background: #F8F9FA; }

    /* Page title */
    .page-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #202124;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #5F6368;
        margin: 0;
    }

    /* Expander — force readable text */
    div[data-testid="stExpander"] { background: white; border: 1px solid #DADCE0; border-radius: 10px; margin-bottom: 0.5rem; }
    div[data-testid="stExpander"] summary { font-size: 1rem !important; font-weight: 700 !important; color: #202124 !important; padding: 0.7rem 1rem !important; }
    div[data-testid="stExpander"] summary span { color: #202124 !important; }
    div[data-testid="stExpander"] summary svg { color: #202124 !important; }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] { font-size: 0.92rem; line-height: 1.65; color: #202124 !important; padding: 0.5rem 1rem 1rem 1rem; }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] li,
    div[data-testid="stExpander"] td, div[data-testid="stExpander"] th,
    div[data-testid="stExpander"] h1, div[data-testid="stExpander"] h2,
    div[data-testid="stExpander"] h3, div[data-testid="stExpander"] h4 { color: #202124 !important; }
    div[data-testid="stExpander"] table { font-size: 0.88rem; }
    div[data-testid="stExpander"] blockquote { border-left: 3px solid #1A73E8; padding-left: 0.8rem; color: #333 !important; }

    .section-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #202124;
        margin: 0.8rem 0 0.5rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid #E8EAED;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #DADCE0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { color: #5F6368 !important; font-size: 0.82rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #1A73E8 !important; font-size: 1.5rem !important; }

    /* Model header in events panel */
    .model-header {
        font-size: 0.92rem;
        font-weight: 700;
        padding: 0.3rem 0;
        margin-top: 0.4rem;
    }

    .event-row {
        background: white;
        border: 1px solid #DADCE0;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.45rem;
        font-size: 0.9rem;
        line-height: 1.45;
        transition: box-shadow 0.15s;
    }
    .event-row:hover { box-shadow: 0 2px 8px rgba(26,115,232,0.12); border-color: #4285F4; }
    .event-date { font-weight: 600; color: #202124; font-size: 0.92rem; }
    .event-score { color: #EA4335; font-weight: 600; font-size: 0.88rem; }
    .event-headline { color: #5F6368; font-size: 0.85rem; margin-top: 3px; line-height: 1.4; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Data Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def discover_tickers():
    d = PROJECT_ROOT / "results" / "pipeline"
    return sorted([p.name for p in d.iterdir() if p.is_dir()]) if d.exists() else []

@st.cache_data(ttl=60)
def load_scores(t):
    p = PROJECT_ROOT / "results" / "pipeline" / t / "scores" / f"{t}_scores.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=60)
def load_raw(t):
    p = PROJECT_ROOT / "results" / "pipeline" / t / "raw" / f"{t}.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data(ttl=60)
def load_xai_cache(t):
    d = PROJECT_ROOT / "data" / "xai_cache" / t
    if not d.exists(): return {}
    return {f.stem: dict(np.load(f, allow_pickle=True)) for f in sorted(d.glob("*.npz"))}

@st.cache_data(ttl=60)
def load_news_cache(t):
    d = PROJECT_ROOT / "data" / "news_cache" / t
    if not d.exists(): return {}
    out = {}
    for f in sorted(d.glob("*.csv")):
        try: out[f.stem] = pd.read_csv(f)
        except: pass
    return out

def load_attribution_report(t):
    p = PROJECT_ROOT / "results" / "pipeline" / t / "attribution" / f"{t}_attribution_report.md"
    return p.read_text(encoding='utf-8') if p.exists() else None

@st.cache_data(ttl=60)
def load_attribution_data(t):
    """Load structured attribution JSON (top headlines per event from LLM)."""
    p = PROJECT_ROOT / "results" / "pipeline" / t / "attribution" / f"{t}_attribution_data.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}

# ── Chart Builders ────────────────────────────────────────────────────────────

def build_main_chart(scores_df, raw_df, ticker, models):
    """Price + score panels with anomaly markers."""
    df = scores_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    n = 1 + len(models)
    heights = [0.45] + [0.55 / len(models)] * len(models)
    titles = [f'{ticker} Close Price'] + [f'{m["name"]}' for m in models]

    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=heights,
                        subplot_titles=titles)

    if raw_df is not None:
        raw = raw_df.copy()
        raw['Date'] = pd.to_datetime(raw['Date'])
        tail = raw.iloc[-len(df):].reset_index(drop=True)
        tail['Date'] = df['Date'].values
        fig.add_trace(go.Scatter(
            x=tail['Date'], y=tail['Close'], mode='lines', name='Close',
            line=dict(color=C['price'], width=1.5),
            fill='tozeroy', fillcolor='rgba(32,33,36,0.04)',
        ), row=1, col=1)

    for i, m in enumerate(models):
        sc = m.get('score_col', f'anomaly_score_{m["label"]}')
        pc = m.get('pred_col', f'prediction_{m["label"]}')
        clr = C.get(m['label'], '#999')
        row = i + 2
        if sc not in df.columns: continue
        v = df.dropna(subset=[sc])
        fig.add_trace(go.Scatter(
            x=v['Date'], y=v[sc], mode='lines', name=f'{m["label"]} Score',
            line=dict(color=clr, width=1),
            fill='tozeroy',
            fillcolor=f'rgba({int(clr[1:3],16)},{int(clr[3:5],16)},{int(clr[5:7],16)},0.06)',
        ), row=row, col=1)
        if pc in df.columns:
            a = v[v[pc] == 1]
            if len(a) > 0:
                fig.add_trace(go.Scatter(
                    x=a['Date'], y=a[sc], mode='markers',
                    name=f'{m["label"]} Anomaly',
                    marker=dict(color=C['anomaly'], size=4, opacity=0.6),
                    showlegend=False,
                ), row=row, col=1)

    fig.update_layout(
        height=480, template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=55, r=20, t=35, b=35),
        showlegend=False,
        font=dict(family='Inter', size=13, color='#1a1a1a'),
        hovermode='x unified',
    )
    for r in range(1, n + 1):
        fig.update_xaxes(gridcolor='#E8EAED', zeroline=False,
                         tickfont=dict(size=12, color='#1a1a1a'), row=r, col=1)
        fig.update_yaxes(gridcolor='#E8EAED', zeroline=False,
                         tickfont=dict(size=12, color='#1a1a1a'), row=r, col=1)
    return fig


def build_ig_chart(xai_data, feature_names, ticker):
    """IG feature importance bar chart."""
    imp = {}
    for k, d in xai_data.items():
        if '_ig' not in k: continue
        lab = k.split('_')[0]
        fi = d.get('feature_importance')
        if fi is not None: imp.setdefault(lab, []).append(fi)
    if not imp: return None

    agg = {}
    for lab, arrs in imp.items():
        a = np.mean(np.stack(arrs), axis=0)
        s = a.sum()
        agg[lab] = (a / s * 100) if s > 0 else a

    labs = list(agg.keys())
    pooled = sum(agg[l] for l in labs) / len(labs)
    idx = np.argsort(pooled)
    names = [feature_names[i] for i in idx]

    fig = go.Figure()
    for lab in labs:
        fig.add_trace(go.Bar(
            y=names, x=agg[lab][idx], name=lab, orientation='h',
            marker_color=C.get(lab, '#999'), opacity=0.85,
        ))

    fig.update_layout(
        barmode='group', height=250,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=140, r=20, t=10, b=40),
        xaxis_title='Feature Importance (%)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=12, color='#1a1a1a')),
        font=dict(family='Inter', size=13, color='#1a1a1a'),
    )
    fig.update_xaxes(gridcolor='#E8EAED', tickfont=dict(size=12, color='#1a1a1a'),
                     title_font=dict(size=13, color='#1a1a1a'))
    fig.update_yaxes(gridcolor='#E8EAED', tickfont=dict(size=12, color='#1a1a1a'))
    return fig


def build_ts_heatmap(xai_data, ticker):
    """TimeSHAP heatmap."""
    curves = {}
    for k, d in xai_data.items():
        if '_timeshap' not in k: continue
        lab = k.split('_')[0]
        sv = d.get('shapley_values')
        if sv is not None: curves.setdefault(lab, []).append(sv)
    if not curves: return None

    labs = list(curves.keys())
    np_ = len(labs)
    cms = {'AT': 'Blues', 'TranAD': 'Greens'}

    fig = make_subplots(rows=1, cols=np_, subplot_titles=labs, horizontal_spacing=0.1)
    for i, lab in enumerate(labs):
        cs = curves[lab]
        ml = min(len(c) for c in cs)
        mat = np.stack([np.abs(c[:ml]) for c in cs])
        ne = mat.shape[0]
        fig.add_trace(go.Heatmap(
            z=mat, x=list(range(ml)), y=[f'E{j+1}' for j in range(ne)],
            colorscale=cms.get(lab, 'Blues'), showscale=(i == np_ - 1),
            colorbar=dict(title='|φ(t)|', len=0.8),
            hovertemplate='Event: %{y}<br>Timestep: %{x}<br>|φ|: %{z:.4f}<extra></extra>',
        ), row=1, col=i + 1)

    fig.update_layout(
        height=250, template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=55, r=65, t=35, b=45),
        font=dict(family='Inter', size=13, color='#1a1a1a'),
    )
    for i in range(1, np_ + 1):
        fig.update_xaxes(title_text='Timestep', gridcolor='#E8EAED',
                         tickfont=dict(size=12, color='#1a1a1a'),
                         title_font=dict(size=13, color='#1a1a1a'), row=1, col=i)
    fig.update_yaxes(title_text='Event', gridcolor='#E8EAED',
                     tickfont=dict(size=12, color='#1a1a1a'),
                     title_font=dict(size=13, color='#1a1a1a'), row=1, col=1)
    return fig


def get_top_events_per_model(scores_df, news_data, attr_data, models, top_n=3):
    """Get top-N anomalous events for EACH model with their top headline.

    Prefers the LLM's #1 ranked headline (from attr_data) over the raw
    GDELT first row (from news_data).

    Returns: dict[model_label] -> list of event dicts
    """
    result = {}
    for m in models:
        lab = m['label']
        sc = m.get('score_col', f'anomaly_score_{lab}')
        if sc not in scores_df.columns: continue
        top = scores_df.dropna(subset=[sc]).nlargest(top_n, sc)
        events = []
        for _, row in top.iterrows():
            date = str(row['Date'])[:10]
            score = float(row[sc])
            headline = ''

            # Priority 1: LLM-ranked top headline
            attr_key = f"{lab}_{date}"
            if attr_key in attr_data:
                headline = str(attr_data[attr_key].get('top_headline', ''))[:160]

            # Fallback: first headline from raw GDELT cache
            if not headline:
                news_df = news_data.get(date)
                if isinstance(news_df, pd.DataFrame) and len(news_df) > 0:
                    if 'title' in news_df.columns:
                        headline = str(news_df.iloc[0]['title'])[:140]

            events.append({'date': date, 'score': score, 'headline': headline})
        result[lab] = events
    return result


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    tickers = discover_tickers()

    if not tickers:
        st.warning("No pipeline results. Run `python pipeline.py TICKER` first.")
        return

    models = [
        {'label': 'AT', 'name': 'Anomaly Transformer',
         'score_col': 'anomaly_score_AT', 'pred_col': 'prediction_AT'},
        {'label': 'TranAD', 'name': 'TimesNet + TranAD',
         'score_col': 'anomaly_score_TranAD', 'pred_col': 'prediction_TranAD'},
    ]

    # ── Title row + ticker selector ─────────────────────────────────────
    title_col, ticker_col = st.columns([4, 1])
    with title_col:
        st.markdown('<p class="page-title">📈 Anomaly Detection Dashboard</p>',
                    unsafe_allow_html=True)
    with ticker_col:
        ticker = st.selectbox("Ticker", tickers, index=0)

    # ── Load data ─────────────────────────────────────────────────────────
    scores_df = load_scores(ticker)
    raw_df = load_raw(ticker)
    xai_data = load_xai_cache(ticker)
    news_data = load_news_cache(ticker)
    attr_report = load_attribution_report(ticker)
    attr_data = load_attribution_data(ticker)

    if scores_df is None:
        st.info(f"No data for {ticker}.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────
    n_rows = len(scores_df)
    anom = {}
    for m in models:
        pc = m.get('pred_col', f'prediction_{m["label"]}')
        if pc in scores_df.columns:
            anom[m['label']] = int(scores_df[pc].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ticker", ticker)
    k2.metric("Trading Days", f"{n_rows:,}")
    k3.metric("AT Anomalies", anom.get('AT', 0))
    k4.metric("TranAD Anomalies", anom.get('TranAD', 0))
    k5.metric("XAI Cached", len(xai_data))

    # ── Row 1: Chart (2/3) + Events (1/3) ─────────────────────────────────
    chart_col, events_col = st.columns([2, 1], gap="medium")

    with chart_col:
        st.markdown('<div class="section-title">📉 Anomaly Scores & Price</div>',
                    unsafe_allow_html=True)
        try:
            fig = build_main_chart(scores_df, raw_df, ticker, models)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    with events_col:
        st.markdown('<div class="section-title">🔥 Top Anomalous Events</div>',
                    unsafe_allow_html=True)
        events_by_model = get_top_events_per_model(scores_df, news_data, attr_data, models, top_n=3)

        for m in models:
            lab = m['label']
            clr = C.get(lab, '#999')
            evts = events_by_model.get(lab, [])
            st.markdown(f'<div class="model-header" style="color:{clr}">▎ {m["name"]}</div>',
                        unsafe_allow_html=True)
            for ev in evts:
                hl = (f'<div class="event-headline">{ev["headline"]}</div>'
                      if ev['headline'] else '')
                st.markdown(f"""
                <div class="event-row">
                    <span class="event-date">{ev['date']}</span>
                    <span class="event-score" style="float:right">Score: {ev['score']:.4f}</span>
                    {hl}
                </div>
                """, unsafe_allow_html=True)
            if not evts:
                st.caption("No anomalies detected.")

    # ── Row 2: IG (1/2) + TimeSHAP (1/2) ─────────────────────────────────
    if xai_data:
        ig_col, ts_col = st.columns(2, gap="medium")

        feat_data = [d for d in xai_data.values() if 'feature_names' in d]
        fnames = list(feat_data[0]['feature_names']) if feat_data else None

        with ig_col:
            st.markdown('<div class="section-title">🔬 Integrated Gradients</div>',
                        unsafe_allow_html=True)
            if fnames:
                ig_fig = build_ig_chart(xai_data, fnames, ticker)
                if ig_fig:
                    st.plotly_chart(ig_fig, use_container_width=True)
                else:
                    st.info("No IG data in cache.")
            else:
                st.info("No feature names in cache.")

        with ts_col:
            st.markdown('<div class="section-title">⏱️ TimeSHAP</div>',
                        unsafe_allow_html=True)
            ts_fig = build_ts_heatmap(xai_data, ticker)
            if ts_fig:
                st.plotly_chart(ts_fig, use_container_width=True)
            else:
                st.info("No TimeSHAP data in cache.")

    # ── Detail Reports (full-width collapsible expanders) ──────────────────
    st.markdown("---")

    if attr_report:
        with st.expander("🤖 LLM Attribution Report", expanded=False):
            st.markdown(attr_report)
    else:
        with st.expander("🤖 LLM Attribution Report", expanded=False):
            st.caption("No attribution report. Run pipeline with GEMINI_API_KEY set.")

    with st.expander("📰 News Headlines Cache", expanded=False):
        if news_data:
            for date_str in sorted(news_data.keys(), reverse=True):
                ndf = news_data[date_str]
                if isinstance(ndf, pd.DataFrame) and len(ndf) > 0:
                    n_articles = len(ndf)
                    st.markdown(f"**{date_str}** — {n_articles} articles")
                    display_cols = [c for c in ['date', 'title', 'domain'] if c in ndf.columns]
                    st.dataframe(ndf[display_cols].head(50),
                                 use_container_width=True, height=180)
                else:
                    st.caption(f"{date_str} — no headlines")
        else:
            st.caption("No cached news headlines.")

    with st.expander("📋 Per-Event XAI Plots", expanded=False):
        xai_dir = PROJECT_ROOT / "results" / "pipeline" / ticker / "xai"
        if xai_dir.exists():
            plots = sorted(xai_dir.glob("*.png"))
            if plots:
                # Show in pairs for better layout
                for i in range(0, len(plots), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(plots):
                            col.image(str(plots[i + j]), caption=plots[i + j].stem,
                                      use_container_width=True)
            else:
                st.caption("No per-event XAI plots found.")
        else:
            st.caption("No XAI output directory.")


if __name__ == "__main__":
    main()