"""
IRR Pitfalls: Investment vs Financing Patterns
Demonstrates how the IRR decision rule flips for financing-type cash flows
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ==============================
# App config: wide + light CSS
# ==============================
st.set_page_config(page_title="IRR Pitfalls Demo", layout="wide")
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .stTabs [data-baseweb="tab-list"] { gap: .25rem; }
      .stTabs [data-baseweb="tab"] { padding: .35rem .7rem; }
      .js-plotly-plot .plotly .main-svg { overflow: visible !important; }
      [data-testid="stDataFrame"] {
        margin-bottom: 0.25rem !important;
        font-size: 0.85rem !important;
      }
      [data-testid="stDataFrame"] table { margin: 0 !important; }
      [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
        padding: 0.2rem 0.4rem !important;
        text-align: center !important;
      }
      h1 { margin-bottom: 0.5rem !important; font-size: 1.8rem !important; }
      h3 { margin-top: 0rem; margin-bottom: 0.25rem; font-size: 1.1rem !important; }
      hr { margin: 0.25rem 0 !important; }
      .element-container { margin-bottom: 0rem !important; }
      [data-testid="stMarkdownContainer"] { padding-top: 0.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Color scheme and fonts
# ==============================
BLUE = "#141450"
GREY = "#808080"
RED = "#FF0000"
GREEN = "#006400"
ALPHA = 0.7

_DEF_FONT = dict(family="Arial, sans-serif", size=13, color="#333333")

# Title
# st.title("")

# Removed global animate/speed controls; per-section sliders added below

# Default hurdle rate (as percent) for initial slider position
DEFAULT_HURDLE_PCT = 0.0

# Helper functions
def npv(rate, cashflows):
    """Calculate NPV for a given hurdle rate and cash flows."""
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

def find_irr(cashflows, tol=1e-6):
    """Find IRR using Newton-Raphson method"""
    # Try multiple starting points
    for r_guess in [0.05, 0.10, 0.15, 0.20]:
        r = r_guess
        for _ in range(1000):
            npv_val = npv(r, cashflows)
            if abs(npv_val) < tol:
                return r
            # Derivative of NPV
            dnpv = sum(-t * cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cashflows))
            if abs(dnpv) < 1e-10:
                break
            r = r - npv_val / dnpv
            if r < -0.99:  # Keep r reasonable
                break
    return None

def _fmt_currency(value: float) -> str:
    """Return a currency string like £1,000 or £-1,900."""
    sign = '-' if value < 0 else ''
    return f"£{sign}{abs(value):,}"

def make_cashflow_df(cashflows):
    """Create a Year/Cash Flow table with formatted values."""
    return pd.DataFrame({
        "Year": [str(i) for i in range(len(cashflows))],
        "Cash Flow": [_fmt_currency(v) for v in cashflows]
    })

def plot_discount_factors_split(cashflows, current_r=None, show_current=False, pattern_type="investment"):
    """Plot discount factors and discounted cashflows as two separate charts"""
    # Use current rate if provided, otherwise use 0
    rate = current_r if current_r is not None else 0.0

    # Dark blue and dark red colors
    DARK_BLUE = "#003366"
    DARK_RED = "#8B0000"

    # Calculate discount factor and discounted cashflow for each year at the current rate
    years = list(range(len(cashflows)))
    discount_factors = [1 / ((1 + rate) ** year) if year > 0 else 1.0 for year in years]
    discounted_cfs = [cashflows[year] * discount_factors[year] for year in years]

    # Calculate NPV as sum of discounted cash flows
    npv_value = sum(discounted_cfs)

    # Chart 1: Discount Factors
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=[f'Year {year}' for year in years],
        y=discount_factors,
        marker_color=DARK_BLUE,
        opacity=0.85,
        text=[f'{df:.3f}' for df in discount_factors],
        textposition='outside',
        textfont_size=13,
        textfont_color=DARK_BLUE,
        hovertemplate='<b>Year %{x}</b><br>Discount Factor: %{y:.4f}<extra></extra>',
        showlegend=False
    ))

    max_factor = max(discount_factors) if discount_factors else 1.0
    fig1.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=30),
        font=dict(family="Arial, sans-serif", size=14, color="#333333"),
        xaxis=dict(
            title="",
            tick_font=dict(size=13),
            showgrid=False
        ),
        yaxis=dict(
            title="Discount Factor",
            title_font=dict(size=15, color=DARK_BLUE),
            tick_font=dict(color=DARK_BLUE, size=12),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            range=[0, max_factor * 1.3]
        ),
        height=350,
        title=dict(
            text=f"Discount Factors (at {rate*100:.1f}%)",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333", weight='bold')
        )
    )

    # Chart 2: Discounted Cashflows
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[f'Year {year}' for year in years],
        y=discounted_cfs,
        marker_color=DARK_RED,
        opacity=0.85,
        text=[f'£{cf:.0f}' for cf in discounted_cfs],
        textposition='outside',
        textfont_size=13,
        textfont_color=DARK_RED,
        hovertemplate='<b>Year %{x}</b><br>Discounted CF: £%{y:.0f}<extra></extra>',
        showlegend=False
    ))

    max_cf = max(abs(cf) for cf in discounted_cfs) if discounted_cfs else 1000
    min_cf = min(discounted_cfs) if discounted_cfs else 0

    fig2.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40),
        font=dict(family="Arial, sans-serif", size=14, color="#333333"),
        xaxis=dict(
            title="Year",
            title_font=dict(size=15, color="#333333"),
            tick_font=dict(size=13),
            showgrid=False
        ),
        yaxis=dict(
            title="Discounted Cashflows (£)",
            title_font=dict(size=15, color=DARK_RED),
            tick_font=dict(color=DARK_RED, size=12),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            range=[min(min_cf * 1.2, -max_cf * 0.2), max_cf * 1.2]
        ),
        height=350,
        title=dict(
            text=f"Discounted Cashflows<br><span style='font-size: 20px; font-weight: bold;'>NPV = £{npv_value:,.0f}</span>",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333", weight='bold')
        )
    )

    return fig1, fig2

def plot_npv_curve(cashflows, title, irr_val, current_r=None, show_current=False, pattern_type="investment"):
    """Plot NPV curve with IRR line using Plotly"""
    rates = np.linspace(0, 0.30, 200)
    npvs = [npv(r, cashflows) for r in rates]

    # Determine color based on pattern type
    line_color = BLUE if pattern_type == "investment" else RED

    fig = go.Figure()

    # NPV curve
    fig.add_trace(go.Scatter(
        x=rates * 100,
        y=npvs,
        mode='lines',
        name='NPV Curve',
        line=dict(color=line_color, width=3),
        hovertemplate='Hurdle Rate=%{x:.1f}%<br>NPV=£%{y:.0f}<extra></extra>'
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color=GREY, line_width=1.5, opacity=0.6)

    # IRR line
    if irr_val is not None:
        fig.add_vline(
            x=irr_val * 100,
            line_dash="dash",
            line_color=RED,
            line_width=2.5,
            opacity=0.8,
            annotation_text=f"IRR={irr_val*100:.2f}%",
            annotation_position="top",
            annotation_font_size=12
        )

    # Current hurdle rate point
    if show_current and current_r is not None:
        current_npv = npv(current_r, cashflows)
        decision_color = GREEN if (
            (pattern_type == "investment" and current_npv > 0) or
            (pattern_type == "financing" and current_npv > 0)
        ) else RED

        fig.add_trace(go.Scatter(
            x=[current_r * 100],
            y=[current_npv],
            mode='markers',
            name=f'Current Hurdle Rate {current_r*100:.1f}%',
            marker=dict(color=decision_color, size=15, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate=f'Hurdle Rate={current_r*100:.1f}%<br>NPV=£{current_npv:.0f}<extra></extra>'
        ))

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=_DEF_FONT),
        margin=dict(l=20, r=20, t=40, b=60),
        font=_DEF_FONT,
        xaxis=dict(
            title="Hurdle Rate (%)",
            autorange=True,
            title_font=dict(size=14, color="#333333"),
            tick_font=_DEF_FONT,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            title_standoff=14
        ),
        yaxis=dict(
            title="NPV ($)",
            autorange=True,
            title_font=dict(size=14, color="#333333"),
            tick_font=_DEF_FONT,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)'
        ),
        height=350,
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333", weight='bold')
        ),
        hovermode='closest'
    )

    return fig

# Define cash flows
# Investment pattern: Upfront cost, then returns (typical project investment)
# Using 420 per year to get IRR > 10%
investment_cfs = [-1000, 420, 420, 420]

# Financing pattern: Receive money first, pay back later (from Problem 2)
financing_cfs = [540, 540, 540, -1900]

# Calculate IRRs (filter out negative IRRs)
irr_investment = find_irr(investment_cfs)
irr_financing = find_irr(financing_cfs)

# Validate IRRs are positive
if irr_investment is not None and irr_investment < 0:
    irr_investment = None
if irr_financing is not None and irr_financing < 0:
    irr_financing = None

st.markdown("---")

# Vertical stacked sections with per-section hurdle sliders

# Investment section
st.markdown("### Investment Pattern")
slider_col, _ = st.columns([1, 2])
with slider_col:
    hurdle_invest_pct = st.slider(
        "Hurdle Rate",
        0.0,
        30.0,
        DEFAULT_HURDLE_PCT,
        0.5,
        key="hurdle_invest",
        format="%0.1f%%"
    )
hurdle_invest = hurdle_invest_pct / 100

inv_col1, inv_col2, inv_col3 = st.columns([0.45, 1.35, 1.2])

with inv_col1:
    st.dataframe(make_cashflow_df(investment_cfs), hide_index=True, use_container_width=True)

with inv_col2:
    # Discount factors and discounted cashflows charts (split) - side by side
    fig_df1_inv, fig_df2_inv = plot_discount_factors_split(
        investment_cfs,
        hurdle_invest,
        show_current=True,
        pattern_type="investment"
    )
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(fig_df1_inv, use_container_width=True, key=f"inv_df1_{hurdle_invest_pct}")
    with chart_col2:
        st.plotly_chart(fig_df2_inv, use_container_width=True, key=f"inv_df2_{hurdle_invest_pct}")

with inv_col3:
    # NPV curve chart
    fig1 = plot_npv_curve(
        investment_cfs,
        "NPV vs Hurdle Rate",
        irr_investment,
        hurdle_invest,
        show_current=True,
        pattern_type="investment"
    )
    st.plotly_chart(fig1, use_container_width=True)

npv_inv = npv(hurdle_invest, investment_cfs)
inv_decision = "<p style='text-align: center; font-size: 1.1rem; font-weight: 500; margin-top: 0.2rem;'>"
if irr_investment is not None and hurdle_invest < irr_investment:
    inv_decision += (
        f"<span style='color: {GREEN};'>✓ ACCEPT | Hurdle Rate ({hurdle_invest_pct:.1f}%) "
        f"< IRR ({irr_investment*100:.2f}%) | NPV = £{npv_inv:.0f}</span>"
    )
elif irr_investment is not None:
    inv_decision += (
        f"<span style='color: {RED};'>✗ REJECT | Hurdle Rate ({hurdle_invest_pct:.1f}%) "
        f"> IRR ({irr_investment*100:.2f}%) | NPV = £{npv_inv:.0f}</span>"
    )
else:
    inv_decision += f"No valid IRR | NPV = £{npv_inv:.0f}"
inv_decision += "</p>"
st.markdown(inv_decision, unsafe_allow_html=True)

st.markdown("---")

# Financing section
st.markdown("### Financing Pattern")
slider_col_fin, _ = st.columns([1, 2])
with slider_col_fin:
    hurdle_fin_pct = st.slider(
        "Hurdle Rate",
        0.0,
        30.0,
        DEFAULT_HURDLE_PCT,
        0.5,
        key="hurdle_fin",
        format="%0.1f%%"
    )
hurdle_fin = hurdle_fin_pct / 100

fin_col1, fin_col2, fin_col3 = st.columns([0.45, 1.35, 1.2])

with fin_col1:
    st.dataframe(make_cashflow_df(financing_cfs), hide_index=True, use_container_width=True)

with fin_col2:
    # Discount factors and discounted cashflows charts (split) - side by side
    fig_df1_fin, fig_df2_fin = plot_discount_factors_split(
        financing_cfs,
        hurdle_fin,
        show_current=True,
        pattern_type="financing"
    )
    chart_col1_fin, chart_col2_fin = st.columns(2)
    with chart_col1_fin:
        st.plotly_chart(fig_df1_fin, use_container_width=True, key=f"fin_df1_{hurdle_fin_pct}")
    with chart_col2_fin:
        st.plotly_chart(fig_df2_fin, use_container_width=True, key=f"fin_df2_{hurdle_fin_pct}")

with fin_col3:
    # NPV curve chart
    fig2 = plot_npv_curve(
        financing_cfs,
        "NPV vs Hurdle Rate",
        irr_financing,
        hurdle_fin,
        show_current=True,
        pattern_type="financing"
    )
    st.plotly_chart(fig2, use_container_width=True)

npv_fin = npv(hurdle_fin, financing_cfs)
fin_decision = "<p style='text-align: center; font-size: 1.1rem; font-weight: 500; margin-top: 0.2rem;'>"
if irr_financing is not None and hurdle_fin > irr_financing:
    fin_decision += (
        f"<span style='color: {GREEN};'>✓ ACCEPT | Hurdle Rate ({hurdle_fin_pct:.1f}%) "
        f"> IRR ({irr_financing*100:.2f}%) | NPV = £{npv_fin:.0f}</span>"
    )
elif irr_financing is not None:
    fin_decision += (
        f"<span style='color: {RED};'>✗ REJECT | Hurdle Rate ({hurdle_fin_pct:.1f}%) "
        f"< IRR ({irr_financing*100:.2f}%) | NPV = £{npv_fin:.0f}</span>"
    )
else:
    fin_decision += f"No valid IRR | NPV = £{npv_fin:.0f}"
fin_decision += "</p>"
st.markdown(fin_decision, unsafe_allow_html=True)
