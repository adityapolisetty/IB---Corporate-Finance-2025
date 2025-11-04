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
st.set_page_config(
    page_title="Corporate Finance Tools",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = 'Home'
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}
st.markdown(
    """
    <style>
      /* Force light theme */
      :root {
        color-scheme: light !important;
      }
      .stApp {
        background: #ffffff !important;
        color: #000000 !important;
      }
      .stApp, .block-container { background: #ffffff !important; }
      .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 100%; }
      .stSelectbox label, .stNumberInput label { font-size: 0.92rem; margin-bottom: .18rem; }
      .stSelectbox > div[data-baseweb="select"] { min-height: 36px; }
      .stNumberInput > div > div > input { padding: 0.35rem 0.5rem !important; font-size: 0.9rem !important; }
      .stCheckbox { margin-top: 0.5rem !important; }
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
      /* Make form more compact */
      [data-testid="stForm"] { padding: 1rem !important; }
      .stButton button { padding: 0.4rem 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Sidebar Menu
# ==============================
st.sidebar.title("📊 Corporate Finance Tools")
st.sidebar.markdown("---")

menu_options = {
    "🏠 Home": "Home",
    "📈 Project Decisions": "Project Decisions"
}

selected = st.sidebar.radio(
    "Select a Tool:",
    list(menu_options.keys()),
    index=0 if st.session_state.current_menu == 'Home' else 1
)

# Update current menu based on selection
st.session_state.current_menu = menu_options[selected]

# Reset to input page when switching menus
if st.session_state.current_menu == 'Project Decisions' and st.session_state.page == 'visualization':
    # Keep the current state if already in Project Decisions
    pass
else:
    if st.session_state.current_menu != 'Project Decisions':
        st.session_state.page = 'input'

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This toolkit provides interactive demonstrations of corporate finance concepts including:
- IRR analysis and pitfalls
- NPV calculations
- Investment vs Financing patterns
""")

# ==============================
# Color scheme and fonts
# ==============================
BLUE = "#141450"
GREY = "#808080"
RED = "#FF0000"
GREEN = "#006400"
ALPHA = 0.7

_DEF_FONT = dict(family="Arial, sans-serif", size=13, color="#333333")

# Helper functions for pattern detection
def detect_pattern_type(cashflows):
    """
    Detect the pattern type based on cashflow signs.
    Returns: 'investment', 'financing', or 'mixed'
    """
    if len(cashflows) < 2:
        return 'mixed'

    # Count sign changes
    sign_changes = 0
    for i in range(1, len(cashflows)):
        if (cashflows[i-1] >= 0 and cashflows[i] < 0) or (cashflows[i-1] < 0 and cashflows[i] >= 0):
            sign_changes += 1

    # Investment pattern: negative first, then positive (one sign change at start)
    if cashflows[0] < 0 and all(cf >= 0 for cf in cashflows[1:]):
        return 'investment'

    # Financing pattern: positive first, then negative (one sign change at end)
    if cashflows[0] > 0 and all(cf >= 0 for cf in cashflows[:-1]) and cashflows[-1] < 0:
        return 'financing'

    # Multiple sign changes indicate potential multiple IRRs
    if sign_changes > 1:
        return 'mixed'

    return 'mixed'

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

def make_cashflow_df(cashflows, start_year=0, perpetuity_year=None):
    """Create a Year/Cash Flow table with formatted values."""
    year_labels = []
    for i in range(len(cashflows)):
        year_num = start_year + i
        label = str(year_num)
        if perpetuity_year is not None and year_num == perpetuity_year:
            label += " (∞)"
        year_labels.append(label)

    return pd.DataFrame({
        "Year": year_labels,
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
            tickfont=dict(size=13),
            showgrid=False
        ),
        yaxis=dict(
            title="Discount Factor",
            title_font=dict(size=15, color=DARK_BLUE),
            tickfont=dict(color=DARK_BLUE, size=12),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            range=[0, max_factor * 1.3]
        ),
        height=350,
        title=dict(
            text=f"<b>Discount Factors (at {rate*100:.1f}%)</b>",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333")
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
            tickfont=dict(size=13),
            showgrid=False
        ),
        yaxis=dict(
            title="Discounted Cashflows (£)",
            title_font=dict(size=15, color=DARK_RED),
            tickfont=dict(color=DARK_RED, size=12),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            range=[min(min_cf * 1.2, -max_cf * 0.2), max_cf * 1.2]
        ),
        height=350,
        title=dict(
            text=f"<b>Discounted Cashflows</b><br><span style='font-size: 20px; font-weight: bold;'>NPV = £{npv_value:,.0f}</span>",
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333")
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
            tickfont=_DEF_FONT,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)',
            title_standoff=14
        ),
        yaxis=dict(
            title="NPV ($)",
            autorange=True,
            title_font=dict(size=14, color="#333333"),
            tickfont=_DEF_FONT,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.15)'
        ),
        height=350,
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            font=dict(family="Arial, sans-serif", size=15, color="#333333")
        ),
        hovermode='closest'
    )

    return fig

# ==============================
# PAGE LOGIC
# ==============================

if st.session_state.current_menu == 'Home':
    # ==============================
    # LANDING PAGE
    # ==============================
    st.title("Welcome to Corporate Finance Interactive Tools")

    st.markdown("""
    ## 🎯 Purpose

    This platform provides interactive tools to help students understand key corporate finance concepts
    through hands-on exploration and visualization.

    ---

    ## 🛠️ Available Tools

    ### 📈 Project Decisions (IRR & NPV Analysis)

    Explore the complexities of project evaluation using:
    - **Internal Rate of Return (IRR)** calculations
    - **Net Present Value (NPV)** analysis
    - **Pattern Detection**: Automatically identifies investment vs financing patterns
    - **Perpetuity Models**: Support for projects with infinite cash flows
    - **Growth Rate Analysis**: Model cash flows with constant growth rates

    **Key Learning Outcomes:**
    - Understand when IRR can be misleading
    - Learn why the decision rule flips for financing-type projects
    - Visualize how discount rates affect NPV
    - Recognize patterns that lead to multiple IRRs

    ---

    ## 🚀 Getting Started

    1. **Select a tool** from the sidebar menu on the left
    2. **Enter your project parameters** in the input form
    3. **Analyze the results** with interactive visualizations
    4. **Experiment** with different scenarios to build intuition

    ---

    ## 📚 How to Use

    ### For Students:
    - Use these tools to complement your coursework
    - Test theoretical concepts with real numbers
    - Explore edge cases and special scenarios
    - Build intuition through visualization

    ### For Instructors:
    - Demonstrate concepts in real-time during lectures
    - Assign exploration exercises to students
    - Use as a starting point for class discussions
    - Generate examples for problem sets

    ---

    """)

    # Quick start cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 💡 Quick Tips
        - All tools provide real-time calculations
        - Visualizations update as you adjust parameters
        - Use the sidebar to navigate between tools
        - Click "Analyze Project" to see detailed results
        """)

    with col2:
        st.markdown("""
        ### 📖 Resources
        - Check your course materials for theoretical background
        - Experiment with different cash flow patterns
        - Pay attention to warning messages
        - Use perpetuity for long-term projects
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Select <strong>Project Decisions</strong> from the sidebar to begin analyzing projects →</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_menu == 'Project Decisions':

    if st.session_state.page == 'input':
        # ==============================
        # INPUT FORM PAGE - PROGRESSIVE
        # ==============================
        st.title("IRR Analysis - Project Setup")

        # Center the form with narrower width
        _, form_col, _ = st.columns([1, 2, 1])

        with form_col:
            st.markdown("### Enter Project Details")

            # Step 1: Basic inputs (always shown)
            col1, col2 = st.columns(2)

            with col1:
                start_year = st.number_input(
                    "Start Year",
                    min_value=0,
                    value=0,
                    step=1,
                    help="The starting year for the project (typically 0)",
                    key="start_year_input"
                )

            with col2:
                pass  # Empty for now

            col3, col4 = st.columns(2)
            with col3:
                min_hurdle = st.number_input(
                    "Min Hurdle Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    format="%.1f",
                    key="min_hurdle_input"
                )

            with col4:
                max_hurdle = st.number_input(
                    "Max Hurdle Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=30.0,
                    step=0.5,
                    format="%.1f",
                    key="max_hurdle_input"
                )

            # Step 2: Number of years (shown after hurdle rates)
            st.markdown("---")

            col5, col6 = st.columns(2)
            with col5:
                num_years = st.number_input(
                    "Number of Years",
                    min_value=2,
                    max_value=50,
                    value=4,
                    step=1,
                    help="Total number of years including year 0",
                    key="num_years_input"
                )

            # Perpetuity option
            with col6:
                is_perpetuity = st.checkbox(
                    "Use Perpetuity",
                    value=False,
                    help="Check if the project has perpetual cash flows",
                    key="perpetuity_checkbox"
                )

            # Show perpetuity suggestion if >10 years
            if num_years > 10 and not is_perpetuity:
                st.warning(f"⚠️ You've entered {int(num_years)} years. Consider using a perpetuity model for long-term projects.")

            # If perpetuity selected, ask for starting year
            perpetuity_start_year = None
            if is_perpetuity:
                col_perp1, col_perp2 = st.columns(2)
                with col_perp1:
                    perpetuity_start_year = st.number_input(
                        "Perpetuity starts in Year",
                        min_value=int(start_year) + 1,
                        max_value=int(start_year) + int(num_years) - 1,
                        value=int(start_year) + 1,
                        step=1,
                        key="perpetuity_start_year_input",
                        help="The year when perpetual cash flows begin"
                    )
                st.info(f"ℹ️ Cash flows will be entered up to Year {perpetuity_start_year}, which continues perpetually.")

            # Step 3: Growth rate (optional, shown after number of years)
            if num_years >= 2:
                st.markdown("---")
                st.markdown("**Growth Rate (Optional)**")

                col7, col8 = st.columns(2)
                with col7:
                    use_growth = st.checkbox(
                        "Apply growth rate to cash flows",
                        value=False,
                        key="use_growth_checkbox"
                    )

                if use_growth:
                    with col8:
                        growth_start_year = st.number_input(
                            "Starting from Year",
                            min_value=int(start_year) + 1,
                            max_value=int(start_year) + int(num_years) - 1,
                            value=int(start_year) + 1,
                            step=1,
                            key="growth_start_year_input"
                        )

                    col9, _ = st.columns(2)
                    with col9:
                        growth_rate = st.number_input(
                            "Growth Rate (%)",
                            min_value=-100.0,
                            max_value=100.0,
                            value=0.0,
                            step=0.5,
                            format="%.1f",
                            key="growth_rate_input",
                            help="Constant growth rate applied to subsequent cash flows"
                        )
                else:
                    growth_rate = 0.0
                    growth_start_year = int(start_year) + 1

            # Step 4: Cash flow inputs (shown after number of years)
            if num_years >= 2:
                st.markdown("---")
                st.markdown("### Cash Flows")

                # Determine how many cash flow fields to show
                if is_perpetuity and perpetuity_start_year is not None:
                    # Only show fields up to perpetuity year
                    num_cf_fields = perpetuity_start_year - int(start_year) + 1
                    st.markdown(f"Enter cash flows up to Year {perpetuity_start_year} (perpetuity year):")
                else:
                    num_cf_fields = int(num_years)
                    st.markdown("Enter the cash flow for each year:")

                # Initialize session state for cash flows if not exists
                if 'cashflows_input' not in st.session_state:
                    st.session_state.cashflows_input = {}

                # Cash flow inputs - more compact with 4 columns
                num_cols = min(num_cf_fields, 4)
                cashflows = []

                for i in range(num_cf_fields):
                    if i % num_cols == 0:
                        cashflow_cols = st.columns(num_cols)

                    col_idx = i % num_cols
                    with cashflow_cols[col_idx]:
                        cf_key = f"cf_{i}"
                        year_label = f"Year {int(start_year) + i}"

                        # Add indicator for perpetuity year
                        if is_perpetuity and perpetuity_start_year is not None and (int(start_year) + i) == perpetuity_start_year:
                            year_label += " (Perpetuity)"

                        cf_value = st.number_input(
                            year_label,
                            value=st.session_state.cashflows_input.get(cf_key, 0.0),
                            step=100.0,
                            format="%.0f",
                            key=cf_key
                        )
                        st.session_state.cashflows_input[cf_key] = cf_value
                        cashflows.append(cf_value)

                # Apply growth rate if specified
                if use_growth and growth_rate != 0.0:
                    st.info(f"Growth rate of {growth_rate:.1f}% will be applied from Year {growth_start_year} onwards")

                # Submit button
                st.markdown("---")
                if st.button("Analyze Project", use_container_width=True, type="primary"):
                    # Apply growth rate to cashflows if specified
                    if use_growth and growth_rate != 0.0:
                        base_year_idx = growth_start_year - int(start_year)
                        if base_year_idx < len(cashflows):
                            base_cf = cashflows[base_year_idx]
                            for i in range(base_year_idx + 1, len(cashflows)):
                                cashflows[i] = base_cf * ((1 + growth_rate/100) ** (i - base_year_idx))

                    # Validate inputs
                    if max_hurdle <= min_hurdle:
                        st.error("Maximum hurdle rate must be greater than minimum hurdle rate.")
                    elif all(cf == 0 for cf in cashflows):
                        st.error("Please enter at least one non-zero cash flow.")
                    else:
                        # Store data in session state
                        st.session_state.project_data = {
                            'start_year': int(start_year),
                            'num_years': int(num_years),
                            'is_perpetuity': is_perpetuity,
                            'perpetuity_start_year': perpetuity_start_year,
                            'min_hurdle': min_hurdle,
                            'max_hurdle': max_hurdle,
                            'growth_rate': growth_rate if use_growth else 0.0,
                            'cashflows': cashflows
                        }
                        st.session_state.page = 'visualization'
                        st.rerun()

    elif st.session_state.page == 'visualization':
        # ==============================
        # VISUALIZATION PAGE
        # ==============================

        # Get data from session state
        data = st.session_state.project_data
        cashflows = data['cashflows']
        min_hurdle = data['min_hurdle']
        max_hurdle = data['max_hurdle']

        # Detect pattern type
        pattern_type = detect_pattern_type(cashflows)

        # Calculate IRR
        irr_val = find_irr(cashflows)
        if irr_val is not None and irr_val < 0:
            irr_val = None

        # Header with back button
        col_back, col_title = st.columns([1, 5])
        with col_back:
            if st.button("← New Project"):
                st.session_state.page = 'input'
                st.rerun()

        with col_title:
            if pattern_type == 'investment':
                st.title("Investment Pattern Analysis")
            elif pattern_type == 'financing':
                st.title("Financing Pattern Analysis")
            else:
                st.title("Mixed Pattern Analysis (Multiple IRRs Possible)")

        # Show perpetuity info if enabled
        if data.get('is_perpetuity') and data.get('perpetuity_start_year') is not None:
            st.info(f"ℹ️ Perpetuity model: Cash flow from Year {data['perpetuity_start_year']} continues indefinitely")

        # Hurdle rate slider
        default_hurdle = (min_hurdle + max_hurdle) / 2
        hurdle_pct = st.slider(
            "Hurdle Rate (%)",
            min_hurdle,
            max_hurdle,
            default_hurdle,
            0.5,
            format="%0.1f%%"
        )
        hurdle_rate = hurdle_pct / 100

        st.markdown("---")

        # Three-column layout
        col1, col2, col3 = st.columns([0.45, 1.35, 1.2])

        with col1:
            st.markdown("### Cash Flows")
            perpetuity_year = data.get('perpetuity_start_year') if data.get('is_perpetuity') else None
            st.dataframe(
                make_cashflow_df(cashflows, start_year=data['start_year'], perpetuity_year=perpetuity_year),
                hide_index=True,
                use_container_width=True
            )

        with col2:
            st.markdown("### Discount Analysis")
            # Discount factors and discounted cashflows charts
            fig_df1, fig_df2 = plot_discount_factors_split(
                cashflows,
                hurdle_rate,
                show_current=True,
                pattern_type=pattern_type
            )
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(fig_df1, use_container_width=True, key=f"df1_{hurdle_pct}")
            with chart_col2:
                st.plotly_chart(fig_df2, use_container_width=True, key=f"df2_{hurdle_pct}")

        with col3:
            st.markdown("### NPV Profile")
            # NPV curve chart
            fig_npv = plot_npv_curve(
                cashflows,
                "NPV vs Hurdle Rate",
                irr_val,
                hurdle_rate,
                show_current=True,
                pattern_type=pattern_type
            )
            st.plotly_chart(fig_npv, use_container_width=True)

        # Decision logic
        npv_value = npv(hurdle_rate, cashflows)
        decision_html = "<p style='text-align: center; font-size: 1.1rem; font-weight: 500; margin-top: 0.2rem;'>"

        if pattern_type == 'investment':
            if irr_val is not None and hurdle_rate < irr_val:
                decision_html += (
                    f"<span style='color: {GREEN};'>✓ ACCEPT | Hurdle Rate ({hurdle_pct:.1f}%) "
                    f"< IRR ({irr_val*100:.2f}%) | NPV = £{npv_value:.0f}</span>"
                )
            elif irr_val is not None:
                decision_html += (
                    f"<span style='color: {RED};'>✗ REJECT | Hurdle Rate ({hurdle_pct:.1f}%) "
                    f"> IRR ({irr_val*100:.2f}%) | NPV = £{npv_value:.0f}</span>"
                )
            else:
                decision_html += f"<span style='color: {GREY};'>No valid IRR | NPV = £{npv_value:.0f}</span>"

        elif pattern_type == 'financing':
            if irr_val is not None and hurdle_rate > irr_val:
                decision_html += (
                    f"<span style='color: {GREEN};'>✓ ACCEPT | Hurdle Rate ({hurdle_pct:.1f}%) "
                    f"> IRR ({irr_val*100:.2f}%) | NPV = £{npv_value:.0f}</span>"
                )
            elif irr_val is not None:
                decision_html += (
                    f"<span style='color: {RED};'>✗ REJECT | Hurdle Rate ({hurdle_pct:.1f}%) "
                    f"< IRR ({irr_val*100:.2f}%) | NPV = £{npv_value:.0f}</span>"
                )
            else:
                decision_html += f"<span style='color: {GREY};'>No valid IRR | NPV = £{npv_value:.0f}</span>"

        else:  # mixed pattern
            if irr_val is not None:
                decision_html += (
                    f"<span style='color: {GREY};'>⚠ MIXED PATTERN | IRR = {irr_val*100:.2f}% "
                    f"| Use NPV for decision: NPV = £{npv_value:.0f}</span>"
                )
            else:
                decision_html += f"<span style='color: {GREY};'>⚠ MIXED PATTERN | No valid IRR | NPV = £{npv_value:.0f}</span>"

        decision_html += "</p>"
        st.markdown(decision_html, unsafe_allow_html=True)
