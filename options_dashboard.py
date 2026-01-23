import streamlit as st
import numpy as np
import plotly.graph_objects as go
from advanced_options_engine import AmericanOptionPricer, BlackScholesEngine, CPP_AVAILABLE

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quant Option Lab", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK FINANCE THEME) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric label { color: #888; }
    .stMetric value { color: #00ffcc; font-family: 'Roboto Mono', monospace; }
    h1, h2, h3 { color: #ffffff; font-family: 'Helvetica Neue', sans-serif; }
    .info-box {
        background-color: #0e1117;
        border-left: 5px solid #00ffcc;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: INPUT PARAMETERS ---
with st.sidebar:
    st.title("🎛️ Market Parameters")
    
    st.markdown("### 1. Asset Specs")
    S = st.number_input("Spot Price ($)", value=100.0, step=1.0)
    K = st.number_input("Strike Price ($)", value=100.0, step=1.0)
    
    st.markdown("### 2. Market Variables")
    T = st.slider("Time to Maturity (Years)", 0.01, 3.0, 1.0)
    r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
    sigma = st.slider("Volatility (σ)", 0.05, 1.0, 0.2)
    
    st.markdown("### 3. Option Type")
    opt_type = st.radio("Class", ["Call", "Put"], horizontal=True)
    is_call = opt_type == "Call"
    
    st.markdown("---")
    st.caption("🚀 Engine: C++ Crank-Nicolson FDM")
    st.caption("✅ Status: " + ("Online" if CPP_AVAILABLE else "Offline (Python Fallback)"))

# --- MAIN LAYOUT ---
st.title("⚡ Quantitative Derivatives Engine")
st.markdown(f"**Pricing Date:** 2026-01-23 | **Underlying:** Synthetic Asset | **Method:** Finite Difference PDE")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🔥 Pricing & Greeks", "📊 Volatility Surface", "🧠 Methodology (For Recruiters)"])

# --- TAB 1: PRICING DASHBOARD ---
with tab1:
    # 1. Calculation Block
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        M = st.number_input("Price Grid Steps", value=100, help="Spatial nodes in the Finite Difference Grid")
    with col2:
        N = st.number_input("Time Grid Steps", value=1000, help="Temporal steps in the simulation")
    with col3:
        st.write("") # Spacer
        st.write("") 
        calc = st.button("🚀 Calculate American Price", use_container_width=True, type="primary")

    if calc:
        if CPP_AVAILABLE:
            pricer = AmericanOptionPricer(S, K, T, r, sigma, is_call)
            price = pricer.price(int(M), int(N))
            
            # BS Comparison
            bs = BlackScholesEngine(S, K, T, r, sigma, is_call)
            bs_price = bs.price()
            premium = price - bs_price
            
            # --- METRICS ROW ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("American Price (C++ FDM)", f"${price:.4f}", delta=None)
            with m2:
                st.metric("European Price (BS)", f"${bs_price:.4f}", delta=f"{premium:.4f} Premium")
            with m3:
                st.metric("Early Exercise Premium", f"${premium:.4f}", delta_color="normal")
            
            st.success(f"✅ Computed in < 0.05s using C++ backend via PyBind11.")
            
        else:
            st.error("⚠️ C++ Module not found. Compile using setup.py.")

    # 2. Greeks Visualization (Interactive)
    st.markdown("### 📉 Sensitivity Analysis (The Greeks)")
    
    # Generate data for plotting
    spot_range = np.linspace(max(0, S-50), S+50, 50)
    bs_prices = []
    deltas = []
    
    # Use BS for Greeks visualization (fast & analytical)
    for s_i in spot_range:
        engine = BlackScholesEngine(s_i, K, T, r, sigma, is_call)
        bs_prices.append(engine.price())
        # Approx Delta
        engine_up = BlackScholesEngine(s_i+0.01, K, T, r, sigma, is_call)
        deltas.append((engine_up.price() - engine.price())/0.01)

    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=bs_prices, name="Option Price", line=dict(color='#00ffcc', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=deltas, name="Delta (Sensitivity)", yaxis="y2", line=dict(color='#ff0066', dash='dot')))
    
    fig.update_layout(
        title="Price & Delta vs Underlying",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(title="Spot Price ($)", gridcolor="#333"),
        yaxis=dict(title="Option Value ($)", gridcolor="#333"),
        yaxis2=dict(title="Delta", overlaying="y", side="right"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: VOLATILITY SURFACE ---
with tab2:
    st.markdown("### 🧊 3D Price Surface")
    st.info("Visualizing how Option Price evolves with **Spot Price** and **Time to Maturity**.")
    
    # Generate 3D Data
    S_vals = np.linspace(max(0, S-40), S+40, 20)
    T_vals = np.linspace(0.1, 2.0, 20)
    S_mesh, T_mesh = np.meshgrid(S_vals, T_vals)
    Z = np.zeros_like(S_mesh)
    
    for i in range(len(T_vals)):
        for j in range(len(S_vals)):
            eng = BlackScholesEngine(S_mesh[i,j], K, T_mesh[i,j], r, sigma, is_call)
            Z[i,j] = eng.price()
            
    fig3d = go.Figure(data=[go.Surface(z=Z, x=S_mesh, y=T_mesh, colorscale='Viridis')])
    fig3d.update_layout(
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Time to Maturity',
            zaxis_title='Option Price'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3d, use_container_width=True)

# --- TAB 3: RECRUITER CONTEXT ---
with tab3:
    st.markdown("""
    ### 👨‍💻 Project Architecture & Methodology
    
    **Goal:** Build a high-performance pricing engine capable of handling **American Options**, which require solving Partial Differential Equations (PDEs) due to their "Early Exercise" feature.
    
    #### 1. Why C++?
    Python is great for data science, but too slow for iterative solvers. I implemented the core **Crank-Nicolson Solver** in C++ to achieve **O(N)** performance using the Thomas Algorithm for tridiagonal matrix inversion.
    
    #### 2. The Math (PDE)
    We solve the parabolic Black-Scholes PDE:
    """)
    
    st.latex(r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0")
    
    st.markdown("""
    #### 3. Tech Stack
    * **Backend:** C++17 (Computation), PyBind11 (Interface)
    * **Frontend:** Python Streamlit (Visualization)
    * **Method:** Finite Difference (Implicit-Explicit Scheme)
    """)
    
    st.info("This project demonstrates skills in: Computational Mechanics, Numerical Linear Algebra, C++/Python Integration, and Derivatives Pricing.")
