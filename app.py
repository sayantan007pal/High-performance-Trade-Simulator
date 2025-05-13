import streamlit as st
import asyncio
import time
from data.websocket_client import WebSocketOrderBookClient
from data.models import AlmgrenChrissModel, SlippageModel, MakerTakerModel, FeeModel
import numpy as np
import pandas as pd
import io
import plotly.graph_objs as go
import plotly.express as px

# --- Shared Model Functions (available in all tabs) ---
def temporary_impact(volume, alpha, eta):
    return eta * volume ** alpha

def permanent_impact(volume, beta, gamma):
    return gamma * volume ** beta

def hamiltonian(inventory, sell_amount, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step=0.5):
    temp_impact = risk_aversion * sell_amount * permanent_impact(sell_amount / time_step, beta, gamma)
    perm_impact = risk_aversion * (inventory - sell_amount) * time_step * temporary_impact(sell_amount / time_step, alpha, eta)
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step * ((inventory - sell_amount) ** 2)
    return temp_impact + perm_impact + exec_risk

def optimal_execution(time_steps, total_shares, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step_size=0.5):
    value_function = np.zeros((time_steps, total_shares + 1), dtype="float64")
    best_moves = np.zeros((time_steps, total_shares + 1), dtype="int")
    inventory_path = np.zeros((time_steps, 1), dtype="int")
    inventory_path[0] = total_shares
    optimal_trajectory = []
    for shares in range(total_shares + 1):
        value_function[time_steps - 1, shares] = np.exp(shares * temporary_impact(shares / time_step_size, alpha, eta))
        best_moves[time_steps - 1, shares] = shares
    for t in range(time_steps - 2, -1, -1):
        for shares in range(total_shares + 1):
            best_value = value_function[t + 1, 0] * np.exp(hamiltonian(shares, shares, risk_aversion, alpha, beta, gamma, eta, volatility, time_step_size))
            best_share_amount = shares
            for n in range(shares):
                current_value = value_function[t + 1, shares - n] * np.exp(hamiltonian(shares, n, risk_aversion, alpha, beta, gamma, eta, volatility, time_step_size))
                if current_value < best_value:
                    best_value = current_value
                    best_share_amount = n
            value_function[t, shares] = best_value
            best_moves[t, shares] = best_share_amount
    for t in range(1, time_steps):
        inventory_path[t] = inventory_path[t - 1] - best_moves[t, inventory_path[t - 1]]
        optimal_trajectory.append(best_moves[t, inventory_path[t - 1]])
    optimal_trajectory = np.asarray(optimal_trajectory)
    return value_function, best_moves, inventory_path, optimal_trajectory

def cost_breakdown(optimal_traj, alpha, eta, beta, gamma, volatility, risk_aversion, time_step_size):
    temp_impact = 0
    perm_impact = 0
    exec_risk = 0
    inventory = np.sum(optimal_traj)
    per_step_costs = []
    for i, shares in enumerate(optimal_traj):
        temp = temporary_impact(shares / time_step_size, alpha, eta)
        perm = permanent_impact(shares / time_step_size, beta, gamma)
        risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step_size * ((inventory - shares) ** 2)
        temp_impact += temp
        perm_impact += perm
        exec_risk += risk
        per_step_costs.append((temp, perm, risk, temp+perm+risk))
        inventory -= shares
    return float(temp_impact), float(perm_impact), float(exec_risk), per_step_costs

def get_costs_df(optimal_traj, per_step_costs):
    df = pd.DataFrame(per_step_costs, columns=["Temporary Impact", "Permanent Impact", "Execution Risk", "Total Cost"])
    df["Step"] = np.arange(1, len(optimal_traj)+1)
    df["Shares Executed"] = optimal_traj
    cols = ["Step", "Shares Executed", "Temporary Impact", "Permanent Impact", "Execution Risk", "Total Cost"]
    return df[cols]

# --- Always use dark theme CSS for UI polish ---
st.markdown("""
    <style>
    .stApp { background-color: #181c25; color: #e3eafc; }
    .block-container { padding-top: 2rem; }
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #90caf9; }
    .stButton>button { background-color: #3949ab; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #283593; }
    .stMetric { background: #232946; border-radius: 8px; color: #e3eafc; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 High-Performance Trade Simulator")

tabs = st.tabs(["Simulator", "Almgren-Chriss Explorer", "Analytics"])

# --- Theme Toggle in Sidebar ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'
with st.sidebar:
    # st.markdown("## ⚙️ Settings")
    # theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state['theme']=='light' else 1)

    st.session_state['theme'] = 'dark'
    st.markdown("---")
    st.markdown("**Quick Links:**")
    st.markdown("- [OKX API Docs](https://www.okx.com/docs-v5/en/)")
    st.markdown("- [Almgren-Chriss Paper](https://www.linkedin.com/pulse/understanding-almgren-chriss-model-optimal-portfolio-execution-pal-pmeqc/)")

# --- Custom CSS for UI polish and theme ---
if st.session_state['theme'] == 'dark':
    st.markdown("""
        <style>
        .stApp { background-color: #181c25; color: #e3eafc; }
        .block-container { padding-top: 2rem; }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #90caf9; }
        .stButton>button { background-color: #3949ab; color: white; border-radius: 6px; }
        .stButton>button:hover { background-color: #283593; }
        .stMetric { background: #232946; border-radius: 8px; color: #e3eafc; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color: #f7f9fa; }
        .block-container { padding-top: 2rem; }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #1a237e; }
        .stButton>button { background-color: #3949ab; color: white; border-radius: 6px; }
        .stButton>button:hover { background-color: #283593; }
        .stMetric { background: #e3eafc; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

# --- Simulator Tab ---
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Input Parameters")
        exchange = st.selectbox("Exchange", ["OKX"], index=0)
        asset = st.text_input("Spot Asset", "BTC-USDT-SWAP")
        order_type = st.selectbox("Order Type", ["market"], index=0)
        quantity = st.number_input("Quantity (USD equivalent)", min_value=1.0, value=100.0)
        volatility = st.number_input("Volatility (%)", min_value=0.0, value=1.0)
        fee_tier = st.selectbox("Fee Tier", ["Tier 1", "Tier 2", "Tier 3"], index=0)
        run_sim = st.button("Run Simulation")
    with col2:
        st.header("Output Parameters")
        output_placeholder = st.empty()
    async def run_simulation(asset, quantity, volatility, fee_tier):
        ws_client = WebSocketOrderBookClient()
        try:
            async for data, ws_latency in ws_client.connect(asset):
                mid = None
                if data['bids'] and data['asks']:
                    best_bid = float(data['bids'][0][0])
                    best_ask = float(data['asks'][0][0])
                    mid = (best_bid + best_ask) / 2
                else:
                    continue
                order_size = quantity / mid if mid else 0
                time_steps = 20
                lam = 0.01
                alpha = 1.0
                eta = 0.05
                beta = 1.0
                gamma = 0.05
                vol = volatility / 100
                time_step_size = 0.5
                _, _, _, optimal_traj = optimal_execution(
                    time_steps, int(order_size), lam, alpha, beta, gamma, eta, vol, time_step_size)
                temp_impact, perm_impact, exec_risk, _ = cost_breakdown(
                    optimal_traj, alpha, eta, beta, gamma, vol, lam, time_step_size)
                impact = float(temp_impact + perm_impact)
                exec_risk = float(exec_risk)
                slip_model = SlippageModel()
                slip_model.fit(np.array([[1],[2],[3]]), np.array([0.1,0.2,0.3]))
                slippage = float(slip_model.predict([[order_size]])[0])
                mt_model = MakerTakerModel()
                mt_model.fit(np.array([[0],[1],[2],[3]]), np.array([0,1,1,0]))
                maker_prob = float(mt_model.predict_proba([[order_size]])[0][0])
                taker_prob = float(mt_model.predict_proba([[order_size]])[0][1])
                fee = float(FeeModel.get_fee(fee_tier, "taker", quantity))
                net_cost = float(slippage + impact + fee + exec_risk)
                return {
                    "slippage": slippage,
                    "fee": fee,
                    "impact": impact,
                    "exec_risk": exec_risk,
                    "net_cost": net_cost,
                    "maker_taker": f"{maker_prob:.2f} / {taker_prob:.2f}",
                    "latency": float(ws_latency)
                }
        except Exception as e:
            return {"error": str(e)}
    def run_async_sim(asset, quantity, volatility, fee_tier):
        return asyncio.run(run_simulation(asset, quantity, volatility, fee_tier))
    if run_sim:
        with st.spinner("Running simulation and waiting for first tick..."):
            result = run_async_sim(asset, quantity, volatility, fee_tier)
            if result and "error" not in result:
                output_placeholder.metric("Expected Slippage", f"{float(result['slippage']):.6f}")
                output_placeholder.metric("Expected Fees", f"{float(result['fee']):.6f}")
                output_placeholder.metric("Expected Market Impact", f"{float(result['impact']):.6f}")
                output_placeholder.metric("Execution Risk", f"{float(result['exec_risk']):.6f}")
                output_placeholder.metric("Net Cost", f"{float(result['net_cost']):.6f}")
                output_placeholder.metric("Maker/Taker Proportion", result['maker_taker'])
                output_placeholder.metric("Internal Latency (ms)", f"{float(result['latency']):.2f}")
            elif result and "error" in result:
                st.error(f"Simulation error: {result['error']}")
            else:
                st.warning("No data received from WebSocket.")

# --- Almgren-Chriss Explorer Tab ---
with tabs[1]:
    with st.expander("ℹ️ About the Almgren-Chriss Model", expanded=False):
        st.markdown("""
        **The Almgren-Chriss model** is a foundational framework for optimal trade execution. It balances:
        - **Market Impact**: The cost of moving the market when trading large sizes (temporary and permanent impact).
        - **Execution Risk**: The risk of adverse price moves while executing over time.
        
        **Key Formulas:**
        - Temporary Impact: $\\text{Temp} = \\eta x_t$
        - Permanent Impact: $\\text{Perm} = \\gamma X_t$
        - Execution Risk: $\\sigma^2 T / N$
        
        **Analogy:**
        > If you sell all your comic books at once, you crash the price. If you sell slowly, you risk prices moving against you. The model finds the best balance!
        """)
    st.subheader("Almgren-Chriss Optimal Execution Simulator")
    ac_col1, ac_col2 = st.columns([2, 3])
    with ac_col1:
        st.markdown("#### Model Parameters")
        ac_order_size = st.number_input("Order Size (shares)", min_value=10, max_value=10000, value=500, step=10, key="ac_order_size")
        ac_time_steps = st.slider("Execution Time Steps", min_value=5, max_value=100, value=51, key="ac_time_steps")
        ac_risk_aversion_choices = [0.001, 0.01, 0.025, 0.05]
        ac_risk_aversion_selected = st.multiselect(
            "Risk Aversion λ (select one or more)",
            options=ac_risk_aversion_choices,
            default=[0.001, 0.01],
            key="ac_risk_aversion_selected"
        )
        ac_temp_impact_eta = st.slider("Temporary Impact η", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="ac_temp_impact_eta")
        ac_temp_impact_alpha = st.slider("Temporary Impact α", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="ac_temp_impact_alpha")
        ac_perm_impact_gamma = st.slider("Permanent Impact γ", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="ac_perm_impact_gamma")
        ac_perm_impact_beta = st.slider("Permanent Impact β", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="ac_perm_impact_beta")
        ac_volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01, key="ac_volatility")
        ac_time_step_size = st.slider("Time Step Size", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="ac_time_step_size")
        ac_run = st.button("Simulate Optimal Execution", key="ac_run")
    with ac_col2:
        if ac_run and ac_risk_aversion_selected:
            st.markdown("#### Optimal Execution Trajectories")
            # Plotly interactive chart for execution trajectories
            plotly_traces = []
            cost_data = {}
            cost_table_df = None
            cum_cost_fig = None
            for lam in ac_risk_aversion_selected:
                value_func, best_moves, inventory_path, optimal_traj = optimal_execution(
                    ac_time_steps, ac_order_size, lam, ac_temp_impact_alpha, ac_perm_impact_beta,
                    ac_perm_impact_gamma, ac_temp_impact_eta, ac_volatility, ac_time_step_size)
                plotly_traces.append(go.Scatter(
                    x=np.arange(len(inventory_path)),
                    y=inventory_path.flatten(),
                    mode='lines+markers',
                    name=f"λ={lam}"
                ))
                if lam == ac_risk_aversion_selected[0]:
                    temp_impact, perm_impact, exec_risk, per_step_costs = cost_breakdown(
                        optimal_traj, ac_temp_impact_alpha, ac_temp_impact_eta, ac_perm_impact_beta, ac_perm_impact_gamma, ac_volatility, lam, ac_time_step_size)
                    cost_data = {
                        'Temporary Impact': temp_impact,
                        'Permanent Impact': perm_impact,
                        'Execution Risk': exec_risk
                    }
                    cost_table_df = get_costs_df(optimal_traj, per_step_costs)
            fig = go.Figure(plotly_traces)
            fig.update_layout(title="Optimal Execution Path(s)", xaxis_title="Trading Periods", yaxis_title="Inventory Remaining", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            # Pie chart for cost breakdown
            if cost_data:
                st.markdown("#### Cost Breakdown (for first λ)")
                pie_fig = px.pie(
                    names=list(cost_data.keys()),
                    values=list(cost_data.values()),
                    title="Cost Breakdown",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                pie_fig.update_traces(textinfo='percent+label')
                st.plotly_chart(pie_fig, use_container_width=True)
            # Cost analytics table
            if cost_table_df is not None:
                st.markdown("#### Per-Step Cost Analytics (First λ)")
                st.dataframe(cost_table_df, use_container_width=True)
                csv = cost_table_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Cost Analytics CSV", csv, "cost_analytics.csv", "text/csv")
                # Cumulative cost line chart (Plotly)
                cum_cost_fig = go.Figure()
                cum_cost_fig.add_trace(go.Scatter(
                    x=cost_table_df["Step"],
                    y=np.cumsum(cost_table_df["Total Cost"]),
                    mode='lines+markers',
                    name='Cumulative Cost',
                    line=dict(color='#3949ab')
                ))
                cum_cost_fig.update_layout(title="Cumulative Cost Over Time (First λ)", xaxis_title="Step", yaxis_title="Cumulative Cost", template="plotly_dark")
                st.plotly_chart(cum_cost_fig, use_container_width=True)
            st.info("Try changing the risk aversion or impact parameters to see how the optimal path and cost breakdown change!")

# --- Analytics Tab (to be implemented in next steps) ---
with tabs[2]:
    st.header("Advanced Analytics & Session History (Coming Soon)")
    st.info("This section will include session history, parameter sweeps, and advanced downloads.") 