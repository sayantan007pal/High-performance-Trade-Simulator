import streamlit as st
import asyncio
import time
from data.websocket_client import WebSocketOrderBookClient
from data.models import AlmgrenChrissModel, SlippageModel, MakerTakerModel, FeeModel
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="High-Performance Trade Simulator", layout="wide")

st.title("High-Performance Trade Simulator")

# --- Almgren-Chriss Educational Section ---
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

# --- Almgren-Chriss Interactive Section ---
st.subheader("Almgren-Chriss Optimal Execution Simulator")
ac_col1, ac_col2 = st.columns([2, 3])

with ac_col1:
    st.markdown("#### Model Parameters")
    ac_order_size = st.number_input("Order Size (shares)", min_value=10, max_value=10000, value=500, step=10, help="Total shares to execute.")
    ac_time_steps = st.slider("Execution Time Steps", min_value=5, max_value=100, value=51, help="Number of intervals to execute the order.")
    ac_risk_aversion = st.slider("Risk Aversion (λ)", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f", help="Higher = more risk averse (slower execution)")
    ac_temp_impact_eta = st.slider("Temporary Impact η", min_value=0.01, max_value=0.2, value=0.05, step=0.01, help="Temporary impact coefficient.")
    ac_temp_impact_alpha = st.slider("Temporary Impact α", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Temporary impact exponent.")
    ac_perm_impact_gamma = st.slider("Permanent Impact γ", min_value=0.01, max_value=0.2, value=0.05, step=0.01, help="Permanent impact coefficient.")
    ac_perm_impact_beta = st.slider("Permanent Impact β", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Permanent impact exponent.")
    ac_volatility = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01, help="Asset volatility.")
    ac_time_step_size = st.slider("Time Step Size", min_value=0.1, max_value=2.0, value=0.5, step=0.1, help="Length of each time interval.")
    ac_run = st.button("Simulate Optimal Execution")

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
    # Terminal condition
    for shares in range(total_shares + 1):
        value_function[time_steps - 1, shares] = np.exp(shares * temporary_impact(shares / time_step_size, alpha, eta))
        best_moves[time_steps - 1, shares] = shares
    # Backward induction
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
    # Optimal trajectory
    for t in range(1, time_steps):
        inventory_path[t] = inventory_path[t - 1] - best_moves[t, inventory_path[t - 1]]
        optimal_trajectory.append(best_moves[t, inventory_path[t - 1]])
    optimal_trajectory = np.asarray(optimal_trajectory)
    return value_function, best_moves, inventory_path, optimal_trajectory

with ac_col2:
    if ac_run:
        st.markdown("#### Optimal Execution Trajectory")
        value_func, best_moves, inventory_path, optimal_traj = optimal_execution(
            ac_time_steps, ac_order_size, ac_risk_aversion, ac_temp_impact_alpha, ac_perm_impact_beta,
            ac_perm_impact_gamma, ac_temp_impact_eta, ac_volatility, ac_time_step_size)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(inventory_path, color='blue', lw=2)
        ax.set_xlabel('Trading Periods')
        ax.set_ylabel('Inventory Remaining')
        ax.set_title('Optimal Execution Path')
        ax.grid(True)
        st.pyplot(fig)
        st.markdown(f"**Total Order Size:** {ac_order_size}")
        st.markdown(f"**Execution Time Steps:** {ac_time_steps}")
        st.markdown(f"**Risk Aversion (λ):** {ac_risk_aversion}")
        st.markdown(f"**Final Inventory:** {int(inventory_path[-1][0])}")
        st.markdown(f"**Optimal Schedule (shares per step):** {optimal_traj}")
        st.info("Try changing the risk aversion or impact parameters to see how the optimal path changes!")

# --- Original Simulator Section ---
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
            # Simulate orderbook walk for slippage (placeholder)
            mid = None
            if data['bids'] and data['asks']:
                best_bid = float(data['bids'][0][0])
                best_ask = float(data['asks'][0][0])
                mid = (best_bid + best_ask) / 2
            else:
                continue
            # Placeholder: assume order size in base asset is quantity / mid
            order_size = quantity / mid if mid else 0
            # Models
            ac_model = AlmgrenChrissModel(volatility=volatility/100, daily_volume=1000000)
            impact = ac_model.estimate_impact(order_size)
            slip_model = SlippageModel()
            slip_model.fit(np.array([[1],[2],[3]]), np.array([0.1,0.2,0.3]))
            slippage = slip_model.predict([[order_size]])[0]
            mt_model = MakerTakerModel()
            mt_model.fit(np.array([[0],[1],[2],[3]]), np.array([0,1,1,0]))
            maker_prob = mt_model.predict_proba([[order_size]])[0][0]
            taker_prob = mt_model.predict_proba([[order_size]])[0][1]
            # Assume taker for fee
            fee = FeeModel.get_fee(fee_tier, "taker", quantity)
            net_cost = slippage + impact + fee
            # Output
            return {
                "slippage": slippage,
                "fee": fee,
                "impact": impact,
                "net_cost": net_cost,
                "maker_taker": f"{maker_prob:.2f} / {taker_prob:.2f}",
                "latency": ws_latency
            }
    except Exception as e:
        return {"error": str(e)}

def run_async_sim(asset, quantity, volatility, fee_tier):
    return asyncio.run(run_simulation(asset, quantity, volatility, fee_tier))

if run_sim:
    with st.spinner("Running simulation and waiting for first tick..."):
        result = run_async_sim(asset, quantity, volatility, fee_tier)
        if result and "error" not in result:
            output_placeholder.metric("Expected Slippage", f"{result['slippage']:.6f}")
            output_placeholder.metric("Expected Fees", f"{result['fee']:.6f}")
            output_placeholder.metric("Expected Market Impact", f"{result['impact']:.6f}")
            output_placeholder.metric("Net Cost", f"{result['net_cost']:.6f}")
            output_placeholder.metric("Maker/Taker Proportion", result['maker_taker'])
            output_placeholder.metric("Internal Latency (ms)", f"{result['latency']:.2f}")
        elif result and "error" in result:
            st.error(f"Simulation error: {result['error']}")
        else:
            st.warning("No data received from WebSocket.") 