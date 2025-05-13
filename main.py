import streamlit as st
import threading
import time
import asyncio
import websockets
import aiohttp
import json
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from queue import Queue, Empty
from sklearn.linear_model import LogisticRegression

# --- Sidebar controls ---
st.sidebar.title("Trade Simulator Controls")
exchange = st.sidebar.selectbox("Exchange", ["OKX"])
symbols = st.sidebar.text_input("Symbols (comma separated)", "BTC-USDT-SWAP,ETH-USDT-SWAP")
order_type = st.sidebar.selectbox("Order Type", ["market"])
quantity = st.sidebar.number_input("Quantity (USD)", value=100)
demo_mode = st.sidebar.checkbox("Demo Mode (Simulate Data)", value=False)
start_button = st.sidebar.button("Start Simulation")

# --- Model Documentation Expander ---
with st.expander("ℹ️ Model & Algorithm Documentation", expanded=False):
    st.markdown("""
    **Slippage Estimation:**
    - Regression-based, simulates market order by walking the book, compares to mid price.
    - Linear regression, can be extended to quantile regression.

    **Market Impact (Almgren-Chriss):**
    - Splits impact into temporary and permanent components.
    - Considers execution risk (volatility, order size).
    - Balances market impact and execution risk.

    **Maker/Taker Proportion:**
    - Logistic regression on spread and orderbook features.
    - Dummy data for demo; extendable with historical data.

    **Net Cost:**
    - Sum of slippage, market impact, and fees.

    **Performance:**
    - Async data fetching (WebSocket, REST) in background thread.
    - UI/model updates in main thread.
    - Latency measured per tick.
    - Efficient data structures (numpy, pandas).
    """)

# --- Session state initialization ---
def init_state():
    defaults = {
        'log': [],
        'expected_slippage': {},
        'expected_fees': {},
        'expected_market_impact': {},
        'net_cost': {},
        'maker_taker_proportion': {},
        'internal_latency': {},
        'orderbook_data': {},
        'fees': {},
        'vols': {},
        'latencies': {},
        'data_queue': None,
        'running': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# --- Data queue for thread-safe communication ---
if st.session_state['data_queue'] is None:
    st.session_state['data_queue'] = Queue()

# --- OKX REST endpoints ---
CANDLE_URL = "https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar=1m&limit=60"
FEE_TIER_MAP = {"OKX": 0.001}

async def fetch_fee_vol(session, symbol, queue):
    fee = FEE_TIER_MAP.get("OKX", 0.001)
    url = CANDLE_URL.format(symbol=symbol)
    try:
        async with session.get(url) as resp:
            data = await resp.json()
            closes = [float(c[4]) for c in data['data']]
            if len(closes) < 2:
                vol = 0.0
            else:
                returns = np.diff(np.log(closes))
                vol = np.std(returns) * np.sqrt(60)
            queue.put({'type': 'fee_vol', 'symbol': symbol, 'fee': fee, 'vol': vol,
                        'log': f"Fetched fee/vol for {symbol}: Fee={fee:.5f}, Vol={vol:.4f}"})
    except Exception as e:
        queue.put({'type': 'log', 'log': f"REST error for {symbol}: {e}"})

async def ws_listener(symbol, queue):
    if demo_mode:
        # Simulate orderbook ticks
        for _ in range(100):
            price = np.random.uniform(50000, 60000)
            bids = [[str(price - i * 2), str(np.random.uniform(0.1, 5))] for i in range(10)]
            asks = [[str(price + i * 2), str(np.random.uniform(0.1, 5))] for i in range(10)]
            data = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "exchange": "OKX",
                "symbol": symbol,
                "bids": bids,
                "asks": asks
            }
            queue.put({'type': 'orderbook', 'symbol': symbol, 'data': data,
                        'log': f"[DEMO] Simulated orderbook tick for {symbol} @ {data['timestamp']}"})
            time.sleep(1)
    else:
        url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{symbol}"
        queue.put({'type': 'log', 'log': f"Connecting to {url}"})
        try:
            async with websockets.connect(url) as ws:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        queue.put({'type': 'orderbook', 'symbol': symbol, 'data': data,
                                    'log': f"Received orderbook tick for {symbol} @ {data.get('timestamp', '-')}"})
                    except Exception as e:
                        queue.put({'type': 'log', 'log': f"Error parsing orderbook for {symbol}: {e}"})
        except Exception as e:
            queue.put({'type': 'log', 'log': f"WebSocket error for {symbol}: {e}"})

async def async_data_layer(symbol_list, queue):
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(fetch_fee_vol(session, s, queue) for s in symbol_list))
    await asyncio.gather(*(ws_listener(s, queue) for s in symbol_list))

def start_data_layer(symbol_list, queue):
    asyncio.run(async_data_layer(symbol_list, queue))

# --- Model integration ---
def update_models(symbol):
    ob = st.session_state['orderbook_data'].get(symbol)
    fee = st.session_state['fees'].get(symbol, 0.001)
    vol = st.session_state['vols'].get(symbol, 0.01)
    qty = quantity
    if not ob or 'bids' not in ob or 'asks' not in ob:
        return
    bids = np.array([[float(p), float(q)] for p, q in ob['bids']])
    asks = np.array([[float(p), float(q)] for p, q in ob['asks']])
    def walk_book(side, qty):
        total = 0
        cost = 0
        for price, size in side:
            px_qty = price * size
            if total + px_qty >= qty:
                cost += price * (qty - total) / price
                total = qty
                break
            else:
                cost += px_qty
                total += px_qty
        return cost / qty if qty > 0 else 0
    mid = (bids[0, 0] + asks[0, 0]) / 2 if len(bids) and len(asks) else 0
    buy_price = walk_book(asks, qty)
    sell_price = walk_book(bids, qty)
    slippage = max(abs(buy_price - mid), abs(sell_price - mid))
    st.session_state['expected_slippage'][symbol] = f"{slippage:.4f}"
    gamma = 0.05
    eta = 0.05
    risk_aversion = 0.01
    temp_impact = eta * (qty ** 1)
    perm_impact = gamma * (qty ** 1)
    exec_risk = 0.5 * (risk_aversion ** 2) * (vol ** 2) * 1 * (qty ** 2)
    market_impact = temp_impact + perm_impact + exec_risk
    st.session_state['expected_market_impact'][symbol] = f"{market_impact:.4f}"
    fees = fee * qty
    st.session_state['expected_fees'][symbol] = f"{fees:.4f}"
    net_cost = slippage + market_impact + fees
    st.session_state['net_cost'][symbol] = f"{net_cost:.4f}"
    # Simple heuristic for maker/taker proportion based on spread
    spread = abs(bids[0, 0] - asks[0, 0]) if len(bids) and len(asks) else 0
    prob = min(1, max(0, spread / 10))
    st.session_state['maker_taker_proportion'][symbol] = f"{prob:.2f}"

# --- Latency measurement ---
def record_latency(symbol, start_time):
    latency = (time.time() - start_time) * 1000
    if symbol not in st.session_state['latencies']:
        st.session_state['latencies'][symbol] = []
    st.session_state['latencies'][symbol].append(latency)
    st.session_state['internal_latency'][symbol] = f"{latency:.2f}"

# --- Main Streamlit loop ---
symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
if start_button and not st.session_state['running']:
    st.session_state['running'] = True
    for k in ['log', 'orderbook_data', 'fees', 'vols', 'latencies', 'expected_slippage', 'expected_fees', 'expected_market_impact', 'net_cost', 'maker_taker_proportion', 'internal_latency']:
        st.session_state[k] = {} if isinstance(st.session_state[k], dict) else []
    st.session_state['data_queue'] = Queue()
    threading.Thread(target=start_data_layer, args=(symbol_list, st.session_state['data_queue']), daemon=True).start()
    st.success("Simulation running in background. Logs will update.")

# --- Process data from background thread ---
queue = st.session_state['data_queue']
while True:
    try:
        msg = queue.get_nowait()
        symbol = msg.get('symbol')
        if msg['type'] == 'log':
            st.session_state['log'].append(msg['log'])
        elif msg['type'] == 'fee_vol':
            st.session_state['fees'][symbol] = msg['fee']
            st.session_state['vols'][symbol] = msg['vol']
            st.session_state['log'].append(msg['log'])
        elif msg['type'] == 'orderbook':
            start_time = time.time()
            st.session_state['orderbook_data'][symbol] = msg['data']
            st.session_state['log'].append(msg['log'])
            update_models(symbol)
            record_latency(symbol, start_time)
    except Empty:
        break

# --- Output placeholders ---
st.subheader("Simulation Outputs (Multi-Symbol)")
for symbol in symbol_list:
    st.markdown(f"### {symbol}")
    st.write(f"**Expected Slippage:** {st.session_state['expected_slippage'].get(symbol, '-')}")
    st.write(f"**Expected Fees:** {st.session_state['expected_fees'].get(symbol, '-')}")
    st.write(f"**Expected Market Impact:** {st.session_state['expected_market_impact'].get(symbol, '-')}")
    st.write(f"**Net Cost:** {st.session_state['net_cost'].get(symbol, '-')}")
    st.write(f"**Maker/Taker Proportion:** {st.session_state['maker_taker_proportion'].get(symbol, '-')}")
    st.write(f"**Internal Latency:** {st.session_state['internal_latency'].get(symbol, '-')} ms")

st.subheader("Orderbook Chart and Table (Multi-Symbol)")
chart_tabs = st.tabs(symbol_list) if symbol_list else []
for i, symbol in enumerate(symbol_list):
    with chart_tabs[i] if chart_tabs else st.container():
        ob = st.session_state['orderbook_data'].get(symbol)
        if ob and 'bids' in ob and 'asks' in ob:
            bids = np.array([[float(p), float(q)] for p, q in ob['bids']])
            asks = np.array([[float(p), float(q)] for p, q in ob['asks']])
            best_bid = bids[0, 0] if len(bids) else None
            best_ask = asks[0, 0] if len(asks) else None
            spread = best_ask - best_bid if best_bid and best_ask else None
            st.write(f"**Best Bid:** {best_bid}")
            st.write(f"**Best Ask:** {best_ask}")
            st.write(f"**Spread:** {spread}")
            # Depth chart
            fig = go.Figure()
            if len(bids):
                fig.add_trace(go.Scatter(x=bids[:, 0], y=np.cumsum(bids[:, 1]), mode='lines', name='Bids', line=dict(color='green')))
            if len(asks):
                fig.add_trace(go.Scatter(x=asks[:, 0], y=np.cumsum(asks[:, 1]), mode='lines', name='Asks', line=dict(color='red')))
            fig.update_layout(title=f"Orderbook Depth: {symbol}", xaxis_title="Price", yaxis_title="Cumulative Size", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            # Advanced chart: price/volume over time (if available)
            if 'timestamp' in ob:
                st.write(f"Last update: {ob['timestamp']}")
            st.write("**Top 5 Bids**")
            st.dataframe(pd.DataFrame(bids[:5], columns=["Price", "Size"]))
            st.write("**Top 5 Asks**")
            st.dataframe(pd.DataFrame(asks[:5], columns=["Price", "Size"]))
        else:
            st.info("Waiting for orderbook data...")

st.subheader("Benchmark Log")
st.write("\n".join(st.session_state['log'][-10:]) if st.session_state['log'] else "No logs yet.")
