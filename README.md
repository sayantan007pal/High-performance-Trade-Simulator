# High-Performance Trade Simulator

A real-time trade simulator for cryptocurrency spot markets, leveraging live L2 orderbook data to estimate transaction costs and market impact. Built with Streamlit for a modern, interactive UI.

## Features
- Real-time L2 orderbook streaming from OKX
- Multi-symbol support
- On-the-fly regression modeling for slippage and maker/taker prediction (scikit-learn)
- Almgren-Chriss market impact model
- Dynamic fetching of volatility and fee tier
- Performance benchmarking (latency, UI update, processing)
- Desktop UI with input/output panels

## Model/Algorithm Explanations

### Slippage Estimation
- Uses a regression-based approach to estimate the price impact of executing a market order of a given size.
- Walks the orderbook to simulate the cost of filling the order, then compares to the mid price.
- Linear regression is used for live estimation; can be extended to quantile regression for more robust modeling.

### Market Impact (Almgren-Chriss Model)
- Implements a simplified version of the Almgren-Chriss optimal execution model.
- Market impact is split into temporary and permanent components:
  - **Temporary Impact:** Immediate price change from executing part of the order.
  - **Permanent Impact:** Lasting price change due to information revealed by the trade.
- Execution risk is also considered, based on asset volatility and order size.
- The model balances market impact and execution risk to estimate the true cost of trading.

### Maker/Taker Proportion
- Uses logistic regression to estimate the probability of a trade being executed as a maker or taker, based on orderbook spread and other features.
- Dummy data is used for demonstration; can be extended with historical data for more accuracy.

### Net Cost
- The sum of expected slippage, market impact, and fees.

## Regression/Market Impact Methodology
- **Slippage:**
  - Simulate market order by walking the book for the specified quantity.
  - Calculate the difference between execution price and mid price.
  - Use regression to model slippage as a function of order size, spread, and volatility.
- **Market Impact:**
  - Almgren-Chriss model parameters (gamma, eta, risk aversion) are configurable.
  - Volatility is estimated from recent historical candles.
- **Maker/Taker:**
  - Logistic regression on spread and other features.

## Performance/Optimization Notes
- All async data fetching (WebSocket, REST) is handled in a background thread with a thread-safe queue.
- UI updates and model calculations are performed in the main Streamlit thread to avoid session state issues.
- Latency is measured and displayed for each tick.
- Efficient data structures (numpy arrays, pandas DataFrames) are used for orderbook and analytics.
- The system is designed to process data faster than the stream is received.

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run main.py
   ```
3. Open your browser to `http://localhost:8501`.

## References
- Almgren, R., & Chriss, N. (2000). Optimal Execution of Portfolio Transactions. Journal of Risk.
- https://www.linkedin.com/pulse/understanding-almgren-chriss-model-optimal-portfolio-execution-pal-pmeqc/
- OKX API documentation 