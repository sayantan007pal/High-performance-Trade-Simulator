# High-Performance Trade Simulator

## Overview
This simulator estimates transaction costs and market impact for cryptocurrency spot trading using real-time L2 orderbook data from OKX. It features a user interface for parameter input and displays processed outputs including slippage, fees, market impact, and latency metrics.

## Features
- Real-time L2 orderbook data via WebSocket
- User interface (Streamlit) for parameter input and output display
- Models for slippage, fees, market impact (Almgren-Chriss), and maker/taker prediction
- Performance benchmarking and optimization

## UI Components
- **Left Panel:** Input parameters (exchange, asset, order type, quantity, volatility, fee tier)
- **Right Panel:** Output parameters (expected slippage, fees, market impact, net cost, maker/taker proportion, internal latency)

## Model Selection and Parameters
### Slippage Estimation
- Uses linear or quantile regression (scikit-learn) on historical or simulated orderbook data.
- Features: order size, volatility, orderbook depth, etc.

### Market Impact (Almgren-Chriss)
- Implements the Almgren-Chriss model for optimal execution and market impact estimation.
- Parameters: volatility, daily volume, risk aversion, order size.
- Reference: [Almgren-Chriss Model](https://www.linkedin.com/pulse/understanding-almgren-chriss-model-optimal-portfolio-execution-pal-pmeqc/)

### Fee Model
- Rule-based, based on OKX fee tiers (see FeeModel in `models.py`).

### Maker/Taker Proportion
- Logistic regression to estimate the probability of an order being maker or taker.

## Regression Techniques
- **Linear Regression:** For slippage estimation.
- **Quantile Regression:** For robust slippage estimation under outliers.
- **Logistic Regression:** For maker/taker prediction.

## Market Impact Calculation
- Almgren-Chriss model is used to estimate the cost of executing large orders over time, considering volatility and liquidity.

## Performance Optimization
- **Async WebSocket:** Non-blocking, high-throughput data processing.
- **Efficient Data Structures:** Uses numpy/pandas for orderbook and tick data.
- **Threading/Async:** Separates network, UI, and model computation.
- **Memory Profiling:** Tools like `memory_profiler` can be used for bottleneck analysis.

## Benchmarking
- **Data Processing Latency:** Time to process each orderbook tick.
- **UI Update Latency:** Time to update UI with new results.
- **End-to-End Latency:** Total time from data receipt to output display.

## Error Handling & Logging
- Structured logging using Python's `logging` module.
- Robust exception handling in WebSocket and model code.

## Running the Simulator
1. Install requirements: `pip install -r requirements.txt`
2. Start the UI(1.0): `streamlit run app.py`
3. Start the UI(2.0): `streamlit run main.py`
3. Running the tests : `python -m pytest tests/`
## File Structure
- `app.py`: Streamlit UI(1.0)
- `main.py`: Streamlit UI(2.0)
- `data/websocket_client.py`: WebSocket client for orderbook data
- `data/models.py`: Models for slippage, market impact, fees, maker/taker
- `README.md`: Documentation

## License
MIT 