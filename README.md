# High-Performance Trade Simulator

A desktop application for simulating cryptocurrency trades using real-time L2 orderbook data, estimating transaction costs, slippage, and market impact. Built with PyQt6 for a responsive UI and leveraging async programming for high performance.

## Features
- Real-time L2 orderbook streaming from OKX (and other exchanges in future)
- Multi-symbol support
- On-the-fly regression modeling for slippage and maker/taker prediction (scikit-learn)
- Almgren-Chriss market impact model
- Dynamic fetching of fee tiers and volatility
- Performance benchmarking (latency, UI update, processing)
- Desktop UI with input/output panels

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the application:**
   ```bash
   python main.py
   ```

## Architecture
- **UI:** PyQt6 (desktop, left: input, right: output)
- **Async Data:** `asyncio`, `websockets`, `aiohttp`
- **Modeling:** `scikit-learn`, `numpy`, `pandas`
- **Benchmarking:** Built-in logging and UI display

## Roadmap
- Add multi-symbol streaming and processing
- Integrate real-time model training
- Implement Almgren-Chriss and regression models
- Expand benchmarking and optimization

## Notes
- No real orders are sent; this is a simulation based on orderbook data only.
- Fee tiers and volatility are fetched dynamically from the exchange. 