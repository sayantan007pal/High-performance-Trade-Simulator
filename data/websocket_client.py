import asyncio
import websockets
import json
import time
import logging
from typing import Callable, Dict, List

logging.basicConfig(level=logging.INFO)

OKX_WS_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/"

class WebSocketManager:
    def __init__(self, symbols: List[str], on_data: Callable[[str, dict], None]):
        self.symbols = symbols
        self.on_data = on_data
        self.tasks = []
        self.running = False

    async def connect_symbol(self, symbol: str):
        url = OKX_WS_URL + symbol
        while self.running:
            try:
                async with websockets.connect(url) as ws:
                    print(f"Connected to {symbol}")
                    async for msg in ws:
                        data = json.loads(msg)
                        self.on_data(symbol, data)
            except Exception as e:
                print(f"WebSocket error for {symbol}: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2)

    async def start(self):
        self.running = True
        self.tasks = [asyncio.create_task(self.connect_symbol(symbol)) for symbol in self.symbols]
        await asyncio.gather(*self.tasks)

    async def stop(self):
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

class WebSocketOrderBookClient:
    def __init__(self, base_url="wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/"):
        self.base_url = base_url
        self.ws = None
        self.symbol = None
        self._running = False

    async def connect(self, symbol):
        self.symbol = symbol
        url = f"{self.base_url}{symbol}"
        while True:
            try:
                async with websockets.connect(url) as ws:
                    self.ws = ws
                    self._running = True
                    logging.info(f"Connected to {url}")
                    async for message in ws:
                        start = time.perf_counter()
                        data = json.loads(message)
                        latency = (time.perf_counter() - start) * 1000
                        yield data, latency
            except Exception as e:
                logging.error(f"WebSocket error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    def stop(self):
        self._running = False

# Standalone test
if __name__ == "__main__":
    def print_data(symbol, data):
        print(f"[{symbol}] {data['timestamp']}")
    symbols = ["BTC-USDT-SWAP"]
    wsm = WebSocketManager(symbols, print_data)
    asyncio.run(wsm.start()) 