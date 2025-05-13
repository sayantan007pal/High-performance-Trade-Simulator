import aiohttp
import asyncio
import logging

# Placeholder endpoints (replace with actual OKX endpoints if available)
FEE_TIER_URL = "https://www.okx.com/api/v5/account/trade-fee?instType=SPOT&instId={symbol}"
VOLATILITY_URL = "https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar=1m&limit=60"
ASSETS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SPOT"

logging.basicConfig(level=logging.INFO)

class RestClient:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def get_fee_tier(self, symbol: str):
        """Fetch fee tier for a given symbol from OKX."""
        url = FEE_TIER_URL.format(symbol=symbol)
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()
                return data['data'][0]['taker']
        except Exception as e:
            logging.error(f"Error fetching fee tier: {e}")
            return None

    async def get_volatility(self, symbol: str):
        """Fetch recent volatility for a given symbol from OKX."""
        url = VOLATILITY_URL.format(symbol=symbol)
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()
                closes = [float(c[4]) for c in data['data']]
                if len(closes) < 2:
                    return None
                returns = [(closes[i+1] - closes[i]) / closes[i] for i in range(len(closes)-1)]
                volatility = (sum((r ** 2 for r in returns)) / len(returns)) ** 0.5
                return volatility
        except Exception as e:
            logging.error(f"Error fetching volatility: {e}")
            return None

    async def get_spot_assets(self):
        """Fetch available spot assets from OKX."""
        url = ASSETS_URL
        try:
            async with self.session.get(url) as resp:
                data = await resp.json()
                assets = [item['instId'] for item in data['data']]
                return assets
        except Exception as e:
            logging.error(f"Error fetching spot assets: {e}")
            return []

# Standalone test
if __name__ == "__main__":
    async def main():
        async with RestClient() as rc:
            fee = await rc.get_fee_tier("BTC-USDT")
            vol = await rc.get_volatility("BTC-USDT")
            assets = await rc.get_spot_assets()
            print("Fee tier:", fee)
            print("Volatility:", vol)
            print("Assets:", assets[:5])
    asyncio.run(main()) 