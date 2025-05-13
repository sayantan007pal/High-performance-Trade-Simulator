import pytest
from data.websocket_client import WebSocketOrderBookClient
import asyncio
import inspect

def test_websocket_client_instantiation():
    client = WebSocketOrderBookClient()
    assert client.base_url.startswith("wss://")
    assert hasattr(client, 'connect')
    agen = client.connect("BTC-USDT-SWAP")
    assert inspect.isasyncgen(agen) 