import pytest
from data.websocket_client import WebSocketOrderBookClient
import asyncio

def test_websocket_client_instantiation():
    client = WebSocketOrderBookClient()
    assert client.base_url.startswith("wss://")
    assert hasattr(client, 'connect')
    assert asyncio.iscoroutinefunction(client.connect) 