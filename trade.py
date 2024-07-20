import pandas as pd
import numpy as np
from binance.client import Client
import dotenv

def place_order(symbol='BTCUSDT', side='BUY') :
    api_key = dotenv.get_key('.env', 'BINANCE_API_KEY')
    api_secret = dotenv.get_key('.env', 'BINANCE_API_SECRET_KEY')
    
    client = Client(api_key, api_secret)

    if side == 'SELL':
        # symbol의 전체 수량 확인
        balance = client.get_asset_balance(asset='ETH')
        balance = float(balance['free'])
        if balance == 0:
            return 'No ETH to sell'
        # 주문 생성
        order = client.create_order(
            symbol=symbol,
            side='SELL',
            type='MARKET',
            quantity=balance
        )
        return order
    elif side == 'BUY':
        # 살수 있는 최대 수량 확인
        balance = client.get_asset_balance(asset='USDT')
        balance = float(balance['free'])
        price = client.get_avg_price(symbol=symbol)
        price = float(price['price'])
        quantity = balance / price

        # 주문 생성
        order = client.create_order(
            symbol=symbol,
            side='BUY',
            type='MARKET',
            quantity=quantity
        )
        return order

def future_place_order(symbol='BTCUSDT', side='BUY'):
    api_key = dotenv.get_key('.env', 'BINANCE_API_KEY')
    api_secret = dotenv.get_key('.env', 'BINANCE_API_SECRET_KEY')

    