import pandas as pd
import numpy as np
from binance.client import Client

import dotenv

# API 키와 시크릿 키 설정 (Binance 계정에서 생성)
api_key = dotenv.get_key('.env', 'BINANCE_API_KEY')
api_secret = dotenv.get_key('.env', 'BINANCE_API_SECRET_KEY')

# Binance 클라이언트 인스턴스 생성
client = Client(api_key, api_secret)

# OHLCV 데이터 가져오기
symbol = 'ETHUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE  # 1분 간격 데이터
start_str = '1 Jan, 2020'

# 데이터를 가져와서 DataFrame으로 변환
klines = client.futures_historical_klines(symbol, interval, start_str)

data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

print(data)

# 필요한 열만 선택하고 데이터 타입 변환
data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)
data = data.astype(float)

# 볼린저 밴드 계산
def bollinger_bands(data, window=20, num_of_std=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    data['bb_high'] = rolling_mean + (rolling_std * num_of_std)
    data['bb_low'] = rolling_mean - (rolling_std * num_of_std)

# RSI 계산
def rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

# MACD 계산
def macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
    data['macd_line'] = fast_ema - slow_ema
    data['macd_signal_line'] = data['macd_line'].ewm(span=signal_period, adjust=False).mean()

# OBV 계산
def obv(data):
    conditions = [
        data['close'] > data['close'].shift(1),
        data['close'] < data['close'].shift(1)
    ]
    choices = [
        data['volume'],
        -data['volume']
    ]
    data['obv'] = np.select(conditions, choices, default=0).cumsum()

# 지표 추가
bollinger_bands(data)
rsi(data)
macd(data)
obv(data)

# 결과 저장
print(data.head())
data.to_csv(f"{symbol}_futures_with_indicators_2024_min.csv")


# import matplotlib.pyplot as plt

# # 데이터 로드
# data = pd.read_csv('BTC_USDT.csv', index_col='timestamp', parse_dates=True)

# # 종가 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(data['close'], label='Close Price')
# plt.title('BTC/USDT Close Price')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()



# data['SMA50'] = data['close'].rolling(window=50).mean()
# data['SMA200'] = data['close'].rolling(window=200).mean()

# # 이동 평균 시각화
# plt.figure(figsize=(12, 6))
# plt.plot(data['close'], label='Close Price')
# plt.plot(data['SMA50'], label='50-day SMA')
# plt.plot(data['SMA200'], label='200-day SMA')
# plt.title('BTC/USDT Moving Averages')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# # 신호 생성
# data['Signal'] = 0
# data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)
# data['Position'] = data['Signal'].diff()

# # 백테스트
# initial_capital = 10000
# positions = pd.DataFrame(index=data.index).fillna(0)
# positions['BTC'] = data['Signal']

# portfolio = positions.multiply(data['close'], axis=0)
# pos_diff = positions.diff()

# portfolio['holdings'] = (positions.multiply(data['close'], axis=0)).sum(axis=1)
# portfolio['cash'] = initial_capital - (pos_diff.multiply(data['close'], axis=0)).sum(axis=1).cumsum()
# portfolio['total'] = portfolio['cash'] + portfolio['holdings']

# portfolio['total'].plot(figsize=(12, 6))

# plt.title('포트폴리오 가치 변화')
# plt.xlabel('Date')
# plt.ylabel('포트폴리오 가치')
# plt.show()