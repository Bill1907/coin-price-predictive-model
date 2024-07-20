import datetime
import pandas as pd
import numpy as np
from binance.client import Client
import dotenv

def fetch_real_time_data(symbol, type='spot'):
    # 실시간 데이터를 수집하는 API 호출
    # API로부터 데이터를 받아 DataFrame으로 변환
    # API 키와 시크릿 키 설정 (Binance 계정에서 생성)
    api_key = dotenv.get_key('.env', 'BINANCE_API_KEY')
    api_secret = dotenv.get_key('.env', 'BINANCE_API_SECRET_KEY')

    # Binance 클라이언트 인스턴스 생성
    client = Client(api_key, api_secret)

    # OHLCV 데이터 가져오기

    interval = Client.KLINE_INTERVAL_1MINUTE  # 1분 간격 데이터
    start_str = datetime.datetime.now().strftime('%d %b, %Y')
    limit = 60

    # 데이터를 가져와서 DataFrame으로 변환
    if type == 'spot':
        klines = client.get_historical_klines(symbol, interval, "1 day ago UTC")
    else:
        klines = client.futures_historical_klines(symbol, interval, "1 day ago UTC")
        
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

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
    return data