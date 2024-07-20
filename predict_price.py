import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def predict_and_act(data):
    model = load_model('eth_future_price_prediction_v1.h5')

    # 필요한 모든 특성 선택
    features = ['close', 'bb_high', 'bb_low', 'rsi', 'macd_line', 'macd_signal_line', 'obv']
    data = data[features].dropna()  # NaN 값이 있는 행 제거

    # 데이터 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, 0])  # 예측 대상은 'close' 가격
        return np.array(X), np.array(Y)


    X_new, _ = create_dataset(data_scaled, 60)

    predicted_prices = model.predict(X_new)

    current_price = data['close'].iloc[-1]
    predicted_future_price = predicted_prices[-1][0]

    predicted_full = np.zeros((predicted_prices.shape[0], 7))  # 7개의 특성을 가진 배열 생성
    predicted_full[:, 0] = predicted_prices[:, 0]  # 첫 번째 특성에만 예측값 삽입

    # 스케일 되돌리기
    predicted_prices_scaled_back = scaler.inverse_transform(predicted_full)[:, 0]  # 첫 번째 특성만 추출

    # return { current_price, predicted_prices_scaled_back[-1] }
    print("start at ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(f"현재 가격: {current_price}")
    print(f"예상 가격: {predicted_prices_scaled_back[-1]}")
    if predicted_prices_scaled_back[-1] < current_price * 0.99:
        return "sell"
    elif predicted_prices_scaled_back[-1] > current_price * 1.02:
        return "buy"
    else:
        return "hold"
    
