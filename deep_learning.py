import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('ETHUSDT_futures_with_indicators_2024_min.csv', index_col='timestamp')

# 필요한 모든 특성 선택
features = ['close', 'bb_high', 'bb_low', 'rsi', 'macd_line', 'macd_signal_line', 'obv']
data = data[features].dropna()  # NaN 값이 있는 행 제거

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 전체 데이터를 훈련 데이터와 테스트 데이터로 분할
train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)

# 훈련 데이터를 다시 훈련 세트와 검증 세트로 분할
train, val = train_test_split(train_data, test_size=0.25, shuffle=False)  # 0.25 * 0.8 = 0.2 of the original data

# 데이터를 시퀀스로 변환하는 함수
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])  # 예측 대상은 'close' 가격
    return np.array(X), np.array(Y)

look_back = 60

# LSTM을 위한 입력 데이터 형태 조정
X_train, Y_train = create_dataset(train, look_back)
X_val, Y_val = create_dataset(val, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# LSTM 모델 구축
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, len(features))),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping을 설정합니다. patience는 성능 향상이 멈춘 후 몇 epoch 더 기다릴지를 정의합니다.
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# 모델 훈련 (early stopping을 콜백으로 추가)
model.fit(
    X_train,
    Y_train,
    epochs=100,  # 큰 수의 epoch를 초기 설정
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping_monitor],
    verbose=2
)

# 모델 저장
model.save('eth_future_price_prediction_v1.h5')

# 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 결과 스케일 되돌리기 (주의: 스케일을 되돌릴 때는 'close' 가격만 처리)
train_predict = scaler.inverse_transform(np.c_[train_predict, np.zeros((train_predict.shape[0], len(features)-1))])[:,0]
test_predict = scaler.inverse_transform(np.c_[test_predict, np.zeros((test_predict.shape[0], len(features)-1))])[:,0]
Y_train_inv = scaler.inverse_transform(np.c_[Y_train, np.zeros((Y_train.shape[0], len(features)-1))])[:,0]
Y_test_inv = scaler.inverse_transform(np.c_[Y_test, np.zeros((Y_test.shape[0], len(features)-1))])[:,0]

# 결과 시각화
plt.plot(Y_test_inv, label='True value')
plt.plot(test_predict, label='Predicted value')
plt.title('Stock Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
