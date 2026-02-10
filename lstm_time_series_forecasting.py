# ============================================================
# Advanced Time Series Forecasting with Neural Networks
# Using LSTM and Permutation Feature Importance
# ============================================================

# -----------------------------
# 1. Imports
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 2. Dataset Generation
# -----------------------------
np.random.seed(42)

n_steps = 1500
time = np.arange(n_steps)

trend = time * 0.005
seasonality = 10 * np.sin(2 * np.pi * time / 50)
temperature = 20 + 5 * np.sin(2 * np.pi * time / 365)
noise = np.random.normal(0, 1, n_steps)

energy = 50 + trend + seasonality + 0.5 * temperature + noise

data = pd.DataFrame({
    "energy": energy,
    "temperature": temperature,
    "day_of_week": time % 7
})

print("Dataset sample:")
print(data.head())

# -----------------------------
# 3. Data Scaling
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# 4. Sequence Creation (Sliding Window)
# -----------------------------
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # energy is target
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# -----------------------------
# 5. Train / Validation / Test Split
# -----------------------------
train_size = int(0.7 * len(X))
val_size = int(0.85 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

print("\nData Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# -----------------------------
# 6. LSTM Model Definition
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# -----------------------------
# 7. Model Training
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# -----------------------------
# 8. Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100

print("\nModel Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)

# -----------------------------
# 9. Permutation Feature Importance
# -----------------------------
def permutation_importance(model, X, y):
    baseline = mean_squared_error(y, model.predict(X))
    importances = []

    for feature in range(X.shape[2]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, :, feature])
        permuted_score = mean_squared_error(y, model.predict(X_permuted))
        importances.append(permuted_score - baseline)

    return importances

feature_names = ["energy", "temperature", "day_of_week"]
importances = permutation_importance(model, X_test, y_test)

print("\nPermutation Feature Importance:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance}")

# -----------------------------
# 10. Visualization
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title("Actual vs Predicted Energy Consumption")
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances)
plt.title("Feature Importance (Permutation)")
plt.ylabel("Increase in MSE")
plt.show()
