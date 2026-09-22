import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ==========================================
# LOAD FEATURE-ENGINEERED DATA
# ==========================================

df = pd.read_csv("data/feature_engineered_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

print("Dataset shape:", df.shape)

# ==========================================
# FEATURES AND TARGET
# ==========================================

features = [
    "Inventory Level",
    "Price",
    "Discount",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "WeekOfYear",
    "IsWeekend",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_30",
    "Rolling_Mean_7",
    "Rolling_Mean_30"
]

target = "Units Sold"

X = df[features]
y = df[target]

# ==========================================
# TIME-BASED TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# ==========================================
# MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

# Avoid division by zero in MAPE
non_zero = y_test != 0

mape = (
    np.mean(
        np.abs(
            (y_test[non_zero] - predictions[non_zero])
            / y_test[non_zero]
        )
    )
    * 100
)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP FEATURES =====")
print(importance.head(10))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "models/demand_forecasting_model.pkl")

print("\nModel saved successfully!")
print("File: models/demand_forecasting_model.pkl")
# ==========================================
# ACTUAL VS PREDICTED DEMAND
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:100],
    label="Actual Demand"
)

plt.plot(
    predictions[:100],
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Demand")
plt.xlabel("Test Data Records")
plt.ylabel("Units Sold")

plt.legend()
plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

print("\nActual vs Predicted graph saved successfully!")
print("File: actual_vs_predicted.png")