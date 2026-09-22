import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sales_data.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data by product, store, and date
df = df.sort_values(
    by=["Store ID", "Product ID", "Date"]
).reset_index(drop=True)

# ==============================
# TIME-BASED FEATURES
# ==============================

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# ==============================
# LAG FEATURES
# ==============================

group = df.groupby(["Store ID", "Product ID"])["Units Sold"]

df["Lag_1"] = group.shift(1)
df["Lag_7"] = group.shift(7)
df["Lag_14"] = group.shift(14)
df["Lag_30"] = group.shift(30)

# ==============================
# ROLLING AVERAGE FEATURES
# ==============================

df["Rolling_Mean_7"] = (
    df.groupby(["Store ID", "Product ID"])["Units Sold"]
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

df["Rolling_Mean_30"] = (
    df.groupby(["Store ID", "Product ID"])["Units Sold"]
    .transform(lambda x: x.shift(1).rolling(window=30).mean())
)

# Remove rows where lag/rolling features are not available
df = df.dropna().reset_index(drop=True)

# Save feature-engineered dataset
df.to_csv(
    "data/feature_engineered_data.csv",
    index=False
)

# ==============================
# INFORMATION
# ==============================

print("===== FEATURE ENGINEERING COMPLETE =====")
print("Original dataset rows: 73100")
print("Rows after feature engineering:", df.shape[0])
print("Columns:", df.shape[1])

print("\nNew features created:")
print([
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
])

print("\nFeature-engineered dataset saved successfully!")
print("File: data/feature_engineered_data.csv")