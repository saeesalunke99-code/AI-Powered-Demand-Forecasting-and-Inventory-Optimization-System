import pandas as pd

# Load dataset
df = pd.read_csv("data/sales_data.csv")

print("Original dataset shape:", df.shape)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicates
print("\nDuplicate rows:", df.duplicated().sum())

# Sort data chronologically
df = df.sort_values(
    by=["Store ID", "Product ID", "Date"]
).reset_index(drop=True)

# Display basic information
print("\nData types after cleaning:")
print(df.dtypes)

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nCleaned dataset shape:", df.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_sales_data.csv", index=False)

print("\nCleaned dataset saved successfully!")