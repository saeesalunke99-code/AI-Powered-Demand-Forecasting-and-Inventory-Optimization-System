import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

print("===== DATASET INFORMATION =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== PRODUCTS =====")
print("Number of products:", df["Product ID"].nunique())

print("\n===== STORES =====")
print("Number of stores:", df["Store ID"].nunique())

print("\n===== CATEGORIES =====")
print(df["Category"].unique())

# Product sales
product_sales = (
    df.groupby("Product ID")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== TOP 10 PRODUCTS BY SALES =====")
print(product_sales.head(10))

# Category sales
category_sales = (
    df.groupby("Category")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY CATEGORY =====")
print(category_sales)

# Store sales
store_sales = (
    df.groupby("Store ID")["Units Sold"]
    .sum()
    .sort_values(ascending=False)
)

print("\n===== SALES BY STORE =====")
print(store_sales)

# Monthly sales
monthly_sales = (
    df.groupby(df["Date"].dt.to_period("M"))["Units Sold"]
    .sum()
)

print("\n===== MONTHLY SALES =====")
print(monthly_sales)

# Convert index to string for plotting
monthly_sales.index = monthly_sales.index.astype(str)

# Plot monthly sales
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Units Sold")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("monthly_sales.png")

print("\nMonthly sales chart saved as monthly_sales.png")


# ==========================================
# INVENTORY VS SALES ANALYSIS
# ==========================================

print("\n===== INVENTORY VS SALES CORRELATION =====")

correlation = df["Inventory Level"].corr(df["Units Sold"])

print(
    "Correlation between Inventory Level and Units Sold:",
    round(correlation, 4)
)

plt.figure(figsize=(8, 5))

sample = df.sample(2000, random_state=42)

plt.scatter(
    sample["Inventory Level"],
    sample["Units Sold"]
)

plt.title("Inventory Level vs Units Sold")
plt.xlabel("Inventory Level")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig("inventory_vs_sales.png")

print("Inventory vs Sales graph saved successfully!")