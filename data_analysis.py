import pandas as pd

# Load dataset
df = pd.read_csv("data/sales_data.csv")

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== BASIC STATISTICS ==========")
print(df.describe(include="all"))