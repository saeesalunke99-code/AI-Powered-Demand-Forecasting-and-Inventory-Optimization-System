import pandas as pd
import numpy as np

# ==========================================
# LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/cleaned_sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Sort data
df = df.sort_values(
    by=["Store ID", "Product ID", "Date"]
).reset_index(drop=True)


# ==========================================
# INVENTORY OPTIMIZATION PARAMETERS
# ==========================================

# Assumed supplier lead time
LEAD_TIME_DAYS = 7

# Service level factor
# Approximately 95% service level
SERVICE_LEVEL_Z = 1.65


# ==========================================
# CALCULATE DEMAND STATISTICS
# ==========================================

demand_stats = (
    df.groupby(["Store ID", "Product ID"])["Units Sold"]
    .agg(
        Average_Daily_Demand="mean",
        Demand_Std="std"
    )
    .reset_index()
)


# ==========================================
# SAFETY STOCK
# ==========================================

demand_stats["Safety_Stock"] = (
    SERVICE_LEVEL_Z
    * demand_stats["Demand_Std"]
    * np.sqrt(LEAD_TIME_DAYS)
)


# ==========================================
# REORDER POINT
# ==========================================

demand_stats["Reorder_Point"] = (
    demand_stats["Average_Daily_Demand"]
    * LEAD_TIME_DAYS
    + demand_stats["Safety_Stock"]
)


# ==========================================
# CURRENT INVENTORY
# ==========================================

latest_inventory = (
    df.sort_values("Date")
    .groupby(["Store ID", "Product ID"])
    .tail(1)
    [["Store ID", "Product ID", "Inventory Level"]]
)


# Merge current inventory
inventory = demand_stats.merge(
    latest_inventory,
    on=["Store ID", "Product ID"],
    how="left"
)


# ==========================================
# RECOMMENDED ORDER QUANTITY
# ==========================================

inventory["Recommended_Order"] = (
    inventory["Reorder_Point"]
    - inventory["Inventory Level"]
).clip(lower=0)


# Round values
inventory["Average_Daily_Demand"] = (
    inventory["Average_Daily_Demand"].round(2)
)

inventory["Demand_Std"] = (
    inventory["Demand_Std"].round(2)
)

inventory["Safety_Stock"] = (
    inventory["Safety_Stock"].round(2)
)

inventory["Reorder_Point"] = (
    inventory["Reorder_Point"].round(2)
)

inventory["Recommended_Order"] = (
    inventory["Recommended_Order"].round(0)
)


# ==========================================
# INVENTORY STATUS
# ==========================================

inventory["Status"] = np.where(
    inventory["Inventory Level"] <= inventory["Reorder_Point"],
    "REORDER",
    "SUFFICIENT"
)


# ==========================================
# SAVE RESULTS
# ==========================================

inventory.to_csv(
    "data/inventory_optimization_results.csv",
    index=False
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("===== INVENTORY OPTIMIZATION COMPLETE =====")

print("\nTotal Store-Product combinations:",
      len(inventory))

print("\nProducts requiring reorder:",
      (inventory["Status"] == "REORDER").sum())

print("\nProducts with sufficient inventory:",
      (inventory["Status"] == "SUFFICIENT").sum())

print("\n===== SAMPLE RESULTS =====")

print(
    inventory.head(10).to_string(index=False)
)

print("\nInventory optimization results saved successfully!")

print(
    "File: data/inventory_optimization_results.csv"
)