# 📦 AI-Powered Demand Forecasting and Inventory Optimization System

An end-to-end machine learning system that forecasts retail product demand and provides inventory replenishment recommendations using historical sales and inventory data.

## 📌 Project Overview

Retail businesses need to maintain enough inventory to satisfy customer demand while avoiding unnecessary overstock.

This project combines:

- Historical retail sales analysis
- Feature engineering
- Machine learning-based demand forecasting
- Inventory optimization
- Reorder point calculation
- Recommended order quantity
- Interactive Streamlit dashboard

The system allows users to select a store and product, generate a demand prediction, and receive an inventory replenishment recommendation.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze historical retail sales and inventory data.
2. Identify patterns in product demand.
3. Create useful time-based and historical demand features.
4. Train a machine learning model to forecast demand.
5. Calculate inventory reorder requirements.
6. Provide actionable inventory recommendations.
7. Build an interactive dashboard for users.

---

## 🏗️ Project Architecture

```text
Historical Sales + Inventory Data
              ↓
        Data Cleaning
              ↓
        Exploratory Analysis
              ↓
       Feature Engineering
              ↓
      Demand Forecasting
              ↓
     Future Demand Prediction
              ↓
      Inventory Optimization
              ↓
       Reorder Recommendation
              ↓
       Streamlit Dashboard

       ## 📊 Sample Analysis

### Example: Store S001 – Product P0001

| Metric | Result |
|---|---:|
| Predicted Daily Demand | 128 units |
| Current Inventory | 223 units |
| Reorder Point | 1,275 units |
| Recommended Order | 1,052 units |

### Interpretation

For this store-product combination, the model predicts approximately **128 units of daily demand**.

Based on the defined inventory assumptions, the calculated reorder point is **1,275 units**. Since the current inventory of **223 units** is below the reorder point, the system recommends ordering approximately **1,052 units**.

### Key Analysis

- **Inventory vs Sales correlation:** ~0.59
- The analysis shows a positive relationship between inventory levels and units sold. However, correlation does not imply causation.
- **Model MAE:** 68.99
- **Model RMSE:** 88.26
- **Model MAPE:** 250.47%
- MAPE is high because percentage error can become unstable when actual sales values are small. Therefore, MAE and RMSE are more useful metrics for interpreting this model's error.

### Visual Analysis

The project includes the following visual analyses:

- **Monthly Sales Trend** — `monthly_sales.png`
- **Inventory vs Sales** — `inventory_vs_sales.png`
- **Actual vs Predicted Demand** — `actual_vs_predicted.png`

The overall workflow is:

**Historical Data → Data Analysis → ML Prediction → Inventory Optimization → Reorder Recommendation**