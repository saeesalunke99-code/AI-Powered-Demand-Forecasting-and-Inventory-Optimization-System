import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📦 Inventory AI")

st.sidebar.markdown(
    """
    ### Navigation

    Use this dashboard to:

    🔮 Forecast product demand

    📦 Analyze inventory levels

    🔔 Identify reorder requirements

    📊 Review inventory recommendations
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning Model: Random Forest"
)

st.sidebar.caption(
    "AI-Powered Demand Forecasting and Inventory Optimization System"
)

# ==========================================
# TITLE
# ==========================================

st.title("📦 AI-Powered Demand Forecasting")
st.subheader("Inventory Optimization System")

st.write(
    """
    This system uses historical sales data and machine learning
    to forecast demand and generate inventory recommendations.
    """
)

# ==========================================
# LOAD INVENTORY DATA
# ==========================================

@st.cache_data
def load_inventory_data():

    df = pd.read_csv(
        "data/inventory_optimization_results.csv"
    )

    return df


df = load_inventory_data()

# ==========================================
# DASHBOARD METRICS
# ==========================================

total_products = len(df)

reorder_products = (
    df["Status"] == "REORDER"
).sum()

sufficient_products = (
    df["Status"] == "SUFFICIENT"
).sum()

total_order_quantity = (
    df["Recommended_Order"].sum()
)

# ==========================================
# DISPLAY METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Store-Product Combinations",
        total_products
    )

with col2:
    st.metric(
        "Products Requiring Reorder",
        reorder_products
    )

with col3:
    st.metric(
        "Sufficient Inventory",
        sufficient_products
    )

with col4:
    st.metric(
        "Recommended Order Units",
        int(total_order_quantity)
    )

# ==========================================
# INVENTORY TABLE
# ==========================================

st.subheader("📊 Inventory Recommendations")

st.dataframe(
    df,
    use_container_width=True
)

# ==========================================
# DEMAND FORECASTING
# ==========================================

st.subheader("🤖 Demand Forecasting")

st.write(
    "Select a store and product to generate a demand prediction "
    "using the trained Random Forest model."
)

# Load feature-engineered data
@st.cache_data
def load_forecasting_data():

    data = pd.read_csv(
        "data/feature_engineered_data.csv"
    )

    data["Date"] = pd.to_datetime(data["Date"])

    return data


forecast_df = load_forecasting_data()

# Load trained model
model = joblib.load(
    "models/demand_forecasting_model.pkl"
)

# Store selection
store = st.selectbox(
    "Select Store",
    sorted(forecast_df["Store ID"].unique())
)

# Product selection
product = st.selectbox(
    "Select Product",
    sorted(forecast_df["Product ID"].unique())
)

# Filter selected product
selected_data = forecast_df[
    (forecast_df["Store ID"] == store) &
    (forecast_df["Product ID"] == product)
]

if len(selected_data) > 0:

    # Get latest available record
    latest = selected_data.iloc[-1]

    st.write("### 📋 Latest Product Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Inventory",
            int(latest["Inventory Level"])
        )

    with col2:
        st.metric(
            "Last Units Sold",
            int(latest["Units Sold"])
        )

    with col3:
        st.metric(
            "Price",
            f"₹{latest['Price']:.2f}"
        )

    with col4:
        st.metric(
            "Discount",
            f"{latest['Discount']}%"
        )

    # ==========================================
    # MODEL INPUT FEATURES
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

    input_data = pd.DataFrame([[
        latest[feature]
        for feature in features
    ]], columns=features)

    # ==========================================
    # PREDICTION
    # ==========================================

    if st.button("🔮 Predict Demand"):

        prediction = model.predict(input_data)[0]

        st.success(
            f"### Predicted Demand: {prediction:.0f} units"
        )

        st.info(
            "This prediction is generated using the trained "
            "Random Forest demand forecasting model."
        )

        # ==========================================
        # INVENTORY OPTIMIZATION
        # ==========================================

        st.subheader("📦 Inventory Optimization")

        # Assumptions
        lead_time_days = 7
        safety_stock_days = 3

        # Calculate reorder point using predicted demand
        daily_demand = prediction

        reorder_point = (
            daily_demand * lead_time_days
            + daily_demand * safety_stock_days
        )

        current_inventory = latest["Inventory Level"]

        recommended_order = max(
            reorder_point - current_inventory,
            0
        )

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Reorder Point",
                f"{reorder_point:.0f} units"
            )

        with col2:
            st.metric(
                "Current Inventory",
                f"{current_inventory:.0f} units"
            )

        with col3:
            st.metric(
                "Recommended Order",
                f"{recommended_order:.0f} units"
            )

        # Inventory status
        if current_inventory <= reorder_point:

            st.warning(
                "⚠️ Inventory is below the recommended "
                "reorder point. Replenishment is recommended."
            )

        else:

            st.success(
                "✅ Current inventory is sufficient "
                "based on predicted demand."
            )

        st.caption(
            "Assumption: 7-day supplier lead time and "
            "3 days of safety-stock coverage."
        )

        # ==========================================
        # DEMAND VS INVENTORY VISUALIZATION
        # ==========================================

        st.subheader("📈 Demand vs Inventory")

        chart_data = pd.DataFrame({
            "Metric": [
                "Predicted Daily Demand",
                "Current Inventory",
                "Reorder Point"
            ],
            "Units": [
                prediction,
                current_inventory,
                reorder_point
            ]
        })

        st.bar_chart(
            chart_data.set_index("Metric")
        )

# ==========================================
# INVENTORY RECOMMENDATION TABLE
# ==========================================

st.subheader("📋 Inventory Recommendation Overview")

# Load optimization results
inventory_data = pd.read_csv(
    "data/inventory_optimization_results.csv"
)

# Display only important columns
display_columns = [
    "Store ID",
    "Product ID",
    "Average_Daily_Demand",
    "Safety_Stock",
    "Reorder_Point",
    "Inventory Level",
    "Recommended_Order",
    "Status"
]

st.dataframe(
    inventory_data[display_columns],
    use_container_width=True
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Powered Demand Forecasting and Inventory Optimization System"
)

# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("📊 Model Performance")

st.write(
    "The Random Forest model was evaluated using "
    "Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), "
    "and Mean Absolute Percentage Error (MAPE)."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        "68.99"
    )

with col2:
    st.metric(
        "RMSE",
        "88.26"
    )

with col3:
    st.metric(
        "MAPE",
        "250.47%"
    )

st.info(
    "Note: MAPE can become very large when actual sales values "
    "are small. Therefore, MAE and RMSE are also considered when "
    "evaluating the model."
)
# ==========================================
# DATA ANALYSIS VISUALIZATIONS
# ==========================================

st.subheader("📈 Data Analysis")

col1, col2 = st.columns(2)

with col1:

    st.write("### Monthly Sales Trend")

    st.image(
        "monthly_sales.png",
        use_container_width=True
    )

with col2:

    st.write("### Inventory vs Sales")

    st.image(
        "inventory_vs_sales.png",
        use_container_width=True
    )

    # ==========================================
# MODEL PREDICTION VISUALIZATION
# ==========================================

st.subheader("🤖 Actual vs Predicted Demand")

st.write(
    "Comparison between actual demand and demand predicted "
    "by the Random Forest model on the test dataset."
)

st.image(
    "actual_vs_predicted.png",
    use_container_width=True
)
# ==========================================
# DOWNLOAD INVENTORY RECOMMENDATIONS
# ==========================================

st.subheader("📥 Download Inventory Recommendations")

csv_data = inventory_data.to_csv(index=False)

st.download_button(
    label="⬇️ Download Inventory Recommendations",
    data=csv_data,
    file_name="inventory_recommendations.csv",
    mime="text/csv"
)

# ==========================================
# ABOUT THE PROJECT
# ==========================================

st.subheader("ℹ️ About This Project")

st.write(
    """
    The AI-Powered Demand Forecasting and Inventory Optimization System
    uses historical retail sales and inventory data to estimate future
    product demand and support inventory replenishment decisions.
    """
)

st.markdown(
    """
    **Key Components:**

    - 📊 Historical sales and inventory analysis
    - 🔧 Feature engineering
    - 🤖 Random Forest demand forecasting
    - 📦 Inventory optimization
    - 🔔 Reorder recommendations
    - 📈 Data visualizations
    - 📥 Downloadable inventory recommendations

    **Technology Stack:**

    Python • Pandas • NumPy • Scikit-learn • Matplotlib • Streamlit
    """
)

st.info(
    "Inventory calculations use assumed parameters of a "
    "7-day supplier lead time and 3 days of safety-stock coverage."
)