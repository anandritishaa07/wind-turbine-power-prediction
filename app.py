import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Page Setup ---
st.set_page_config(page_title="Wind Turbine Predictive Analytics", layout="wide")
st.title("Wind Turbine Power Prediction & Underperformance Tracking")
st.markdown(
    "Supervised regression (Random Forest) for power output forecasting and "
    "unsupervised anomaly detection (Isolation Forest) for maintenance flagging."
)

# --- 1. Load Dataset ---
CSV_FILENAME = "T1.csv"
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILENAME)

if not os.path.exists(CSV_PATH):
    st.error(
        f"'{CSV_FILENAME}' not found. Place it in the same folder as app.py "
        f"({os.path.dirname(CSV_PATH)})."
    )
    st.stop()


@st.cache_data
def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = [
        "Timestamp",
        "ActivePower_kW",
        "WindSpeed_ms",
        "TheoreticalPower_kWh",
        "WindDirection_deg",
    ]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d %m %Y %H:%M")
    df["Hour"] = df["Timestamp"].dt.hour
    df["Month"] = df["Timestamp"].dt.month
    return df


df = load_and_clean_data(CSV_PATH)

FEATURES = ["WindSpeed_ms", "WindDirection_deg", "Hour", "Month"]

# --- Sidebar Controls ---
st.sidebar.header("Model Hyperparameters")
n_estimators = st.sidebar.slider("Random Forest Trees", 10, 100, 50, step=10)
contamination_rate = st.sidebar.slider(
    "Anomaly Contamination Rate", 0.01, 0.10, 0.03, step=0.01
)


# --- 2. Supervised Model (Power Prediction) ---
@st.cache_resource
def train_prediction_model(data, trees):
    X = data[FEATURES]
    y = data["ActivePower_kW"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=trees, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return model, rmse, r2


with st.spinner("Training predictive model..."):
    rf_model, model_rmse, model_r2 = train_prediction_model(df, n_estimators)


# --- 3. Unsupervised Model (Anomaly Detection) ---
@st.cache_data
def detect_anomalies(data, rate):
    data = data.copy()
    data["PowerDeficit"] = data["TheoreticalPower_kWh"] - data["ActivePower_kW"]

    iso_features = data[["WindSpeed_ms", "ActivePower_kW", "PowerDeficit"]]
    iso_forest = IsolationForest(contamination=rate, random_state=42, n_jobs=-1)

    # -1 = anomaly, 1 = normal
    data["Anomaly_Status"] = iso_forest.fit_predict(iso_features)
    return data


df = detect_anomalies(df, contamination_rate)

# --- 4. Dashboard Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Records", f"{len(df):,}")
col2.metric("Model R² Score", f"{model_r2:.2%}")
col3.metric("Prediction Error (RMSE)", f"{model_rmse:.2f} kW")

anomalies_count = int((df["Anomaly_Status"] == -1).sum())
col4.metric(
    "Flagged Underperforming Intervals",
    f"{anomalies_count:,}",
    delta=f"Rate: {contamination_rate:.0%}",
    delta_color="inverse",
)

st.markdown("---")

# --- 5. Visualizations ---
st.subheader("Power Curve & Anomaly Map")
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.markdown("**Theoretical vs. Predicted Power Curve**")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    sample_df = df.sample(n=min(2000, len(df)), random_state=42).sort_values(
        by="WindSpeed_ms"
    )

    # Smooth ML curve: vary wind speed only, hold other inputs at typical values
    ws_grid = np.linspace(0, df["WindSpeed_ms"].max(), 200)
    grid = pd.DataFrame(
        {
            "WindSpeed_ms": ws_grid,
            "WindDirection_deg": df["WindDirection_deg"].median(),
            "Hour": 12,
            "Month": 6,
        }
    )[FEATURES]
    grid_pred = rf_model.predict(grid)

    ax1.scatter(
        sample_df["WindSpeed_ms"],
        sample_df["ActivePower_kW"],
        color="lightgray",
        alpha=0.5,
        label="Actual SCADA Output",
    )
    ax1.plot(
        sample_df["WindSpeed_ms"],
        sample_df["TheoreticalPower_kWh"],
        color="darkblue",
        linewidth=2.5,
        label="OEM Theoretical Curve",
    )
    ax1.plot(
        ws_grid,
        grid_pred,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="ML Predicted Curve",
    )

    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.set_ylabel("Power Output (kW)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

with col_plot2:
    st.markdown("**Predictive Maintenance: Isolation Forest Flags**")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    normal_all = df[df["Anomaly_Status"] == 1]
    normal_ops = normal_all.sample(n=min(2000, len(normal_all)), random_state=42)
    anomalies_ops = df[df["Anomaly_Status"] == -1]

    ax2.scatter(
        normal_ops["WindSpeed_ms"],
        normal_ops["ActivePower_kW"],
        color="seagreen",
        alpha=0.4,
        label="Normal Operation",
    )
    ax2.scatter(
        anomalies_ops["WindSpeed_ms"],
        anomalies_ops["ActivePower_kW"],
        color="orangered",
        alpha=0.8,
        label="Underperformance",
    )

    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.set_ylabel("Power Output (kW)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# --- 6. Live Prediction Tool ---
st.markdown("---")
st.subheader("Live Turbine Prediction")
st.write("Enter current conditions to get the predicted power output:")

input_col1, input_col2, input_col3, input_col4 = st.columns(4)
ws_in = input_col1.number_input(
    "Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=12.0
)
wd_in = input_col2.number_input(
    "Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0
)
hr_in = input_col3.slider("Hour of Day", 0, 23, 12)
mo_in = input_col4.slider("Month of Year", 1, 12, 6)

simulated_features = pd.DataFrame(
    [[ws_in, wd_in, hr_in, mo_in]], columns=FEATURES
)
simulated_prediction = rf_model.predict(simulated_features)[0]

st.info(f"Predicted power output: **{simulated_prediction:.2f} kW**")