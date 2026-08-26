"""
CNC Predictive Maintenance - Web Dashboard API
Phase 7: Flask/FastAPI backend serving the trained model as a REST API.

Run with:
    uvicorn api:app --reload --port 8000
"""

import random
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Load trained model bundle (produced by cnc_predictive_maintenance.py)
# ─────────────────────────────────────────────
try:
    bundle = joblib.load('model_bundle.joblib')
except FileNotFoundError:
    raise RuntimeError(
        "model_bundle.joblib not found. Run cnc_predictive_maintenance.py "
        "first to train the model and save the bundle."
    )

model = bundle['model']
scaler = bundle['scaler']
feature_columns = bundle['feature_columns']
decision_threshold = bundle['decision_threshold']
THRESHOLDS = bundle['thresholds']

# Historical dataset, used only to simulate a live sensor stream for the demo
RAW_SENSOR_COLUMNS = [
    'Spindle_Speed_RPM', 'Feed_Rate_mm_per_min', 'Cutting_Depth_mm',
    'Vibration_X_mm_s', 'Vibration_Y_mm_s', 'Vibration_Z_mm_s',
    'Spindle_Temp_C', 'Coolant_Temp_C', 'Coolant_Flow_L_per_min',
    'Power_Consumption_kW', 'Tool_Wear_Percent', 'Acoustic_Emission_dB',
    'Servo_Load_X_Percent', 'Servo_Load_Y_Percent', 'Servo_Load_Z_Percent',
    'Surface_Roughness_Ra_um',
]

df_live = pd.read_excel('cncmilingdata2023-2026.xlsx', sheet_name='Sensor_Data')
df_live = df_live.sort_values('Timestamp').reset_index(drop=True)
for col in RAW_SENSOR_COLUMNS:
    df_live[col] = pd.to_numeric(df_live[col], errors='coerce')
df_live[RAW_SENSOR_COLUMNS] = df_live[RAW_SENSOR_COLUMNS].fillna(
    df_live[RAW_SENSOR_COLUMNS].mean()
)
df_live['Tool_Wear_Percent'] = df_live['Tool_Wear_Percent'].clip(0, 100)
df_live['Surface_Roughness_Ra_um'] = df_live['Surface_Roughness_Ra_um'].clip(lower=0)

app = FastAPI(title="CNC Predictive Maintenance API")

# In-memory alert history (most recent first), capped for the demo
alert_history = []
MAX_HISTORY = 200


class SensorReading(BaseModel):
    Spindle_Speed_RPM: float
    Feed_Rate_mm_per_min: float
    Cutting_Depth_mm: float
    Vibration_X_mm_s: float
    Vibration_Y_mm_s: float
    Vibration_Z_mm_s: float
    Spindle_Temp_C: float
    Coolant_Temp_C: float
    Coolant_Flow_L_per_min: float
    Power_Consumption_kW: float
    Tool_Wear_Percent: float
    Acoustic_Emission_dB: float
    Servo_Load_X_Percent: float
    Servo_Load_Y_Percent: float
    Servo_Load_Z_Percent: float
    Surface_Roughness_Ra_um: float


# ─────────────────────────────────────────────
# Shared prediction/alerting logic (mirrors cnc_predictive_maintenance.py)
# ─────────────────────────────────────────────

def check_alerts(reading: dict):
    alerts = []
    for sensor, threshold in THRESHOLDS.items():
        val = reading[sensor]
        if sensor == 'Coolant_Flow_L_per_min':
            if val < threshold:
                alerts.append(f"LOW {sensor}: {val:.2f} (threshold: >{threshold})")
        else:
            if val > threshold:
                alerts.append(f"HIGH {sensor}: {val:.2f} (threshold: <{threshold})")
    return alerts


def classify_failure_mode(reading: dict, probability: float):
    if reading['Spindle_Temp_C'] > 85 and reading['Coolant_Flow_L_per_min'] < 2:
        return "THERMAL OVERLOAD"
    elif reading['Tool_Wear_Percent'] > 80:
        return "TOOL WEAR FAILURE" if reading['Tool_Wear_Percent'] > 95 else "TOOL WEAR WARNING"
    elif reading['Coolant_Flow_L_per_min'] < 2:
        return "COOLANT SYSTEM FAILURE" if reading['Coolant_Flow_L_per_min'] < 1 else "COOLANT FLOW WARNING"
    elif (reading['Vibration_X_mm_s'] > 1.0 or reading['Vibration_Y_mm_s'] > 1.0
          or reading['Vibration_Z_mm_s'] > 1.0):
        return "VIBRATION ANOMALY"
    elif reading['Power_Consumption_kW'] > 7:
        return "SPINDLE LOAD ANOMALY"
    else:
        return "ANOMALOUS OPERATING PATTERN" if probability >= 0.70 else "NORMAL OPERATION"


def generate_maintenance_report(reading: dict, probability: float):
    causes, actions = [], []

    if reading['Spindle_Temp_C'] > 85:
        causes.append("High Spindle Temperature")
        actions.append("Inspect spindle cooling system")
    if reading['Coolant_Flow_L_per_min'] < 2:
        causes.append("Low Coolant Flow")
        actions.append("Check coolant pump and coolant lines")
    if reading['Tool_Wear_Percent'] > 80:
        causes.append("Excessive Tool Wear")
        actions.append("Replace cutting tool")
    if reading['Power_Consumption_kW'] > 7:
        causes.append("High Power Consumption")
        actions.append("Inspect spindle load conditions")
    if reading['Acoustic_Emission_dB'] > 90:
        causes.append("Abnormal Acoustic Emission")
        actions.append("Check for chatter and tool damage")
    if (reading['Vibration_X_mm_s'] > 1.0 or reading['Vibration_Y_mm_s'] > 1.0
            or reading['Vibration_Z_mm_s'] > 1.0):
        causes.append("Abnormal Machine Vibration")
        actions.append("Inspect bearings, spindle and machine alignment")

    if probability > 0.90:
        priority = "CRITICAL"
    elif probability > 0.70:
        priority = "HIGH"
    elif probability > 0.50:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return causes, actions, priority


def run_prediction(reading: dict):
    alerts = check_alerts(reading)

    input_df = pd.DataFrame([reading])
    input_df['Vibration_Magnitude'] = np.sqrt(
        input_df['Vibration_X_mm_s']**2
        + input_df['Vibration_Y_mm_s']**2
        + input_df['Vibration_Z_mm_s']**2
    )
    input_df['Temp_Difference'] = input_df['Spindle_Temp_C'] - input_df['Coolant_Temp_C']
    input_df['Efficiency_Ratio'] = input_df['Power_Consumption_kW'] / input_df['Spindle_Speed_RPM']
    input_df['Spindle_Temp_Rolling'] = input_df['Spindle_Temp_C']
    input_df['Vibration_Rolling'] = input_df['Vibration_Magnitude']
    input_df['Temp_Rate_Change'] = 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction = 1 if probability >= decision_threshold else 0

    causes, actions, priority = generate_maintenance_report(reading, probability)
    failure_mode = classify_failure_mode(reading, probability)

    early_warning = prediction == 0 and len(causes) > 0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sensor_reading": reading,
        "prediction": "DEFECT DETECTED" if prediction == 1 else "NORMAL",
        "defect_probability": round(probability * 100, 2),
        "priority": priority,
        "failure_mode": failure_mode,
        "root_causes": causes,
        "recommended_actions": actions,
        "threshold_alerts": alerts,
        "early_warning": early_warning,
    }

    alert_history.insert(0, result)
    del alert_history[MAX_HISTORY:]
    return result


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "decision_threshold": decision_threshold}


@app.post("/api/predict")
def predict(reading: SensorReading):
    return run_prediction(reading.model_dump())


@app.get("/api/simulate")
def simulate():
    """Draw one random historical row and run it through the live pipeline,
    simulating an incoming CNC sensor reading (mixed normal/defective)."""
    row = df_live.sample(1).iloc[0]
    reading = {col: float(row[col]) for col in RAW_SENSOR_COLUMNS}
    return run_prediction(reading)


@app.get("/api/history")
def history(limit: int = 50):
    return alert_history[:limit]


@app.delete("/api/history")
def clear_history():
    alert_history.clear()
    return {"status": "cleared"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
