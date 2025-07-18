# scripts/prophet_predict.py

import json
import requests
import os
from datetime import datetime
import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, firestore

# --- Load Firebase credentials from GitHub secrets ---
firebase_creds = {
    "type": "service_account",
    "project_id": os.environ["FIREBASE_PROJECT_ID"],
    "private_key": os.environ["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
    "client_email": os.environ["FIREBASE_CLIENT_EMAIL"],
}

cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Fetch historical IRCC draw data ---
URL = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
response = requests.get(URL)
data = response.json()["RoundsOfInvitation"]

# --- Extract relevant fields for modeling ---
df = pd.DataFrame([{
    "ds": d["drawDate"],
    "y": int(d["score"])
} for d in data if d.get("score") and d.get("drawDate")])

# Convert drawDate to datetime
df["ds"] = pd.to_datetime(df["ds"])

# --- Prophet forecast ---
model = Prophet()
model.fit(df)

# Predict the next draw date (roughly every 14 days)
future_date = df["ds"].max() + pd.Timedelta(days=14)
future = pd.DataFrame({"ds": [future_date]})
forecast = model.predict(future).iloc[0]

# --- Prepare prediction data ---
prediction = {
    "predictedDrawDate": forecast["ds"].isoformat(),
    "crs_yhat": round(forecast["yhat"]),
    "crs_yhat_upper": round(forecast["yhat_upper"]),
    "crs_yhat_lower": round(forecast["yhat_lower"]),
    "modelUpdatedAt": datetime.utcnow().isoformat()
}

# --- Upload to Firestore ---
db.collection("predictions").document("nextDraw").set(prediction)
print("✅ Prediction uploaded to Firestore:", prediction)
