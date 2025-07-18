# scripts/prophet_predict.py

import os
import json
import requests
from datetime import datetime
import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, firestore

# --- Full Firebase credential dict from environment variables ---

firebase_creds = {
    "type": "service_account",
    "project_id": os.environ["FIREBASE_PROJECT_ID"],
    "private_key": os.environ["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
    "client_email": os.environ["FIREBASE_CLIENT_EMAIL"],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
}

cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Fetch IRCC data ---
URL = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
response = requests.get(URL)
data = response.json()["RoundsOfInvitation"]

# --- Prepare DataFrame ---
df = pd.DataFrame([{
    "ds": d["drawDate"],
    "y": int(d["score"])
} for d in data if d.get("score") and d.get("drawDate")])

df["ds"] = pd.to_datetime(df["ds"])

# --- Prophet Model ---
model = Prophet()
model.fit(df)

future_date = df["ds"].max() + pd.Timedelta(days=14)
future = pd.DataFrame({"ds": [future_date]})
forecast = model.predict(future).iloc[0]

# --- Prediction data ---
prediction = {
    "predictedDrawDate": forecast["ds"].isoformat(),
    "crs_yhat": round(forecast["yhat"]),
    "crs_yhat_upper": round(forecast["yhat_upper"]),
    "crs_yhat_lower": round(forecast["yhat_lower"]),
    "trend": round(forecast["trend"], 2),
    "seasonal": round(forecast["seasonal"], 2),
    "seasonal_lower": round(forecast["seasonal_lower"], 2),
    "seasonal_upper": round(forecast["seasonal_upper"], 2),
    "modelUpdatedAt": datetime.utcnow().isoformat()
}

db.collection("predictions").document("nextDraw").set(prediction)
print("✅ Prediction uploaded to Firestore:", prediction)
