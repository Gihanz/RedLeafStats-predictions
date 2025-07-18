import os
import json
import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, firestore

# ✅ Setup Firebase Admin
firebase_credentials = {
"type": "service_account",
    "project_id": os.environ["FIREBASE_PROJECT_ID"],
    "private_key": os.environ["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),

    "client_email": os.environ["FIREBASE_CLIENT_EMAIL"],

    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
}

cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ✅ Fetch data from Firestore (e.g. ee_rounds)
docs = db.collection("ee_rounds").order_by("drawDate").stream()

data = []
for doc in docs:
    d = doc.to_dict()
    if "drawDate" in d and "drawCRS" in d:
        data.append({
            "drawDate": d["drawDate"],
            "drawCRS": d["drawCRS"]
        })

# ✅ Ensure we have data
if not data:
    print("⚠️ No data returned from Firestore. Exiting.")
    exit(1)

df = pd.DataFrame(data)

# ✅ Ensure correct columns exist
expected_cols = {"drawDate", "drawCRS"}
if not expected_cols.issubset(df.columns):
    print(f"❌ Missing expected columns in Firestore data: {expected_cols - set(df.columns)}")
    exit(1)

# ✅ Rename for Prophet
df = df.rename(columns={"drawDate": "ds", "drawCRS": "y"})
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds")

# ✅ Fit Prophet model
model = Prophet()
model.fit(df)

# ✅ Create future dataframe (e.g., predict next 10 draws)
future = model.make_future_dataframe(periods=10, freq='W')
forecast = model.predict(future)

# ✅ Display or upload the forecast
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

# ✅ Optional: upload predictions to Firestore
for i, row in forecast.tail(10).iterrows():
    doc = {
        "predictedDate": row["ds"].strftime("%Y-%m-%d"),
        "predictedCRS": int(row["yhat"]),
        "lowerBound": int(row["yhat_lower"]),
        "upperBound": int(row["yhat_upper"]),
        "source": "prophet"
    }
    db.collection("ee_forecasts").add(doc)

print("✅ Prophet forecast completed and uploaded.")
