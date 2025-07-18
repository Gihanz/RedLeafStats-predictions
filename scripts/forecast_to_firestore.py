import json
import requests
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Load Firebase credentials from environment
firebase_creds = {
    "type": "service_account",
    "project_id": os.environ["FIREBASE_PROJECT_ID"],
    "client_email": os.environ["FIREBASE_CLIENT_EMAIL"],
    "private_key": os.environ["FIREBASE_PRIVATE_KEY"].replace("\\n", "\n"),
}

cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load JSON from IRCC website
URL = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
response = requests.get(URL)
data = response.json()

# Extract relevant fields from latest draw
latest_draw = data["RoundsOfInvitation"][-1]
prediction = {
    "drawNumber": latest_draw["drawNumber"],
    "drawDate": latest_draw["drawDate"],
    "drawType": latest_draw["drawType"],
    "numberOfInvitationsIssued": latest_draw["numberOfInvitationsIssued"],
    "crsScore": latest_draw["score"],  # CRS cutoff
    "timestamp": datetime.utcnow().isoformat()
}

# Save to Firestore (e.g., predictions/nextDraw)
db.collection("predictions").document("nextDraw").set(prediction)
print("✅ Uploaded prediction to Firestore")
