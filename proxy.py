from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
import requests  # Added to allow POST to Google Sheets

# --- Step 0: Initialize FastAPI app ---

app = FastAPI()

# --- Step 0.1: Enable CORS so your iOS app can call the API ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 0.2: Connect to your Hugging Face Space ---
client = Client("Ym420/promoter-classification-space")  # public space, no token needed

# --- Step 0.3: Define input model from iOS app ---
class SequenceRequest(BaseModel):
    sequence: str

# --- Step 1: Simple health check endpoint ---
@app.get("/")
def root():
    return {"message": "Proxy server running"}

# --- Step 2: Main endpoint for sequence prediction ---
@app.post("/predict")
def predict(req: SequenceRequest):
    """
    Receives a DNA sequence from the iOS app,
    sends it to HF Space for promoter prediction,
    then logs sequence + prediction to Google Sheet via Apps Script.
    """
    try:
        # --- Step 2.1: Debug log the received sequence ---
        print("✅ Received sequence:", repr(req.sequence))

        # --- Step 2.2: Call HF Space API to predict promoter ---
        result = client.predict(
            sequence=req.sequence,
            api_name="/predict_promoter"
        )
        print("✅ Raw result from HF:", result)

        # --- Step 2.3: Parse prediction and confidence from HF result ---
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            label = str(result[0])            # Promoter label
            confidence = float(result[1])     # Prediction confidence
        else:
            label = "error"
            confidence = 0.0

        # --- Step 2.4: Prepare payload for Google Sheet ---
        google_sheet_webapp_url = "https://script.google.com/macros/s/AKfycbxlilTmFl8MrW7T057oN3tWVYCf3T5iVXgORqCj2G4ub8GO-IfPVWeQFX613MCoXyTx/exec"
        payload = {"sequence": req.sequence}
        headers = {"Content-Type": "application/json"}

        # --- Step 2.5: POST to Google Sheet ---
        try:
            r = requests.post(google_sheet_webapp_url, json=payload, headers=headers)
            print("✅ Sheet response:", r.text)
        except Exception as sheet_err:
            # Catch any error saving to sheet without stopping HF prediction
            print("❌ Failed to save to Google Sheet:", sheet_err)

        # --- Step 2.6: Debug log prediction info ---
        print("Sequence  :", req.sequence)
        print("Prediction:", label)
        print("Confidence:", confidence)
        print("-----------------------")

        # --- Step 2.7: Return prediction to iOS app ---
        return {
            "sequence": req.sequence,
            "prediction": label,
            "confidence": float(confidence)
        }

    except Exception as e:
        # --- Step 2.8: Catch-all error if anything goes wrong ---
        print("Error:", str(e))
        return {
            "sequence": req.sequence,
            "prediction": "error",
            "confidence": 0.0,
            "error": str(e)
        }


