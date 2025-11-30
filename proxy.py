from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client
import requests

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to your Hugging Face Space
client = Client("Ym420/promoter-classification-space")  # public space, no token needed

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "Proxy server running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        # --- Debug: print exact sequence received ---
        print("✅ Received sequence:", repr(req.sequence))

        # Call the HF Space API endpoint
        result = client.predict(
            sequence=req.sequence,
            api_name="/predict_promoter"  # note the leading slash
        )

        print("✅ Raw result from HF:", result)

        raw_label = result[0]
        #confidence = result[1]
        #confidence = float(result[1]) if isinstance(result[1], (int, float, str)) else 0.0

        # Expecting tuple like ("Promoter", 0.92)
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            label = str(result[0])
            confidence = float(result[1])
        else:
            label = "error"
            confidence = 0.0

    """    
    # --- Step 2.4: Send sequence to Google Sheet ---
        google_sheet_webapp_url = "https://script.google.com/macros/s/AKfycbxlilTmFl8MrW7T057oN3tWVYCf3T5iVXgORqCj2G4ub8GO-IfPVWeQFX613MCoXyTx/exec"
        payload = {
            "sequence": req.sequence
        }
        headers = {"Content-Type": "application/json"}
            try:
                r = requests.post(google_sheet_webapp_url, json=payload, headers=headers, timeout=5)
                sheet_status = r.text
                print("✅ Sheet response:", sheet_status)

            except Exception as sheet_err:
                sheet_status = f"Failed to save to Google Sheet: {sheet_err}"
                print("❌", sheet_status)
    """

        # Debug logs for Vercel
        print("Sequence  :", req.sequence)
        print("Prediction:", label)
        print("Confidence:", confidence)
        print("-----------------------")

        return {
            "sequence": req.sequence,
            "prediction": label,
            "confidence": float(confidence)
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "sequence": req.sequence,
            "prediction": "error",
            "confidence": 0.0,
            "error": str(e)
        }
