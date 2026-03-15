import os
import json
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

# -----------------------------------
# 1. FastAPI app
# -----------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# 2. Environment variables
# -----------------------------------
PROMOTER_SPACE = os.getenv("PROMOTER_SPACE")
SHEET_URL = os.getenv("SHEET_URL_Promoter")
SECRET_TOKEN = os.getenv("SECRET_TOKEN_Promoter")

print("PROMOTER_SPACE:", PROMOTER_SPACE)
print("SHEET_URL:", SHEET_URL)

# -----------------------------------
# 3. Connect to HuggingFace Space
# -----------------------------------
client = Client(PROMOTER_SPACE)

# -----------------------------------
# 4. Request model
# -----------------------------------
class SequenceRequest(BaseModel):
    sequence: str


# -----------------------------------
# 5. Health check
# -----------------------------------
@app.get("/")
def health():
    return {"status": "proxy running"}


# -----------------------------------
# 6. Prediction endpoint
# -----------------------------------
@app.post("/predict")
def predict(req: SequenceRequest):

    try:

        sequence = req.sequence.strip()

        print("Received sequence:", sequence)

        # -----------------------------------
        # Call Hugging Face Space
        # -----------------------------------
        result = client.predict(
            sequence=sequence,
            api_name="/predict_promoter"
        )

        print("HF raw result:", result)

        # -----------------------------------
        # Unwrap gradio result
        # -----------------------------------
        while isinstance(result, (list, tuple)) and len(result) == 1:
            result = result[0]

        if isinstance(result, (list, tuple)) and len(result) == 2:
            label = str(result[0])
            confidence = float(result[1])
        else:
            raise ValueError(f"Unexpected HF output: {result}")

        print("Prediction:", label)
        print("Confidence:", confidence)

        # -----------------------------------
        # Send to Google Sheet
        # -----------------------------------
        payload = {
            "sequence": sequence,
            "prediction": label,
            "confidence": confidence,
            "secret_token": SECRET_TOKEN
        }

        try:

            r = requests.post(
                SHEET_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "text/plain"},
                timeout=10
            )

            print("Sheet status:", r.status_code)
            print("Sheet response:", r.text)

        except Exception as sheet_error:
            print("Google sheet error:", sheet_error)

        # -----------------------------------
        # Return to iOS
        # -----------------------------------
        return {
            "sequence": sequence,
            "prediction": label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "sequence": req.sequence,
            "prediction": "error",
            "confidence": 0.0,
            "error": str(e)
        }


