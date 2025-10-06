# proxy.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

# Enable CORS so your Swift app can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, replace * with your app's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class SequenceRequest(BaseModel):
    sequence: str

# Prediction endpoint
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        # Call the Hugging Face Space API directly
        hf_url = "https://Ym420-promoter-finder-space.hf.space/api/predict/"
        response = requests.post(hf_url, json={"data": [req.sequence]}, timeout=20)

        data = response.json()
        output = data["data"][0]  # [prediction, confidence]

        return {
            "sequence": req.sequence,
            "prediction": output[0],
            "confidence": output[1]
        }
    except Exception as e:
        return {"error": str(e)}
