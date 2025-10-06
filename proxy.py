from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

print("✅ Proxy starting up...")   # Add this

app = FastAPI()

@app.on_event("startup")
def startup_event():
    print("🚀 FastAPI app started successfully!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceRequest(BaseModel):
    sequence: str

@app.post("/predict")
def predict(req: SequenceRequest):
    print(f"🧬 Received sequence: {req.sequence[:10]}...")   # Log input
    try:
        hf_url = "https://Ym420-promoter-finder-space.hf.space/api/predict/"
        response = requests.post(hf_url, json={"data": [req.sequence]}, timeout=20)
        data = response.json()
        output = data["data"][0]
        print(f"✅ Prediction done: {output}")
        return {
            "sequence": req.sequence,
            "prediction": output[0],
            "confidence": output[1]
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}

