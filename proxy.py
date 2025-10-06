from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "Proxy server running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        hf_url = "https://Ym420-promoter-finder-space.hf.space/api/predict/"
        resp = requests.post(hf_url, json={"data": [req.sequence]}, timeout=20)
        result = resp.json()
        output = result["data"][0]
        return {"sequence": req.sequence, "prediction": output[0], "confidence": output[1]}
    except Exception as e:
        return {"error": str(e)}
