# proxy.py
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your Swift app can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, replace * with your app's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to your Hugging Face Space
client = Client("Ym420/promoter-finder-space")

# Request model
class SequenceRequest(BaseModel):
    sequence: str

# Prediction endpoint
@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        # Call the Gradio Space predict function
        result = client.predict(req.sequence, api_name="/predict_promoter")
        return {"prediction": result[0], "confidence": result[1]}
    except Exception as e:
        return {"error": str(e)}
