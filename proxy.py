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
