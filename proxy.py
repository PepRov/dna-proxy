from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client("Ym420/promoter-finder-space", hf_token=None)

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "Proxy server running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        # Call via Gradio Client
        result = client.predict(req.sequence, api_name="/predict_promoter")
        
        # Optional: print for debugging in Vercel logs
        print("Sequence  :", req.sequence)
        print("Prediction:", result[0])
        print("Confidence:", result[1])
        print("-----------------------")
        
        return {
            "sequence": req.sequence,
            "prediction": result[0],
            "confidence": result[1]
        }
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
