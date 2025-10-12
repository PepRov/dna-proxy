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

client = Client("Ym420/promoter-finder", hf_token=None)

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
        raw_label = result["label"]
        confidence = result["confidence"]
        
        #raw_label = result[0]
        #confidence = result[1]

        # ✅ Map to human-readable label
        if raw_label.lower() == "promoter":
            display_label = "σ⁷⁰ promoter"
        elif raw_label.lower() == "non-promoter":
            display_label = "no σ⁷⁰ consensus motifs"
        else:
            display_label = raw_label  # fallback in case model returns something else
        
        # Optional: print for debugging in Vercel logs
        print("Sequence  :", req.sequence)
        print("Prediction:", display_label)
        print("Confidence:", confidence)
        print("-----------------------")
        
        return {
            "sequence": req.sequence,
            "prediction": display_label,
            "confidence": confidence
        }
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}
