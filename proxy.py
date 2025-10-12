rom fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to your Hugging Face Space
client = Client("Ym420/promoter-classification-space")  # no token needed for public spaces

class SequenceRequest(BaseModel):
    sequence: str

@app.get("/")
def root():
    return {"message": "Proxy server running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    try:
        # Call the Gradio function by name
        result = client.predict(
    	req.sequence,
    	fn_name="predict_promoter"  # matches the function name in app.py
	)

        
        raw_label = result[0]
        confidence = result[1]

        # Map to human-readable label
        if raw_label.lower() == "promoter":
            display_label = "σ⁷⁰ promoter"
        elif raw_label.lower() == "non-promoter":
            display_label = "no σ⁷⁰ consensus motifs"
        else:
            display_label = raw_label  # fallback
        
        # Debug logs for Vercel
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

