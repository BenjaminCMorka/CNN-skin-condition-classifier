from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
import torch.nn.functional as F
from model import CNN
from utils import preprocess_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = CNN()
model_path = "quantized_model.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
torch.backends.quantized.engine = 'qnnpack'

model = torch.jit.load(model_path)
model.eval()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            confidence = confidence.item()

        labels = {0: "Acne and Rosacea", 1: "Eczema"}

        if confidence < 0.7:
            prediction = "None of the Above"
        else:
            prediction = labels.get(int(predicted.item()), "None of the Above")

        return {"prediction": prediction, "confidence": round(confidence, 2)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
