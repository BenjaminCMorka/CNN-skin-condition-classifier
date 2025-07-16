from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
from model import CNN
from utils import preprocess_image
from download_model import download_model_from_s3

app = FastAPI()

# Enable CORS for your frontend domain (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model if not present, then load it
model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")

# Ensure model is downloaded before loading
download_model_from_s3()

# Load model once at startup
model = CNN()
model.load_state_dict(torch.load(model_path))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess_image(image).unsqueeze(0)  
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    labels = {0: "Acne and Rosacea", 1: "Eczema"}
    prediction = labels[int(predicted.item())]
    
    return {"prediction": prediction}
