from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
from model import CNN
from utils import preprocess_image
from download_model import download_model

app = FastAPI()

# Enable CORS for your frontend domain (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
model = None
model_loaded = False

def load_model():
    global model, model_loaded
    if not model_loaded:
        # Download model if it doesn't exist
        download_model()
        
        # Load model
        model = CNN()
        model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        model_loaded = True
    return model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load model on first request
    current_model = load_model()
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess_image(image).unsqueeze(0)  
    
    with torch.no_grad():
        outputs = current_model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    labels = {0: "Acne and Rosacea", 1: "Eczema"}
    prediction = labels[int(predicted.item())]
    
    return {"prediction": prediction}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
