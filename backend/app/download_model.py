import os
import requests

MODEL_URL = "https://github.com/BenjaminCMorka/skin-disease-classifier/releases/download/v1.0/best_model.pth"
MODEL_PATH = "best_model.pth" 

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Model downloaded and saved to {MODEL_PATH}")

if __name__ == "__main__":
    download_model()
