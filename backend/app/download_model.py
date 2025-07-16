import os
import requests
import sys

MODEL_URL = "https://github.com/BenjaminCMorka/skin-disease-classifier/releases/download/v1.0/best_model.pth"
MODEL_PATH = "best_model.pth" 

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading model from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()

     
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:  
            os.makedirs(model_dir, exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Model downloaded and saved to {MODEL_PATH}")
        
     
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            print(f"Model file verified: {os.path.getsize(MODEL_PATH)} bytes")
        else:
            print("ERROR: Model file was not downloaded correctly")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()