import os
import boto3
import sys
from botocore.exceptions import ClientError, NoCredentialsError

S3_BUCKET = "dermnet-model"
S3_KEY = "best_model.pth"
MODEL_PATH = "best_model.pth"

def download_model_from_s3():
    """Download model from S3 bucket"""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading model from S3: s3://{S3_BUCKET}/{S3_KEY}")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Create directory if needed
        model_dir = os.path.dirname(MODEL_PATH)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        
        # Download from S3
        s3_client.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)
        
        print(f"Model downloaded and saved to {MODEL_PATH}")
        
        # Verify download
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            print(f"Model file verified: {os.path.getsize(MODEL_PATH)} bytes")
        else:
            print("ERROR: Model file was not downloaded correctly")
            sys.exit(1)
            
    except NoCredentialsError:
        print("ERROR: AWS credentials not found")
        sys.exit(1)
    except ClientError as e:
        print(f"ERROR: S3 client error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model_from_s3()
