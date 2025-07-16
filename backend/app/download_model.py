import os
import boto3
import sys
from botocore.exceptions import ClientError, NoCredentialsError

S3_BUCKET = "dermnet-model"
S3_KEY = "quantized_model.pt"
MODEL_PATH = "quantized_model.pt"

def download_model():
    s3_bucket = S3_BUCKET
    s3_key = S3_KEY
    model_path = MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"{model_path} already exists, skipping download.")
        return
    
    s3 = boto3.client('s3')
    print(f"Downloading model from s3://{s3_bucket}/{s3_key} ...")
    s3.download_file(s3_bucket, s3_key, model_path)
    print("Download complete.")

if __name__ == "__main__":
    download_model()
