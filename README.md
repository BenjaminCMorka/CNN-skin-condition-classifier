# Skin Condition Classifier

Skin Condition Classifier is a deep learning project that uses convolutional neural networks (CNNs) to classify images of skin conditions, focusing on distinguishing **Acne and Rosacea** from **Eczema**. The project utilizes the **DermNet** dataset and provides an end-to-end pipeline for data preparation, model training, and inference.

---

## Project Overview

This project aims to develop an accurate and robust image classifier to assist dermatology diagnostics by automatically categorizing and labelling skin condistion images. Leveraging a CNN architecture trained on curated subsets of the DermNet dataset, the model learns to recognize visual patterns unique to different skin conditions.

---

## Dataset

- **Source:** [DermNet Dataset by shubhamgoel27 on Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)
- **Subset Used:** Only two categories retained:
  - *Acne and Rosacea Photos*  
  - *Eczema Photos*
- **Data Split:** Organized into `train` and `test` folders under `data/processed/`
- Images are preprocessed and filtered for quality and consistency.

---

## Model

- Implemented in **PyTorch**
- CNN architecture customized for skin lesion image classification
- Trained with data augmentation for improved generalization
- Model weights saved as `best_model.pth`
- Achieved 80% classification accuracy on the test set distinguishing Acne and Rosacea from Eczema

---

## Code Structure

```bash
src/
├── download_dataset.py     # Download and prepare dataset filtered by classes
├── dataset.py              # Custom PyTorch Dataset for loading images
├── model.py                # CNN model architecture definition
├── train.py                # Training and validation pipeline
├── augmentation.py         # Image augmentation utilities

```

---

## Setup instructions

Clone the Repository:
git clone https://github.com/BenjaminCMorka/CNN-skin-condition-classifier.git
cd skin-condition-classifier/backend/app

Install Python Backend Dependencies:
pip install -r requirements.txt

Download Model Weights
Python download_model.py

Download and Prepare Dataset
cd ../../src
python download_dataset.py

## Running the Web Application

Backend (FastAPI):
from the root directory, enter command uvicorn main:app --reload

Frontend (React & Next.js):
from the root directory, enter commands:
cd frontend
npm install
npm run dev

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white) 
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white) 
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) 
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)


