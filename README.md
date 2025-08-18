# Skin Condition Classifier

Skin Condition Classifier is a deep learning project that uses convolutional neural networks (CNNs) to classify images of skin conditions, focusing on distinguishing **Acne** from **Eczema**. The project utilizes the **DermNet** dataset and provides an end-to-end pipeline for data preparation, model training, and inference.

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
- Achieved 80% classification accuracy on the test set distinguishing Acne from Eczema

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

## Running the Web Application

The Web Application can be found at: [https://skin-condition-classifier.vercel.app](https://skin-condition-classifier.vercel.app)

---

## Demo Video

A demo video running through the application can be found here: [https://youtu.be/-szI7czoZAc](https://youtu.be/-szI7czoZAc)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB) ![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
