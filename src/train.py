import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import DermNetDataset  
from model import CNN
from augmentation import get_enhanced_transforms  
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'test')

def get_sampler(labels):
    class_sample_count = np.bincount(labels)
    weights = 1. / class_sample_count
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct.double() / len(train_loader.dataset)

    return epoch_loss, epoch_acc.item()

def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct.double() / len(val_loader.dataset)

    return epoch_loss, epoch_acc.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Use enhanced transforms from augmentation.py
    train_transform = get_enhanced_transforms(mode='train', image_size=224)
    val_transform = get_enhanced_transforms(mode='val', image_size=224)

    train_dataset = DermNetDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = DermNetDataset(root_dir=val_dir, transform=val_transform)

    logging.info(f'Total training samples: {len(train_dataset)}')
    logging.info(f'Total validation samples: {len(val_dataset)}')

    logging.info(f'Training directory: {train_dir}')
    logging.info(f'Classes found: {train_dataset.classes}')
    logging.info(f'Class to index mapping: {train_dataset.class_to_idx}')
    logging.info(f'Unique labels in training: {set(train_dataset.labels)}')
    
    # Weighted sampler for balanced sampling
    sampler = get_sampler(train_dataset.labels)

    train_loader = DataLoader(
        train_dataset, batch_size=64, sampler=sampler, num_workers=8, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=8, persistent_workers=True
    )

    num_classes = len(train_dataset.classes)
    logging.info(f'Number of classes: {num_classes}')
    
    if num_classes == 0:
        logging.error("No classes found in the dataset. Please check your directory structure.")
        return
    
    logging.info(f"setting up {num_classes}-class classification problem")

    model = CNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    num_epochs = 10

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        epoch_duration = time.time() - start_time
        logging.info(f'Epoch {epoch+1}/{num_epochs} completed in {epoch_duration:.2f} seconds')
        logging.info(f'Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}')
        logging.info(f'Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info('Best model saved!')

if __name__ == '__main__':
    main()