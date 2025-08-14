import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class Augmentation:
    """augmentation class for skin images"""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, img):
        if random.random() < self.prob:
            return self.apply_augmentation(img)
        return img
    
    def apply_augmentation(self, img):
        return img

class GaussianNoise(Augmentation):
    """add gaussian noise to image"""
    
    def __init__(self, mean=0, std=0.1, prob=0.3):
        super().__init__(prob)
        self.mean = mean
        self.std = std
    
    def apply_augmentation(self, img):
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)
        
        noise = torch.randn(img.size()) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        
        return F.to_pil_image(noisy_img)

class GaussianBlur(Augmentation):
    """add gaussian blur to image"""
    
    def __init__(self, kernel_size=5, prob=0.3):
        super().__init__(prob)
        self.kernel_size = kernel_size
    
    def apply_augmentation(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.kernel_size))

def get_enhanced_transforms(mode='train', image_size=224):
    """
    get enhanced transforms for training and validation
    
    Args:
        mode: 'train' or 'val'
        image_size: target image size
    
    Returns:
        transforms.Compose object
    """
    
    if mode == 'train':
        
        train_transform = transforms.Compose([
            # geometric transformations
            transforms.Resize((image_size + 32, image_size + 32)),  
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  
            transforms.RandomRotation(degrees=30),
            
            # color augmentations
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.2),
            
            # custom augmentations
            GaussianBlur(kernel_size=3, prob=0.2),
            GaussianNoise(mean=0, std=0.05, prob=0.3),
            

            transforms.ToTensor(),
            
            # apply random erasing 
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            
            # normalize it
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return train_transform
    
    else:  # validation
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        return val_transform

def get_progressive_transforms(epoch, total_epochs, image_size=224):
    """
    progressive augmentation start with mild augmentations and increase intensity
    
    Args:
        epoch: current epoch
        total_epochs: total number of epochs
        image_size: target image size
    
    Returns:
        transforms.Compose object
    """
    
    # Calculate augmentation intensity based on epoch
    progress = epoch / total_epochs
    
    # Start with mild augmentations, gradually increase
    rotation_degrees = 15 + (30 * progress)  # 15 to 45 degrees
    brightness = 0.1 + (0.3 * progress)     # 0.1 to 0.4
    contrast = 0.1 + (0.3 * progress)       # 0.1 to 0.4
    
    transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2 + 0.1 * progress),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=0.1 + (0.2 * progress),
            hue=0.05 + (0.1 * progress)
        ),
        GaussianBlur(kernel_size=3, prob=0.1 + 0.2 * progress),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1 + 0.2 * progress),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    return transform

def get_test_time_augmentation_transforms(image_size=224, num_augmentations=5):
    """
    create multiple augmented versions for test-time augmentation
    
    Args:
        image_size: target image size
        num_augmentations: number of augmented versions to create
    
    Returns:
        list of transforms
    """
    
    transforms_list = []
    
    # Original (no augmentation)
    transforms_list.append(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Augmented versions
    for i in range(num_augmentations - 1):
        augmented_transform = transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transforms_list.append(augmented_transform)
    
    return transforms_list

# test-time augmentation for inference
def predict_with_tta(model, image, device, num_augmentations=5):
    """
    predict using test-time augmentation
    """
    model.eval()
    transforms_list = get_test_time_augmentation_transforms(num_augmentations=num_augmentations)
    
    predictions = []
    
    with torch.no_grad():
        for transform in transforms_list:
            augmented_image = transform(image).unsqueeze(0).to(device)
            output = model(augmented_image)
            predictions.append(torch.softmax(output, dim=1))
    
    # avg predictions
    final_prediction = torch.mean(torch.stack(predictions), dim=0)
    
    return final_prediction