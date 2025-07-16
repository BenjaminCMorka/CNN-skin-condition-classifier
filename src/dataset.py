import os
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
train_dir = os.path.join(DATA_DIR, 'train')

class DermNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): dir with all the images organized by class subfolders
            transform (callable, optional): transform to be applied on an image sample
        """
        self.root_dir = root_dir
        self.transform = transform

        # get all image file paths and their labels as in folder names
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        self.classes = [cls for cls in self.classes if os.path.isdir(os.path.join(root_dir, cls))]
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # to test
    from augmentation import get_enhanced_transforms
    
    # quick test
    dataset = DermNetDataset(root_dir=train_dir, transform=get_enhanced_transforms(mode='val'))
    print(f"Number of samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class to index mapping: {dataset.class_to_idx}")
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Image tensor shape: {img.shape}, Label: {label}")