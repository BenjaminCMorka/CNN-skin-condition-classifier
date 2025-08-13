import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 3 convolutional blocks -> flatten -> FC layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = 2) -> nn.Module:
    """
    Returns the custom CNN model for acne/rosacea vs eczema classification.
    """
    return SimpleCNN(num_classes=num_classes)


def save_checkpoint(path: str, model: nn.Module, class_names):
    """
    Saves model weights and class names.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
        },
        path,
    )


def load_model(path: str) -> tuple[nn.Module, list]:
    """
    Loads model from a checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    class_names = checkpoint.get("class_names", ["acne_rosacea", "eczema"])
    model = get_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names
