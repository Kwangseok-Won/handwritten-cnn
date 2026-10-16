import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.cnn(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    "models/alphanum_cnn.pth",
    map_location=device,
    weights_only=True,
)
classes = checkpoint["classes"]

model = NeuralNetwork(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)

image_path = Path(sys.argv[1])
image = ImageOps.exif_transpose(Image.open(image_path)).convert("L")
input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

with torch.inference_mode():
    probabilities = model(input_tensor).softmax(dim=1)[0]
    scores, indices = probabilities.topk(5)


def readable_label(label):
    code = int(label)
    return "null" if code == 999 else chr(code)


for score, index in zip(scores.tolist(), indices.tolist()):
    code = classes[index]
    print(f"{readable_label(code)!r} (class {code}): {score:.1%}")
