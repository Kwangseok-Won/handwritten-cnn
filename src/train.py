from pathlib import Path
from zipfile import ZipFile

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get CPU or GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Download/load the dataset
def get_class_folder(split):
    zip_path = hf_hub_download(
        repo_id="louisraedisch/AlphaNum",
        repo_type="dataset",
        filename=f"data/{split}.zip",
    )

    extract_to = Path("data/unzipped") / split
    extract_to.mkdir(parents=True, exist_ok=True)

    # Extract once; reuse the extracted files on later runs.
    if not any(extract_to.iterdir()):
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_to)

    # Find the directory whose subfolders contain images.
    image_types = {".png", ".jpg", ".jpeg", ".bmp"}
    folders = [extract_to] + [path for path in extract_to.rglob("*") if path.is_dir()]

    for folder in folders:
        class_folders = [path for path in folder.iterdir() if path.is_dir()]
        if len(class_folders) > 1 and any(
            any(file.suffix.lower() in image_types for file in class_folder.iterdir())
            for class_folder in class_folders
        ):
            return folder

    raise FileNotFoundError(f"Could not find character folders for {split}")


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

training_data = datasets.ImageFolder(
    root=get_class_folder("train"),
    transform=transform,
)

test_data = datasets.ImageFolder(
    root=get_class_folder("test"),
    transform=transform,
)

classes = training_data.classes
print(f"Found {len(classes)} classes and {len(training_data)} training images")

if test_data.classes != training_data.classes:
    raise ValueError("Train and test folders have different classes")

for i in torch.randperm(len(training_data))[:10].tolist():
    image, target = training_data[i]
    folder_label = classes[target]
    character = "null" if folder_label == "999" else chr(int(folder_label))
    print(training_data.samples[i][0], folder_label, repr(character))


# Create data loaders
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X: {X.shape}")  # torch.Size([64, 1, 28, 28])
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# A small CNN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(classes)),
        )

    def forward(self, x):
        return self.cnn(x)


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


Xb, yb = next(iter(train_dataloader))
Xb, yb = Xb.to(device), yb.to(device)

probe = NeuralNetwork().to(device)
probe_optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

for step in range(300):
    probe.train()
    probe_optimizer.zero_grad(set_to_none=True)

    logits = probe(Xb)
    loss = loss_fn(logits, yb)

    loss.backward()
    probe_optimizer.step()

probe.eval()
with torch.inference_mode():
    accuracy = (probe(Xb).argmax(dim=1) == yb).float().mean().item()

print(f"One-batch loss: {loss.item():.4f}")
print(f"One-batch accuracy: {accuracy:.1%}")


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():.4f}")


def test(dataloader, model, split="Eval"):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"{split} accuracy: {100 * correct / total:.1f}%")


# Train
epochs = 30
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n----------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(train_dataloader, model, split="Train")
    test(test_dataloader, model, split="Test")


Path("models").mkdir(exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
    },
    "models/alphanum_cnn.pth",
)

print("Done!")
