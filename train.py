import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Dataset path
data_dir = "dataset"

# ── Transform ──────────────────────────────────────────────────────────────────
# IMPORTANT: Must use same normalization as main.py (ImageNet mean/std)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset = datasets.ImageFolder(data_dir, transform=transform)
print("Classes:", dataset.classes)
print(f"Total images: {len(dataset)}")

train_size = int(0.8 * len(dataset))
test_size  = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ── Training setup ─────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

epochs = 10

loss_list     = []
accuracy_list = []

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    accuracy = 100 * correct / total
    loss_list.append(running_loss)
    accuracy_list.append(accuracy)
    print(f"Epoch {epoch+1}/{epochs}  Loss: {running_loss:.4f}  Train Accuracy: {accuracy:.2f}%")

# ── Save model ─────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "plant_model.pth")
print("\n✅ Model saved as plant_model.pth")

# ── Loss graph ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs+1), loss_list, marker='o', color='#1d8a4e')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_graph.png")
print("Loss graph saved")

# ── Accuracy graph ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, epochs+1), accuracy_list, marker='o', color='#d6620f')
plt.title("Training Accuracy per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("accuracy_graph.png")
print("Accuracy graph saved")

# ── Confusion matrix ───────────────────────────────────────────────────────────
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved")
print("\n✅ All done!")
