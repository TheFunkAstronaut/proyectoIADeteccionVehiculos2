import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Configuraciones
train_data_dir = "/home/not_funker/PycharmProjects/proyectoIADeteccionVehiculos/data/train_resized"
test_data_dir = "/home/not_funker/PycharmProjects/proyectoIADeteccionVehiculos/data/test_resized"
model_path = "/home/not_funker/PycharmProjects/proyectoIADeteccionVehiculos/vehicle_classifier.pth"
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Datasets y loaders
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modelo
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 categorías
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Época [{epoch + 1}/{num_epochs}], Pérdida: {running_loss / len(train_loader):.4f}")

# Guardar modelo
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en {model_path}")

# Evaluación en test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Precisión en test: {accuracy:.2%}")

