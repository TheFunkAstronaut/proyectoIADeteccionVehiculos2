import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Configuración
model_path = "/home/not_funker/PycharmProjects/proyectoIADeteccionVehiculos/vehicle_classifier.pth"
classes = ['auto', 'bici', 'camion', 'motocicleta']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cargar el modelo
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 clases
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Captura de video en tiempo real
cap = cv2.VideoCapture(2)  # Índice de la cámara USB
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen al formato necesario
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    # Inferencia del modelo
    output = model(img)
    _, predicted = torch.max(output, 1)

    confidence = torch.softmax(output, dim=1)[0][predicted].item()
    label = classes[predicted]

    # Dibujar rectángulo y texto si la confianza es alta
    if confidence > 0.5:  # Umbral para mostrar detecciones
        height, width, _ = frame.shape
        start_point = (width // 4, height // 4)  # Coordenadas del rectángulo (aproximado al centro)
        end_point = (3 * width // 4, 3 * height // 4)  # Coordenadas opuestas
        color = (0, 255, 0)  # Verde
        thickness = 2

        # Dibujar el rectángulo
        cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Mostrar texto con la clase detectada
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la ventana de video
    cv2.imshow('Vehicle Detection', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

