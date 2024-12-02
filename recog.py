from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
import numpy as np

# Константы
MODEL_PATH = "personality_net.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # Размер изображения для модели

# Подготовка трансформации
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# Загрузка модели
class PersonalityNet(nn.Module):
    def init(self, num_classes):
        super(PersonalityNet, self).init()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


# Загрузка сохранённой модели
model = PersonalityNet(40).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# Функция анализа изображения
def analyze_photo_with_model(photo_path):
    try:
        # Загрузка изображения
        image = Image.open(photo_path).convert("RGB")
        input_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

        # Прогнозирование
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().flatten()

        # Преобразование в словарь
        attribute_names = [f"attribute_{i}" for i in range(len(prediction))]
        print(attribute_names)
        decoded_predictions = {attr: float(prob) for attr, prob in zip(attribute_names, prediction)}

        return decoded_predictions

    except Exception as e:
        print(f"Ошибка в анализе изображения: {e}")
        return None


analyze_photo_with_model('static/uploads/vova.jpeg')
