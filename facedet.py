import torch
import cv2
import numpy as np
import os
from PIL import Image

# Пути к YOLOv5 файлам
DETECT_MODEL = 'face_detector.pt'  # Путь к весам модели YOLOv5 детекция людей
INPUT_FOLDER = 'test_images'  # Папка с тестовыми изображениями
OUTPUT_FOLDER = 'results'  # Папка для результатов

# Создаём папку для результатов, если её нет
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Определение устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Загрузка модели YOLOv5
model = torch.hub.load('yolov5', 'custom', path=DETECT_MODEL, source='local').to(device)

model.eval()

# Получаем классы напрямую из модели
classes = model.names  # Это встроенные классы в модель

# Генерация случайных цветов для каждого класса
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Функция для рисования прозрачного фона и полупрозрачной границы вокруг bbox и текста
def draw_prediction(img, class_id, confidence, x, y, w, h, classeslist):
    color = [int(c) for c in COLORS[class_id]]
    overlay = img.copy()
    alpha_border = 0.6  # Полупрозрачность границы bbox
    thickness = 4  # Толщина границы

    # Полупрозрачная граница bbox
    for i in range(thickness):
        cv2.rectangle(overlay, (x - i, y - i), (x + w + i, y + h + i), color, 1)
    cv2.addWeighted(overlay, alpha_border, img, 1 - alpha_border, 0, img)

# Основная функция для обработки изображений

def process_images():
    for img_name in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, img_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(img_name)[0]}_result.jpg")
        
        # Загружаем изображение
        img = cv2.imread(input_path)
        if img is None:
            print(f"Unable to read image: {img_name}")
            continue
        
        # Преобразуем изображение в формат, подходящий для YOLOv5
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Выполнение детекции
        results = model(pil_image)  # Детектируем людей

        # Получение результатов
        boxes = results.xyxy[0].cpu().numpy()  # Люди

        # Обрабатываем результаты второй модели (лица людей)
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            if conf > 0.2:  # Порог уверенности
                draw_prediction(img, int(class_id), conf, int(x1), int(y1), int(x2 - x1), int(y2 - y1), classes)

        # Сохранение результата
        cv2.imwrite(output_path, img)
        print(f"Processed and saved: {output_path}")

if __name__ == '__main__':
    process_images()