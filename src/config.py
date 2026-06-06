import torch

# Количество классов жестов
NUM_CLASSES = 1000

# Размер входных признаков (landmarks)
# 21 точка * 2 руки * 3 координаты (x, y, z) = 126
# ×2 для скорости и ускорения = 378
INPUT_SIZE = 378

# Гиперпараметры модели (лучшая найденная конфигурация)
FINAL_PARAMS = {
    'dropout': 0.61,
    'lr': 0.0024,
    'weight_decay': 0.0023,
    'hidden_size': 256,
    'conv_channels': 64,
    'num_layers': 1,
    'label_smoothing': 0.1
}

# Параметры обучения
EPOCHS = 250
PATIENCE = 15
OVERFIT_THRESHOLD = 20

# Размер батча
BATCH_SIZE = 64

# Целевая длина последовательности (после сжатия/растяжения)
TARGET_LEN = 55

# Устройство для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')