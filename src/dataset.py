import random
import torch
import numpy as np
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    """
    Датасет для загрузки последовательностей лэндмарков жестов.
    
    Поддерживает аугментацию данных: масштабирование, сдвиг, добавление шума.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = True,
        noise_std: float = 0.02,
        scale_range: tuple = (0.95, 1.05),
        translate_range: float = 0.05,
        max_len: int = 73
    ):
        """
        Args:
            X: Входные данные [samples, time, features]
            y: Метки классов
            augment: Включить аугментацию
            noise_std: Стандартное отклонение шума
            scale_range: Диапазон масштабирования
            translate_range: Диапазон сдвига
            max_len: Фиксированная длина последовательности для batch
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.max_len = max_len
        self.lengths = torch.tensor((X.sum(axis=2) != 0).sum(axis=1), dtype=torch.long)
        self.lengths = torch.clamp(self.lengths, min=1)
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.translate_range = translate_range

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.X)

    def apply_augmentations(self, sequence: torch.Tensor, length: int) -> torch.Tensor:
        """
        Применяет аугментации к одной последовательности.
        
        Args:
            sequence: Исходная последовательность
            length: Реальная длина последовательности (без паддинга)
            
        Returns:
            Аугментированная последовательность
        """
        seq = sequence.clone()

        # Случайное масштабирование
        if random.random() > 0.5:
            scale = random.uniform(*self.scale_range)
            seq = seq * scale

        # Случайный сдвиг
        if random.random() > 0.5:
            translation = random.uniform(-self.translate_range, self.translate_range)
            seq = seq + translation

        # Добавление шума (только к реальным кадрам)
        if random.random() > 0.5:
            noise = torch.randn_like(seq) * self.noise_std
            mask = torch.zeros_like(seq)
            mask[:length] = 1.0
            seq = seq + noise * mask

        return seq

    def __getitem__(self, idx: int):
        """
        Возвращает элемент датасета по индексу.
        
        Returns:
            tuple: (sequence, label, length)
        """
        X_seq = self.X[idx]
        y_label = self.y[idx]
        length = self.lengths[idx].item()

        if self.augment:
            X_seq = self.apply_augmentations(X_seq, length)

        return X_seq, y_label, length