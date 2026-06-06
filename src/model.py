import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM классификатор для распознавания жестов с использованием MediaPipe landmarks.
    
    Архитектура:
    - Conv1D слой для извлечения пространственных признаков
    - BiLSTM для обработки временной динамики
    - Механизм внимания для взвешивания важных кадров
    - Два полносвязных слоя для классификации
    """
    
    def __init__(
        self,
        input_size: int = 378,
        conv_channels: int = 128,
        kernel_size: int = 3,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 1000,
        dropout: float = 0.25
    ):
        """
        Args:
            input_size: Размер входных признаков на один кадр
            conv_channels: Количество каналов в свёрточном слое
            kernel_size: Размер ядра свёртки
            hidden_size: Размер скрытого состояния LSTM
            num_layers: Количество слоёв LSTM
            num_classes: Количество классов жестов
            dropout: Вероятность dropout
        """
        super().__init__()

        # Conv1D слой
        self.conv1 = nn.Conv1d(
            input_size, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_bn = nn.BatchNorm1d(conv_channels)
        self.conv_dropout = nn.Dropout(dropout)

        # BiLSTM слой
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Механизм внимания
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Классификатор
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            x: Входной тензор формы (batch, seq_len, input_size)
            lengths: Длины последовательностей для pack_padded_sequence
            
        Returns:
            Логиты предсказаний формы (batch, num_classes)
        """
        # Conv1D: (batch, seq_len, features) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv_bn(x)
        x = torch.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        # LSTM с pack_padded_sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Механизм внимания
        attn_weights = self.attention(output).squeeze(-1)
        mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (output * attn_weights.unsqueeze(-1)).sum(dim=1)
        pooled = self.dropout(pooled)

        # Классификатор
        x = self.fc1(pooled)
        x = self.fc1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits