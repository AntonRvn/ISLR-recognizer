"""
Скрипт для тестирования обученной модели.
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.model import LSTMClassifier
from src.dataset import GestureDataset
from src.config import NUM_CLASSES, INPUT_SIZE, DEVICE, FINAL_PARAMS, BATCH_SIZE


def evaluate_model(model, test_loader, device):
    """
    Вычисляет метрики на тестовой выборке.
    
    Returns:
        tuple: (top1_acc, f1_macro, top5_acc)
    """
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []

    with torch.no_grad():
        for X_batch, y_batch, lengths in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            logits = model(X_batch, lengths)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_batch.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)

    top1_acc = 100. * np.mean(test_preds == test_targets)
    f1_macro = f1_score(test_targets, test_preds, average='macro', zero_division=0) * 100
    top5_acc = top_k_accuracy_score(test_targets, test_probs, k=5, labels=range(NUM_CLASSES)) * 100

    return top1_acc, f1_macro, top5_acc


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели распознавания жестов')
    parser.add_argument('--model_path', type=str, required=True, help='Путь к файлу модели (.pt)')
    parser.add_argument('--data_dir', type=str, default='data/', help='Директория с данными')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Загрузка данных
    print("\nЗагрузка тестовых данных...")
    X_test = np.load(f'{args.data_dir}X_test_norm.npy')
    y_test = np.load(f'{args.data_dir}y_test.npy')

    test_dataset = GestureDataset(X_test, y_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Загрузка модели
    print(f"Загрузка модели из {args.model_path}")
    checkpoint = torch.load(args.model_path, weights_only=False)

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        conv_channels=FINAL_PARAMS['conv_channels'],
        kernel_size=3,
        hidden_size=FINAL_PARAMS['hidden_size'],
        num_layers=FINAL_PARAMS['num_layers'],
        num_classes=NUM_CLASSES,
        dropout=FINAL_PARAMS['dropout']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Оценка
    top1_acc, f1_macro, top5_acc = evaluate_model(model, test_loader, DEVICE)

    # Вывод результатов
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"Test Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Test F1 (macro): {f1_macro:.2f}%")
    print(f"Test Top-5 Accuracy: {top5_acc:.2f}%")

    # Если в чекпоинте есть сохранённые метрики
    if 'val_f1' in checkpoint:
        print("\n" + "="*60)
        print("СОХРАНЁННЫЕ МЕТРИКИ (из чекпоинта)")
        print("="*60)
        print(f"Validation F1: {checkpoint['val_f1']:.2f}%")
        print(f"Validation Top-1: {checkpoint.get('val_top1_acc', 0):.2f}%")
        print(f"Validation Top-5: {checkpoint.get('val_top5', 0):.2f}%")


if __name__ == '__main__':
    main()