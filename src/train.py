import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score
from torch.utils.data import DataLoader

from src.model import LSTMClassifier
from src.dataset import GestureDataset
from src.config import (
    NUM_CLASSES, INPUT_SIZE, FINAL_PARAMS, 
    EPOCHS, PATIENCE, OVERFIT_THRESHOLD, BATCH_SIZE, DEVICE
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Одна эпоха обучения."""
    model.train()
    train_preds = []
    train_targets = []
    train_probs = []
    train_loss = 0

    for X_batch, y_batch, lengths in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(X_batch, lengths)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(y_batch.detach().cpu().numpy())
        train_probs.extend(probs.detach().cpu().numpy())

    avg_loss = train_loss / len(train_loader)
    top1_acc = 100. * np.mean(np.array(train_preds) == np.array(train_targets))
    f1 = f1_score(train_targets, train_preds, average='macro', zero_division=0) * 100
    top5 = top_k_accuracy_score(train_targets, train_probs, k=5, labels=range(NUM_CLASSES)) * 100

    return avg_loss, top1_acc, f1, top5


def validate_epoch(model, val_loader, criterion, device):
    """Одна эпоха валидации."""
    model.eval()
    val_preds = []
    val_targets = []
    val_probs = []
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch, lengths in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            logits = model(X_batch, lengths)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    top1_acc = 100. * np.mean(np.array(val_preds) == np.array(val_targets))
    f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0) * 100
    top5 = top_k_accuracy_score(val_targets, val_probs, k=5, labels=range(NUM_CLASSES)) * 100

    return avg_loss, top1_acc, f1, top5


def main():
    parser = argparse.ArgumentParser(description='Обучение модели распознавания жестов')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Количество эпох')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='models/best_model.pt', help='Путь для сохранения')
    parser.add_argument('--data_dir', type=str, default='data/', help='Директория с данными')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Параметры модели: {FINAL_PARAMS}")

    # Загрузка данных
    print("\nЗагрузка данных...")
    X_train = np.load(f'{args.data_dir}X_train_norm.npy')
    X_val = np.load(f'{args.data_dir}X_val_norm.npy')
    y_train = np.load(f'{args.data_dir}y_train.npy')
    y_val = np.load(f'{args.data_dir}y_val.npy')

    train_dataset = GestureDataset(X_train, y_train, augment=True)
    val_dataset = GestureDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Создание модели
    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        conv_channels=FINAL_PARAMS['conv_channels'],
        kernel_size=3,
        hidden_size=FINAL_PARAMS['hidden_size'],
        num_layers=FINAL_PARAMS['num_layers'],
        num_classes=NUM_CLASSES,
        dropout=FINAL_PARAMS['dropout']
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=FINAL_PARAMS['label_smoothing'])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINAL_PARAMS['lr'],
        weight_decay=FINAL_PARAMS['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = 0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'Epoch':^6} | {'Train Loss':^10} | {'Train Top-1':^10} | {'Train F1':^10} | {'Train Top-5':^10} | "
          f"{'Val Top-1':^10} | {'Val F1':^10} | {'Val Top-5':^10}")
    print("-" * 110)

    for epoch in range(args.epochs):
        train_loss, train_top1, train_f1, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_top1, val_f1, val_top5 = validate_epoch(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_f1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:^6} | {train_loss:^10.4f} | {train_top1:^10.2f} | {train_f1:^10.2f} | {train_top5:^10.2f} | "
                  f"{val_top1:^10.2f} | {val_f1:^10.2f} | {val_top5:^10.2f}")

        # Проверка на переобучение
        overfit_gap = train_f1 - val_f1
        if overfit_gap > OVERFIT_THRESHOLD and epoch > 10:
            print(f"\n⚠️ Обучение остановлено: сильное переобучение (разрыв {overfit_gap:.1f}%)")
            break

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_top1_acc': val_top1,
                'val_top5': val_top5,
                'train_f1': train_f1,
                'train_top1_acc': train_top1,
                'train_top5_acc': train_top5,
                'params': FINAL_PARAMS,
            }, args.save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping на эпохе {epoch+1}")
            break

    print(f"\nЛучшая модель: эпоха {best_epoch}, Val F1: {best_val_f1:.2f}%")


if __name__ == '__main__':
    main()