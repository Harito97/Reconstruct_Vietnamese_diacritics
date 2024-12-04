# src/data_processing/processing.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score
from src.model_building.model import Transformer
from src.data_processing.processing import load_processed_data

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device="cpu"):
    model = model.to(device)
    best_train_loss = float("inf")
    best_val_loss = float("inf")

    # Tạo thư mục lưu model nếu chưa tồn tại
    os.makedirs("./models", exist_ok=True)
    best_train_model_path = "./models/best_train_Transformer_model.pth"
    best_val_model_path = "./models/best_val_Transformer_model.pth"

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Training loop
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)  # [batch_size, seq_len, vocab_size]
            predictions = predictions.view(-1, predictions.size(-1))  # [N, C]
            y_batch = y_batch.view(-1)  # [N]

            loss = criterion(predictions, y_batch)  # CrossEntropyLoss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item() * y_batch.size(0)  # Tổng loss
            train_correct += (torch.argmax(predictions, dim=-1) == y_batch).sum().item()
            train_total += y_batch.numel()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                predictions = model(X_val)  # [batch_size, seq_len, vocab_size]
                predictions = predictions.view(-1, predictions.size(-1))  # [N, C]
                y_val = y_val.view(-1)  # [N]

                loss = criterion(predictions, y_val)  # CrossEntropyLoss
                val_loss += loss.item() * y_val.size(0)  # Tổng loss
                val_correct += (torch.argmax(predictions, dim=-1) == y_val).sum().item()
                val_total += y_val.numel()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Log kết quả
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Save best model
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), best_train_model_path)
            print(f"  Saved best train model with loss {best_train_loss}.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_val_model_path)
            print(f"  Saved best val model with loss {best_val_loss}.")

    print("Training complete.")
    print(f"Best training loss: {best_train_loss:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processed data
    X, y = load_processed_data()
    dataset = TensorDataset(X, y)

    # Split into train and validation sets
    train_ratio = 0.8  # 80% train, 20% validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize the model
    model = Transformer()
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Padding index = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device=device)

if __name__ == "__main__":
    main()
