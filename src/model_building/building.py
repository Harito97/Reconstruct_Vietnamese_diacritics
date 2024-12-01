import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Đường dẫn dữ liệu
input_file_path = "./data/processed/corpus-title-no-accent-unicode.txt"  # Dữ liệu đầu vào
output_file_path = "./data/processed/corpus-title-unicode.txt"          # Dữ liệu đầu ra

# Hàm load data (không thay đổi)
def load_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in:
        X = [list(map(int, line.strip().split())) for line in f_in]

    with open(output_file, "r", encoding="utf-8") as f_out:
        y = [list(map(int, line.strip().split())) for line in f_out]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

# Lớp ANN (không thay đổi)
class ANN(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=150, dropout_rate=0.2):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        return x

# Hàm train_model (không thay đổi)
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device="cpu"):
    model = model.to(device)
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_model = None
    best_val_model = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                predictions = model(X_val)
                loss = criterion(predictions, y_val)
                val_loss += loss.item()

        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            print("Warning: Validation loader is empty!")
            val_loss = float('inf')  # Hoặc giá trị mặc định khác

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_model = model.state_dict()
            torch.save(best_train_model, "./models/vietnamese_diacritics_best_train.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_model = model.state_dict()
            torch.save(best_val_model, "./models/vietnamese_diacritics_best_val.pth")

def model_info():
    # Hiển thị thông tin về mô hình
    # number of parameters
    # model architecture
    model = ANN(input_dim=150, hidden_dim=150, dropout_rate=0.2)
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


# Hàm chính với tối ưu đa nhân
def main(batch_size=20 * 10**3 * 6 * 3):
    # 30 GB RAM => 90 GB RAM
    # Phát hiện số nhân CPU
    num_workers = os.cpu_count() if os.cpu_count() else 20  # Mặc định 20 nếu không xác định được số nhân
    print(f"Using {num_workers} workers for data loading.")

    # Load dữ liệu
    print("Loading data...")
    X, y = load_data(input_file_path, output_file_path)

    # Chia dữ liệu
    num_train = 9 * 10 ** 6
    train_data = TensorDataset(X[:num_train], y[:num_train])
    val_data = TensorDataset(X[num_train:], y[num_train:])

    # Tạo DataLoader với tối ưu đa nhân
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_batch_size = min(9487416 - 9000000, len(val_data))  # Batch size tối đa bằng kích thước tập validation
    val_loader = DataLoader(
        val_data, batch_size=val_batch_size, num_workers=num_workers, pin_memory=True
    )

    # Xây dựng mô hình
    print("Building model...")
    model = ANN(input_dim=150, hidden_dim=150, dropout_rate=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    print("Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device=device)

if __name__ == "__main__":
    main()
