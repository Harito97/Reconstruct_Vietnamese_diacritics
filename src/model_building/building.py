# src/model_building/building.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Đường dẫn dữ liệu
input_file_path = "./data/processed/corpus-title-no-accent-unicode.txt"  # Dữ liệu đầu vào
output_file_path = "./data/processed/corpus-title-unicode.txt"          # Dữ liệu đầu ra

def load_data(input_file, output_file):
    """
    Load dữ liệu từ file và chuyển đổi thành Tensor.
    """
    with open(input_file, "r", encoding="utf-8") as f_in:
        X = [list(map(int, line.strip().split())) for line in f_in]

    with open(output_file, "r", encoding="utf-8") as f_out:
        y = [list(map(int, line.strip().split())) for line in f_out]

    # Chuyển sang Tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

class ANN(nn.Module):
    """
    Mạng ANN với 5 lớp dense, mỗi lớp có 150 neuron.
    """
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

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device="cpu"):
    """
    Huấn luyện mô hình và lưu hai phiên bản:
    - Mô hình có loss thấp nhất trên tập train.
    - Mô hình có loss thấp nhất trên tập validation.
    """
    model = model.to(device)
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_train_model = None
    best_val_model = None

    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                predictions = model(X_val)
                loss = criterion(predictions, y_val)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Lưu model tốt nhất trên tập train
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_model = model.state_dict()

        # Lưu model tốt nhất trên tập validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_model = model.state_dict()

    # Lưu hai model
    print("Saving best train model...")
    torch.save(best_train_model, "./models/vietnamese_diacritics_best_train.pth")
    print("Saving best validation model...")
    torch.save(best_val_model, "./models/vietnamese_diacritics_best_val.pth")

def predict(model, X, device="cpu"):
    """
    Dự đoán đầu ra và làm tròn về số nguyên gần nhất.
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        predictions = model(X)
        predictions = torch.round(predictions).long()
    return predictions.cpu().numpy()

def unicode_to_text(unicode_list):
    """
    Chuyển đổi danh sách mã Unicode thành chuỗi ký tự.
    """
    return "".join(chr(code) for code in unicode_list if code > 0)

def try_model(model, X):
    # Test mô hình
    print("Testing model...")
    test_input = X[:10]  # Lấy 10 dòng đầu tiên để kiểm tra
    predictions = predict(model, test_input)

    # Chuyển đổi mã Unicode thành text để kiểm tra
    for i in range(len(test_input)):
        input_text = unicode_to_text(test_input[i].tolist())
        predicted_text = unicode_to_text(predictions[i].tolist())
        print(f"Input: {input_text}")
        print(f"Prediction: {predicted_text}")

def main():
    # Load dữ liệu
    print("Loading data...")
    X, y = load_data(input_file_path, output_file_path)

    # Chia dữ liệu thành tập train và validation
    num_train = 9 * 10 ** 9  # 9 triệu record
    num_val = len(X) - num_train
    train_data = TensorDataset(X[:num_train], y[:num_train])
    val_data = TensorDataset(X[num_train:], y[num_train:])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    # Xây dựng mô hình
    print("Building model...")
    model = ANN(input_dim=150, hidden_dim=150, dropout_rate=0.2)
    criterion = nn.MSELoss()  # Loss cho bài toán hồi quy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện mô hình
    print("Training model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device="cpu")

    # Kiểm tra mô hình
    try_model(model, X)

if __name__ == '__main__':
    main()
