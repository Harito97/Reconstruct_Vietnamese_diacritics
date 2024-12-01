import torch
import torch.nn as nn

# Mô hình ANN
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

def load_model(model_path, input_dim=150, hidden_dim=150, dropout_rate=0.2):
    """
    Load mô hình từ file đã lưu.
    """
    model = ANN(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def remove_vietnamese_accent(text):
    """
    Loại bỏ dấu tiếng Việt khỏi một câu.
    """
    import unidecode
    return unidecode.unidecode(text)

def convert_text_to_unicode(text, token_size=150):
    """
    Chuyển đổi một chuỗi ký tự sang mã Unicode, đảm bảo kích thước cố định.
    """
    result = [ord(char) for char in text]
    if len(result) > token_size:
        result = result[:token_size]
    elif len(result) < token_size:
        result += [0] * (token_size - len(result))
    return result

def unicode_to_text(unicode_list):
    """
    Chuyển danh sách mã Unicode ngược lại thành chuỗi.
    """
    return ''.join(chr(int(round(num))) for num in unicode_list if int(round(num)) != 0)

def predict_with_models(sentence, model_train, model_val, device="cpu"):
    """
    Chạy dự đoán với hai mô hình đã lưu.
    """
    # Chuyển câu thành Unicode vector
    input_vector = convert_text_to_unicode(remove_vietnamese_accent(sentence))
    input_tensor = torch.tensor([input_vector], dtype=torch.float32).to(device)

    # Dự đoán với từng mô hình
    output_train = model_train(input_tensor).detach().cpu().numpy()[0]
    output_val = model_val(input_tensor).detach().cpu().numpy()[0]

    # Làm tròn và chuyển về chuỗi ký tự
    predicted_text_train = unicode_to_text(output_train)
    predicted_text_val = unicode_to_text(output_val)

    return predicted_text_train, predicted_text_val

def main():
    # Load hai mô hình
    model_train_path = "./models/vietnamese_diacritics_best_train.pth"
    model_val_path = "./models/vietnamese_diacritics_best_val.pth"

    print("Loading models...")
    model_train = load_model(model_train_path)
    model_val = load_model(model_val_path)

    # Chọn device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_train.to(device)
    model_val.to(device)

    print("Models loaded!")

    while True:
        # Nhập câu từ người dùng
        input_sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
        if input_sentence.lower() == "exit":
            break

        # In ra kết quả dự đoán từ hai mô hình
        predicted_train, predicted_val = predict_with_models(input_sentence, model_train, model_val, device)
        print(f"\nOriginal Sentence: {input_sentence}")
        print(f"Predicted by Train Model: {predicted_train}")
        print(f"Predicted by Validation Model: {predicted_val}")

if __name__ == "__main__":
    main()
