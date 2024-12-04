# src/model_building/model.py
import string
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass

    def model_info(self, show_arch:bool=False, show_num_para:bool=True):
        num_parameters = sum(p.numel() for p in self.parameters())
        if show_arch:
            print("Model architecture:")
            print(self)
        if show_num_para:
            print(f"Number of parameters: {num_parameters}")

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=True))
        self.eval()
        return self

# Mô hình ANN (Artificial Neural Network)
class ANN(Model):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.valid_range_unicode = range(0x110000)  # Phạm vi Unicode hợp lệ
        self.token_size = input_dim

    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        x = self.output_layer(x)
        return x

    def text_to_unicode(self, text):
        result = [ord(char) for char in text]
        if len(result) > self.token_size:
            result = result[:self.token_size]
        elif len(result) < self.token_size:
            result += [0] * (self.token_size - len(result))
        return result

    def predict(self, text):
        x = torch.tensor([self.text_to_unicode(text)], dtype=torch.float32).to(self.device)
        y = self.forward(x).detach().cpu().numpy()[0]
        return ''.join(chr(max(0, min(int(round(num)), 0x10FFFF))) for num in y if int(round(num)) in self.valid_range_unicode)

class Transformer(Model):
    def __init__(self, input_dictionary=None, output_dictionary=None, emb_dim=128, nhead=8, num_layers=7, hidden_dim=256, dropout=0.1, max_seq_len=150):
        super(Transformer, self).__init__()
        if input_dictionary is None:
            input_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ") + list("bcfghjklmnpqrstvwxz0123456789 ") + list(string.punctuation)
        if output_dictionary is None:
            output_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệoóòỏõọôốồổỗộơớờởỡợiíìỉĩịuúùủũụưứừửữựyýỳỷỹỵ")
        self.input_dictionary = input_dictionary
        self.input_vocab_size = len(input_dictionary) + 1  # Thêm 1 cho ký tự "rỗng" (index 0)
        self.output_dictionary = output_dictionary
        self.output_vocab_size = len(output_dictionary) + 1  # Thêm 1 cho ký tự "rỗng" (index 0)
        self.emb_dim = emb_dim

        # Lớp embedding
        self.embedding = nn.Embedding(self.input_vocab_size, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, emb_dim))  # Positional Encoding

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(emb_dim, self.output_vocab_size)

        # Tạo bảng mã hóa từ điển
        self.char_to_index = {char: i + 1 for i, char in enumerate(input_dictionary)}  # Index từ 1
        self.index_to_char = {i + 1: char for i, char in enumerate(output_dictionary)}  # Index từ 1

    def forward(self, x):
        x = x.to(self.device)
        seq_len = x.size(1)
        # Embedding
        x = self.embedding(x.long())  # Kích thước: (batch_size, seq_len, emb_dim)
        # Thêm positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        # Transformer Encoder
        x = self.encoder(x)  # Kích thước: (batch_size, seq_len, emb_dim)
        # Output Layer
        x = self.output_layer(x)  # Kích thước: (batch_size, seq_len, vocab_size)
        return x

    def decode_predictions(self, predictions, original_text):
        result = []
        for i, idx in enumerate(predictions):
            if idx == 0:
                result.append(original_text[i] if i < len(original_text) else '')
            else:
                result.append(self.index_to_char.get(idx, ''))
        return ''.join(result)

    def encode_text(self, text):
        """Mã hóa chuỗi thành vector index dựa trên từ điển."""
        encoded = []
        for char in text:
            encoded.append(self.char_to_index.get(char, 0))  # Ký tự không có trong từ điển -> 0
        # Cắt hoặc bổ sung padding để đạt độ dài cố định
        if len(encoded) > self.positional_encoding.size(1):
            encoded = encoded[:self.positional_encoding.size(1)]
        else:
            encoded += [0] * (self.positional_encoding.size(1) - len(encoded))
        return torch.tensor([encoded], dtype=torch.long)

    def predict(self, text):
        """Dự đoán chuỗi với khả năng khôi phục ký tự ban đầu nếu không thể thêm dấu."""
        # Mã hóa chuỗi đầu vào
        encoded_input = self.encode_text(text).to(next(self.parameters()).device)
        # Dự đoán
        logits = self.forward(encoded_input).detach().cpu()
        predictions = torch.argmax(logits, dim=-1).numpy()[0]  # Lấy nhãn xác suất cao nhất
        # Giải mã kết quả
        return self.decode_predictions(predictions, text)


def get_model(model_type='Transformer', **kwargs):
    if model_type == 'ANN':
        return ANN(**kwargs)
    elif model_type == 'Transformer':
        return Transformer(**kwargs)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a text model.")
    parser.add_argument("mode", choices=["info", "predict"], help="Mode to run the model.")
    parser.add_argument("--model_type", default="Transformer", help="Model type (ANN or Transformer).")
    parser.add_argument("--input_text", default="", help="Text input for prediction.")
    parser.add_argument("--model_path", default="model.pt", help="Path to the saved model.")
    args = parser.parse_args()

    # Load model
    model = get_model(model_type=args.model_type)
    if args.mode == "info":
        model.model_info(show_arch=True)
    elif args.mode == "predict":
        model.load_model(args.model_path)
        output_text = model.predict(args.input_text)
        print("Input:", args.input_text)
        print("Output:", output_text)
