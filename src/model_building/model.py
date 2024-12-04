# src/model_building/model.py
import string
import torch
import torch.nn as nn
import argparse
import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass

    def model_info(self, show_arch=False, show_num_para=True):
        num_parameters = sum(p.numel() for p in self.parameters())
        if show_arch:
            print(self)
        if show_num_para:
            print(f"Number of parameters: {num_parameters}")

    def save(self, model_path):
        torch.save({'model_state_dict': self.state_dict()}, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        return self


class Transformer(Model):
    def __init__(self, input_dictionary=None, output_dictionary=None, emb_dim=128, nhead=8, num_layers=7, hidden_dim=256, dropout=0.1, max_seq_len=150):
        super(Transformer, self).__init__()
        if input_dictionary is None:
            input_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ") + list("bcfghjklmnpqrstvwxz0123456789 ") + list(string.punctuation)
        if output_dictionary is None:
            output_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệoóòỏõọôốồổỗộơớờởỡợiíìỉĩịuúùủũụưứừửữựyýỳỷỹỵ")
        self.input_dictionary = input_dictionary
        self.output_dictionary = output_dictionary
        self.input_vocab_size = len(input_dictionary) + 1
        self.output_vocab_size = len(output_dictionary) + 1

        self.embedding = nn.Embedding(self.input_vocab_size, emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Đặt batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(emb_dim, self.output_vocab_size)

        self.char_to_index = {char: i + 1 for i, char in enumerate(input_dictionary)}
        self.index_to_char = {i + 1: char for i, char in enumerate(output_dictionary)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoding = self.create_positional_encoding(max_seq_len, emb_dim)

    def create_positional_encoding(self, seq_len, emb_dim):
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim))
        pe = torch.zeros(seq_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(self.device)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x.long()) + self.positional_encoding[:, :seq_len, :]
        x = self.encoder(x)
        return self.output_layer(x)

    def encode_text(self, text):
        encoded = [self.char_to_index.get(char, 0) for char in text]
        max_len = self.positional_encoding.size(1)
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded += [0] * (max_len - len(encoded))
        return torch.tensor([encoded], dtype=torch.long)

    def decode_predictions(self, predictions, original_text):
        result = []
        for i, idx in enumerate(predictions):
            if idx == 0:
                result.append(original_text[i] if i < len(original_text) else '')
            else:
                result.append(self.index_to_char.get(idx, ''))
        return ''.join(result)

    def predict(self, text):
        encoded_input = self.encode_text(text).to(self.device)
        logits = self.forward(encoded_input).detach().cpu()
        predictions = torch.argmax(logits, dim=-1).numpy()[0]
        return self.decode_predictions(predictions, text)


def model_info():
    model = Transformer()
    model.model_info(show_arch=True)

if __name__ == "__main__":
    # Eg: python model.py "info"
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["info", "predict"], help="Run mode")
    parser.add_argument("--model_path", default="model.pt", help="Path to model")
    parser.add_argument("--input_text", help="Text to predict")
    args = parser.parse_args()

    model = Transformer()
    if args.mode == "info":
        model.model_info(show_arch=True)
    elif args.mode == "predict":
        model.load_model(args.model_path)
        print("Predicted:", model.predict(args.input_text))
