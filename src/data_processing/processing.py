# src/data_processing/processing.py
import torch
import string
from typing import List

# Bảng dictionary ký tự
input_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ") + list("bcfghjklmnpqrstvwxz0123456789 ") + list(string.punctuation)
# punctuation is '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# 136 character

# Tạo ánh xạ ký tự -> index
char_to_index = {char: idx + 1 for idx, char in enumerate(input_dictionary)}  # +1 để 0 dùng cho padding
max_length = 150  # Độ dài tối đa của chuỗi

def convert_text_to_indices(text: str, token_size: int = max_length, char_to_index=char_to_index) -> List[int]:
    """
    Chuyển đổi câu thành danh sách index dựa trên dictionary.
    Nếu độ dài câu < token_size thì thêm padding; nếu dài hơn thì cắt ngắn.
    """
    indices = [char_to_index.get(char, 0) for char in text.lower()]  # 0 cho ký tự không có trong dictionary
    if len(indices) > token_size:
        indices = indices[:token_size]
    else:
        indices += [0] * (token_size - len(indices))  # Padding với giá trị 0
    return indices

def process_and_save_data(input_file_path: str, output_file_path: str):
    """
    Xử lý file đầu vào, chuyển đổi các câu thành vector chỉ số, và lưu kết quả.
    """
    X, y = [], []
    with open(input_file_path, "r", encoding="utf-8") as f_in, \
         open(output_file_path, "r", encoding="utf-8") as f_out:
        input_lines = f_in.readlines()
        output_lines = f_out.readlines()

        if len(input_lines) != len(output_lines):
            raise ValueError("Số dòng trong file đầu vào và đầu ra không khớp.")

        for in_line, out_line in zip(input_lines, output_lines):
            in_vector = convert_text_to_indices(in_line.strip())
            out_vector = convert_text_to_indices(out_line.strip())
            X.append(in_vector)
            y.append(out_vector)

    # Chuyển đổi thành Tensor và lưu file
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)
    torch.save(X_tensor, "./data/processed/X_transformer.pt")
    torch.save(y_tensor, "./data/processed/y_transformer.pt")
    print(f"Processed {len(X)} samples. Saved: X -> X_transformer.pt, y -> y_transformer.pt")

def load_processed_data(X_file_path="./data/processed/X_transformer.pt", y_file_path="./data/processed/y_transformer.pt"):
    """
    Tải dữ liệu đã được xử lý từ file.
    """
    X = torch.load(X_file_path)
    y = torch.load(y_file_path)
    print(f"Data loaded: X -> {X.shape}, y -> {y.shape}")
    return X, y

def main():
    # input_file_path = "./data/raw/input.txt"  # File đầu vào
    # output_file_path = "./data/raw/output.txt"  # File đầu ra
    input_file_path = "./data/processed/corpus-title-no-accent.txt"  # File đầu vào
    output_file_path = "./data/raw/corpus-title.txt"  # File đầu ra

    process_and_save_data(input_file_path, output_file_path)
    X, y = load_processed_data() # "X_transformer.pt", "y_transformer.pt"

if __name__ == "__main__":
    main()
