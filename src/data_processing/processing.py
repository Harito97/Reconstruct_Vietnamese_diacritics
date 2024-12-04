# src/data_processing/processing.py
import torch
import string
from typing import List
from multiprocessing import Pool, cpu_count

# Bảng dictionary ký tự
input_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ") + list("bcfghjklmnpqrstvwxz0123456789 ") + list(string.punctuation)
output_dictionary = list("aáàảãạăắằẳẵặâấầẩẫậdđeéèẻẽẹêếềểễệoóòỏõọôốồổỗộơớờởỡợiíìỉĩịuúùủũụưứừửữựyýỳỷỹỵ")
input_char_to_index = {char: idx + 1 for idx, char in enumerate(input_dictionary)}  # +1 để 0 dùng cho padding
output_char_to_index = {char: idx + 1 for idx, char in enumerate(output_dictionary)}  # +1 để 0 dùng cho padding
max_length = 150  # Độ dài tối đa của chuỗi

def convert_text_to_indices(text: str, token_size: int = max_length, char_to_index=input_char_to_index) -> List[int]:
    """
    Chuyển đổi câu thành danh sách index dựa trên dictionary.
    """
    indices = [char_to_index.get(char, 0) for char in text.lower()]  # 0 cho ký tự không có trong dictionary
    if len(indices) > token_size:
        indices = indices[:token_size]
    else:
        indices += [0] * (token_size - len(indices))  # Padding với giá trị 0
    return indices

def process_lines(args):
    """
    Hàm xử lý song song cho một phần của dữ liệu.
    """
    input_lines, output_lines = args
    X, y = [], []
    for in_line, out_line in zip(input_lines, output_lines):
        X.append(convert_text_to_indices(in_line.strip(), char_to_index=input_char_to_index))
        y.append(convert_text_to_indices(out_line.strip(), char_to_index=output_char_to_index))
    return X, y

def process_and_save_data(input_file_path: str, output_file_path: str):
    """
    Xử lý file đầu vào, chuyển đổi các câu thành vector chỉ số, và lưu kết quả.
    """
    with open(input_file_path, "r", encoding="utf-8") as f_in, \
         open(output_file_path, "r", encoding="utf-8") as f_out:
        input_lines = f_in.readlines()
        output_lines = f_out.readlines()

    if len(input_lines) != len(output_lines):
        raise ValueError("Số dòng trong file đầu vào và đầu ra không khớp.")

    # Chia dữ liệu thành các đoạn nhỏ để xử lý song song
    num_cores = cpu_count()
    chunk_size = len(input_lines) // num_cores + 1
    chunks = [
        (input_lines[i:i + chunk_size], output_lines[i:i + chunk_size])
        for i in range(0, len(input_lines), chunk_size)
    ]

    # Xử lý song song bằng Pool
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_lines, chunks)

    # Gộp kết quả
    X, y = [], []
    for X_chunk, y_chunk in results:
        X.extend(X_chunk)
        y.extend(y_chunk)

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
    X = torch.load(X_file_path, weights_only=True)
    y = torch.load(y_file_path, weights_only=True)
    print(f"Data loaded: X -> {X.shape}, y -> {y.shape}")
    return X, y

def main():
    input_file_path = "./data/processed/corpus-title-no-accent.txt"  # File đầu vào
    output_file_path = "./data/raw/corpus-title.txt"  # File đầu ra

    process_and_save_data(input_file_path, output_file_path)
    X, y = load_processed_data()

if __name__ == "__main__":
    main()
