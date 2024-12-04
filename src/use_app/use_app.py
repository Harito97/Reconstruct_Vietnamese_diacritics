import torch
import unidecode
from src.model_building.model import Transformer

def remove_vietnamese_accent(text):
    # Input: string
    # Output: string
    # Example: remove_vietnamese_accent("Hôm nay trời đẹp quá!") -> "Hom nay troi dep qua!"
    # ---
    # Use:
    # original_text = "Tôi yêu Việt Nam"
    # text_without_accent = remove_vietnamese_accent(original_text)
    # print(text_without_accent)
    # # Result: "Toi yeu Viet Nam"
    # ---
    # pip install unidecode
    return unidecode.unidecode(text)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mô hình đã huấn luyện
    model_path = "./models/best_val_Transformer_model.pth"  # Đường dẫn tới file mô hình tốt nhất
    model = Transformer().to(device)
    model.load_model(model_path)
    print("Mô hình đã được tải thành công!")

    # Giao diện người dùng
    print("\nChọn tùy chọn:")
    print("1: Nhập một câu có dấu và kiểm tra dự đoán khôi phục dấu từ câu không dấu.")
    print("2: Nhập một câu không dấu để mô hình dự đoán thêm dấu.")
    print("q: Thoát.")

    while True:
        option = input("\nChọn (1/2/q): ").strip()
        if option == "q":
            print("Thoát chương trình.")
            break

        elif option == "1":
            # Chế độ kiểm tra với câu có dấu
            input_text = input("Nhập câu có dấu: ").strip()
            no_accent_text = remove_vietnamese_accent(input_text)
            print(f"Câu không dấu: {no_accent_text}")

            prediction = model.predict(no_accent_text)
            print(f"Dự đoán thêm dấu: {prediction}")
            print(f"So sánh:\n  Gốc: {input_text}\n  Dự đoán: {prediction}")

        elif option == "2":
            # Chế độ thêm dấu từ câu không dấu
            input_text = input("Nhập câu không dấu: ").strip()
            prediction = model.predict(input_text)
            print(f"Dự đoán thêm dấu: {prediction}")

        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn 1, 2 hoặc q.")


if __name__ == "__main__":
    main()
