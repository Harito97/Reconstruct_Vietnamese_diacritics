# src/data_collection/data_no_accent.py
import unidecode
from concurrent.futures import ProcessPoolExecutor

def remove_vietnamese_accent(text):
    """
    Loại bỏ dấu tiếng Việt khỏi chuỗi.
    """
    return unidecode.unidecode(text)

def process_batch(lines):
    """
    Xử lý một batch dòng: loại bỏ dấu tiếng Việt.
    """
    return [remove_vietnamese_accent(line.strip()) for line in lines]

def main():
    """
    Loại bỏ dấu tiếng Việt từ một tệp văn bản và lưu kết quả vào tệp khác.
    """
    print("Starting to remove Vietnamese accents from a text file...")

    # Đường dẫn tệp
    input_file = "./data/raw/corpus-title.txt"
    output_file = "./data/processed/corpus-title-no-accent.txt"
    batch_size = 10_000  # Số dòng xử lý mỗi batch

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        lines = []
        for i, line in enumerate(f_in, start=1):
            lines.append(line)
            if i % batch_size == 0:  # Đủ batch, xử lý
                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(process_batch, [lines]))
                for batch_result in results:
                    f_out.write("\n".join(batch_result) + "\n")
                lines = []  # Reset batch

        # Xử lý phần còn lại nếu có
        if lines:
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(process_batch, [lines]))
            for batch_result in results:
                f_out.write("\n".join(batch_result) + "\n")

    print("Done!")

if __name__ == "__main__":
    main()
