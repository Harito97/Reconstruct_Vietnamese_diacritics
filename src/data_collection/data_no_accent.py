# src/data_collection/data_no_accent.py
import unidecode

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

# def convert_text_to_unicode(text, token_size:int=150):
#     # Input: string
#     # Output: list
#     # Example: convert_text_to_unicode("Hom nay troi dep qua!")
#     # -> [72, 111, 109, 32, 110, 97, 121, 32, 116, 114, 111, 105, 32, 100, 101, 112, 32, 113, 117, 97, 33]
#     # ---
#     # Use:
#     # text = "Hom nay troi dep qua!"
#     # unicode_text = convert_text_to_unicode(text)
#     # print(unicode_text)
#     # # Result: [72, 111, 109, 32, 110, 97, 121, 32, 116, 114, 111, 105, 32, 100, 101, 112, 32, 113, 117, 97, 33]
#     # ---
#     # return [ord(char) for char in text]
#     result = [ord(char) for char in text]
#     if len(result) > token_size:
#         result = result[:token_size]
#     elif len(result) < token_size:
#         result += [0] * (token_size - len(result))  # 0 is the null character
#     return result

def test_remove_vietnamese_accent():
    assert remove_vietnamese_accent("Hôm nay trời đẹp quá!") == "Hom nay troi dep qua!"
    assert remove_vietnamese_accent("Tôi yêu Việt Nam") == "Toi yeu Viet Nam"
    print("PASSED: test_remove_vietnamese_accent()")

# def test_convert_text_to_unicode():
#     assert convert_text_to_unicode("Hom nay troi dep qua!") == [72, 111, 109, 32, 110, 97, 121, 32, 116, 114, 111, 105, 32, 100, 101, 112, 32, 113, 117, 97, 33] + [0] * 129
#     print("PASSED: test_convert_text_to_unicode()")

def test():
    test_remove_vietnamese_accent()
    # test_convert_text_to_unicode()

# test()

def main():
    # Remove Vietnamese accent from a text file and save the result to another file
    print("Starting remove Vietnamese accent from a text file and save the result to another file ...")
    file_to_process = "./data/raw/corpus-title.txt" # Output data for the model
    file_to_save_no_accent = "./data/processed/corpus-title-no-accent.txt"  # Input data for the model
    with open(file_to_process, "r", encoding="utf-8") as f_in, \
         open(file_to_save_no_accent, "w", encoding="utf-8") as f_out:
        for line in f_in:
            # Loại bỏ dấu tiếng Việt khỏi từng dòng
            no_accent_line = remove_vietnamese_accent(line.strip())
            f_out.write(no_accent_line + "\n")
    print("Done!")
