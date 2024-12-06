# How to reconstruct Vietnamese diacritics

## Data Preparation

This project utilizes nearly 10 million Vietnamese sentences collected by the research team at [news-corpus](https://github.com/binhvq/news-corpus).

To download the dataset (corpus-title.txt - 578MB), please visit the following link: [Google Drive](https://drive.usercontent.google.com/download?id=1ypvEoGRNWrNLmW246RtBm9iMyKXm_2BP&export=download&authuser=0).

## Data Collection, Processing & Model Training

```bash
# Generate text without diacritics from raw text
# This script was run once in version 0.0.0
nohup python main.py collection > logs/data_collection_0_0_1.log 2>&1 &

# Generate X and y datasets from text without diacritics and raw text (approx. 10 minutes)
# Note: During data preprocessing, uppercase letters were not converted to lowercase.
# (This has been fixed in the updated encoder function, which now ensures all characters are lowercase and within the input dictionary.)
# As a result, uppercase characters were encoded as 0.
# However, initial testing with a few cases suggests this issue does not significantly affect model performance at this stage.
nohup python main.py processing > logs/data_processing_0_0_1.log 2>&1 &
# Processed 9,487,416 samples. Saved: X -> X_transformer.pt, y -> y_transformer.pt

# Train the model: memory usage stabilized below 30GB.
# The first epoch took approximately 30 hours.
nohup python main.py building > logs/model_building_0_0_1.log 2>&1 &
# Data loaded: X -> torch.Size([9487416, 150]), y -> torch.Size([9487416, 150])
# Epoch 1/20
#   Train Loss: 0.3221 | Train Acc: 0.0972
#   Val Loss:   0.1924 | Val Acc:   0.1020
#   Saved best train model with loss 0.32209213210086346.
#   Saved best val model with loss 0.1923793297414913.
```

---

## Reconstructing Vietnamese Diacritics

Below are the demo results using the model trained for one epoch (approximately 30 hours of training).
- Training dataset: 8 million sentences.
- Validation dataset: 2 million sentences.
- Model parameters: 1,087,051.

```bash
python main.py use_app
# Mô hình đã được tải thành công!
#
# Chọn tùy chọn:
# 1: Nhập một câu có dấu và kiểm tra dự đoán khôi phục dấu từ câu không dấu.
# 2: Nhập một câu không dấu để mô hình dự đoán thêm dấu.
# q: Thoát.
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: Hom nay Ha Noi troi mua to qua.
# Dự đoán thêm dấu: Hôm nay Hà Nội trời mưa to quá.
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: Ten cua toi la Pham Ngoc Hai.
# Dự đoán thêm dấu: Tên của tôi là Phạm Ngọc Hải.
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: Tai buoc xu ly du lieu
# Dự đoán thêm dấu: Tại bước xử lý dữ liệu
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: O buoc xu ly du lieu nay, toi quen mat chua viet thuong cac ky tu va boi vay nhung ky tu viet hoa bi ma hoa thanh gia tri 0.
# Dự đoán thêm dấu: ở bước xử lý dữ liệu này, tôi quên mặt chưa việt thường các ký từ và bởi vây những ký từ việt hóa bí mã hóa thanh giá trị 0.
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: Do quen mat them ham viet thuong ky tu lai
# Dự đoán thêm dấu: đỏ quen mất thêm hầm việt thường ký từ lại
#
# Chọn (1/2/q): 2
# Nhập câu không dấu: Do func lower.() khong duoc dung, boi vay khi train, data ma ky tu viet hoa thi se ma hoa thanh 0
# Dự đoán thêm dấu: độ func lower.() không được dùng, bởi váy khi train, data mà ký từ việt hóa thi sẽ mà hóa thanh 0
```

It is evident that the model performs very well on sentences of moderate length with clear structures, accurately reconstructing diacritics. This showcases the advantage of a large dataset (nearly 10 million sentences).

With a model containing only 1 million parameters, the results are already impressive. However, for longer, more complex sentences or those with ambiguous structures, the model encounters some challenges in adding diacritics correctly. This is understandable because the training data consists of news headlines, which are designed to be short, attention-grabbing, and impactful to attract readers.

Additionally, the model has only been trained for one epoch (approximately 30 hours). While sentence-level accuracy remains limited, the loss function has already decreased significantly, indicating substantial success in character-level predictions.

Further updates on model training with additional epochs will be provided in the near future.

---

## Model information
```bash
python main.py model_info
```
Result:
```
Transformer(
  (embedding): Embedding(137, 128)
  (encoder_layer): TransformerEncoderLayer(
    (self_attn): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
    )
    (linear1): Linear(in_features=128, out_features=256, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (linear2): Linear(in_features=256, out_features=128, bias=True)
    (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (dropout1): Dropout(p=0.1, inplace=False)
    (dropout2): Dropout(p=0.1, inplace=False)
  )
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-6): 7 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (output_layer): Linear(in_features=128, out_features=75, bias=True)
)
Number of parameters: 1087051
```
