# Quy trình

### **1. Chuẩn bị dữ liệu**
#### **1.1. Thu thập dữ liệu**
- **Nguồn dữ liệu**:
  - Sách, báo trực tuyến, tài liệu học thuật, diễn đàn, hoặc các tập dữ liệu tiếng Việt có sẵn như VNTQ Corpus.
  - Các dữ liệu này cần có đầy đủ dấu tiếng Việt.
- **Tiền xử lý**:
  - Loại bỏ các ký tự không liên quan (emoji, ký hiệu lạ).
  - Đưa văn bản về dạng câu chuẩn hóa.
  - Tách câu thành từng cặp:
    - **Đầu vào**: Loại bỏ toàn bộ dấu tiếng Việt từ câu gốc (ví dụ, "Hôm nay tôi đi học muộn." thành "Hom nay toi di hoc muon.").
    - **Đầu ra**: Giữ nguyên câu gốc có dấu.

#### **1.2. Chuẩn hóa dữ liệu**
- Loại bỏ các câu quá ngắn hoặc quá dài (nên giới hạn khoảng 5–120 ký tự). Or lấy max độ dài của 1 câu trong dữ liệu.
- Đảm bảo số lượng từ vựng đủ lớn, bao gồm cả tên riêng và từ đặc thù trong tiếng Việt.
- Tạo tập dữ liệu huấn luyện, kiểm tra, và thử nghiệm với tỷ lệ ví dụ 60:20:20. Do dữ liệu rất nhiều nên để tập train ít đi tý cho đỡ quá trình huấn luyện lại.

#### **1.3. Tokenization**
- Với tiếng Việt, mỗi từ có thể được biểu diễn dưới dạng ký tự (character-based) hoặc thành phần nhỏ hơn như âm tiết hoặc âm vị (subsyllable-based).
- Biểu diễn câu như chuỗi ký tự:
  ```
  Input: ["H", "o", "m", " ", "n", "a", "y", " ", "t", "o", "i", ...]
  Output: ["H", "ô", "m", " ", "n", "a", "y", " ", "t", "ô", "i", ...]
  ```

---

### **2. Xây dựng mô hình**
#### **2.1. Kiến trúc mô hình**
- **Lựa chọn mô hình**:
  - **Bidirectional RNN** (GRU/LSTM): Phù hợp nếu tài nguyên hạn chế và chuỗi đầu vào không quá dài.
  - **Transformer**: Hiệu quả với chuỗi dài, đặc biệt nếu cần xử lý cú pháp phức tạp. (Ví dụ: Sử dụng kiến trúc tương tự BERT hoặc mBART).

#### **2.2. Các bước triển khai**
- **Embedding layer**: Biểu diễn các ký tự đầu vào thành vector.
- **Encoder**:
  - Nếu sử dụng RNN: Multi-layer Bidirectional GRU/LSTM để mã hóa ngữ cảnh hai chiều.
  - Nếu sử dụng Transformer: Multi-head self-attention để trích xuất ngữ cảnh.
- **Decoder**:
  - Mạng dense với softmax để dự đoán ký tự có dấu tương ứng.

#### **2.3. Loss function**
- Sử dụng **categorical cross-entropy loss** cho từng ký tự.

#### **2.4. Optimizer**
- Sử dụng **Adam optimizer** với learning rate decay.

---

### **3. Huấn luyện mô hình**
#### **3.1. Quá trình huấn luyện**
- Đầu vào: Chuỗi không dấu.
- Đầu ra: Chuỗi có dấu.
- Sử dụng mini-batch gradient descent.
- Áp dụng regularization (dropout, layer normalization) để giảm overfitting.

#### **3.2. Lưu ý**
- Theo dõi các chỉ số như **accuracy**, **character-level F1-score** trên tập kiểm tra.

---

### **4. Kiểm tra và đánh giá**
#### **4.1. Trên tập thử nghiệm**
- Sử dụng các chỉ số:
  - **Character accuracy**: Tỷ lệ ký tự được gán đúng dấu.
  - **Sentence accuracy**: Tỷ lệ câu khớp hoàn toàn với câu gốc.
- Đánh giá trên tập thử nghiệm để tránh hiện tượng overfitting.

#### **4.2. Phân tích lỗi**
- Kiểm tra các trường hợp khó: tên riêng, từ đa nghĩa, các câu không rõ ngữ cảnh.
- Phân tích các vị trí ký tự dự đoán sai để điều chỉnh dữ liệu huấn luyện hoặc mô hình.

---

### **5. Triển khai**
- Kết hợp mô hình với một hệ thống API hoặc ứng dụng thực tế.
- Tối ưu hóa thời gian suy luận (inference time) để đảm bảo tốc độ nhanh, sử dụng framework như TensorFlow Lite hoặc ONNX.

---

### **6. Cải tiến (tùy chọn)**
- Áp dụng **ngôn ngữ mô hình (language model)** như GPT hoặc BERT được fine-tune trên tiếng Việt để sửa các lỗi ngữ pháp sau khi phục hồi dấu.
- Sử dụng kỹ thuật augmentation để tạo thêm dữ liệu từ tập hiện có.
