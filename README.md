# VN30-SVN-TREND

## Mục tiêu đề tài
Nghiên cứu cơ sở toán học và triển khai các biến thể của Support Vector Machine (SVM) để dự đoán xu hướng giá cổ phiếu (Tăng/Giảm) của chỉ số VN30.

## Các mô hình triển khai
- **Baseline:** SVM (Standard Hinge Loss)
- **Variant 1:** LS-SVM (Least Squares SVM)
- **Variant 2:** Pin-SVM (Pinball Loss)
- **SOTA:** Wave-SVM (Wave Loss - Tối ưu hóa bằng Adam)

## Hướng dẫn sử dụng
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy lần lượt các Notebook từ `notebooks/00` $\rightarrow$ `notebooks/07`.
   - Lưu ý: Notebook 05 (Pin-SVM) mất khoảng 5–6 phút để chạy.
3. Dữ liệu thô có sẵn tại `data/raw`; chỉ chạy NB01 nếu muốn tải lại dữ liệu từ SSI iBoard.

## Ghi chú về mã nguồn
- **src/models.py:** Chứa triển khai chuẩn của 4 mô hình dựa trên lý thuyết bài báo.
- **src/wave_svm_official.py:** Bản mô phỏng lại code từ GitHub chính thức. Lưu ý rằng bản này có một số điểm lệch so với lý thuyết bài báo (đã được ghi chú chi tiết trong code) và được dùng làm cơ sở đối chiếu để xây dựng bản chuẩn trong `models.py`.
