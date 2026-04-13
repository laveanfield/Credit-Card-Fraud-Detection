# Credit Card Fraud Detection — ML Assignment

Project này triển khai và so sánh bốn hướng tiếp cận cho bài toán phát hiện gian lận thẻ tín dụng trên bộ dữ liệu Credit Card Fraud Detection:

- `Logistic Regression` làm baseline học máy truyền thống
- `MLP` đại diện cho hướng deep learning
- `XGBoost` và `LightGBM` đại diện cho nhóm gradient boosting

Các notebook được tổ chức theo pipeline rõ ràng: tiền xử lý chung, huấn luyện từng mô hình, sau đó so sánh tổng hợp bằng cùng tập test và cùng bộ metric.

## Cấu trúc thư mục

```text
Assignment_ML/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   └── creditcard.csv              # tải thủ công từ Kaggle
│
├── notebooks/
│   ├── 1_EDA_and_Preprocessing.ipynb
│   ├── 2_logistic_regression.ipynb
│   ├── 3_gradient_boosting.ipynb
│   ├── 4_MLP.ipynb
│   ├── 5_comparison.ipynb
│   └── credit_utils/
│       └── utils.py                # hàm dùng chung cho preprocessing, SMOTE, threshold
│
├── notebooks - Copy/               # bản lưu/snapshot tham chiếu của notebook
│
├── features/                       # xác suất dự đoán, summary và artifact trung gian
├── models/                         # model đã train và metadata để tránh retrain
├── outputs/                        # hình ảnh/biểu đồ export phục vụ báo cáo
│
└── .venv/                          # môi trường ảo cục bộ (nếu có)
```

## Ý nghĩa ngắn của các notebook

- `1_EDA_and_Preprocessing.ipynb`
  - Khám phá dữ liệu, phân tích imbalance và chuẩn hóa cách chia tập.
- `2_logistic_regression.ipynb`
  - Huấn luyện baseline Logistic Regression, so sánh các nhánh imbalance handling và lưu `lr_y_prob.npy`.
- `3_gradient_boosting.ipynb`
  - Huấn luyện XGBoost và LightGBM, lưu xác suất đầu ra để so sánh tổng hợp.
- `4_MLP.ipynb`
  - Huấn luyện các kiến trúc MLP, chọn mô hình tốt nhất theo validation và lưu `nn_y_prob.npy`.
- `5_comparison.ipynb`
  - Nạp các artifact từ `features/`, dựng lại `y_test` chung và so sánh toàn bộ mô hình bằng bảng metric và biểu đồ.

## Các thư mục artifact

- `features/`
  - Chứa các file xác suất dự đoán như `lr_y_prob.npy`, `nn_y_prob.npy`, `xgb_y_prob.npy`, `lgb_y_prob.npy`
  - Có thể kèm các file summary như validation results hoặc lịch sử huấn luyện
- `models/`
  - Chứa model đã train (`.joblib`, `.keras`) và file metadata (`.json`)
  - Được dùng để load lại kết quả, giảm việc phải train lại từ đầu
- `outputs/`
  - Chứa ảnh đã export từ notebook để chèn vào báo cáo

## Cách chạy

### 1. Tạo và kích hoạt môi trường ảo

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

### 3. Chuẩn bị dữ liệu

Tải dataset `creditcard.csv` từ Kaggle:

- Link: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Đặt file vào thư mục:

```text
data/creditcard.csv
```

### 4. Chạy notebook theo thứ tự

Thứ tự khuyến nghị:

1. `notebooks/1_EDA_and_Preprocessing.ipynb`
2. `notebooks/2_logistic_regression.ipynb`
3. `notebooks/3_gradient_boosting.ipynb`
4. `notebooks/4_MLP.ipynb`
5. `notebooks/5_comparison.ipynb`

Nếu chỉ cần xem so sánh cuối cùng, có thể chạy trực tiếp `5_comparison.ipynb` sau khi các notebook huấn luyện đã tạo đủ artifact trong `features/` và `models/`.

## Lưu ý khi chạy

- Các notebook huấn luyện sử dụng chung helper trong `notebooks/credit_utils/utils.py`.
- Một số notebook có cơ chế load lại model và metadata đã lưu để tránh retrain không cần thiết.
- `5_comparison.ipynb` không huấn luyện lại mô hình; notebook này chỉ dùng các xác suất đầu ra đã được lưu từ các notebook trước.
- Với bài toán mất cân bằng mạnh, kết luận chính nên ưu tiên các metric như `PR-AUC`, `ROC-AUC`, `Precision`, `Recall` và `F1-score` thay vì chỉ nhìn `Accuracy`.

## Kết quả đầu ra chính

Sau khi chạy đầy đủ pipeline, các file thường cần có gồm:

- `features/lr_y_prob.npy`
- `features/nn_y_prob.npy`
- `features/xgb_y_prob.npy`
- `features/lgb_y_prob.npy`

và các model tương ứng trong thư mục `models/`.
