import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import zipfile
import os

SEED = 42

#=================================================
# Tiền xử lý dữ liệu Credit Card Fraud Detection.
#=================================================

def preprocess_credit_card_data(df, val_size=0.1):
    """
    Tiền xử lý dữ liệu Credit Card Fraud Detection.
    """
    print("1. Tách Features (X) và Target (y)...")
    X = df.drop("Class", axis=1)
    y = df["Class"]
 
    print("2. Chia tập Train+Val / Test (80/20) với Stratify...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
 
    # val_size gốc là % trên toàn data, cần quy đổi sang % trên train+val
    # ví dụ: val=10% toàn data, train+val=80% → val chiếm 10/80 = 12.5% của train+val
    val_size_adjusted = val_size / (1 - 0.2)
    print(f"3. Chia tập Train / Val ({round((0.8 - val_size)*100)}/{round(val_size*100)}) với Stratify...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=SEED,
        stratify=y_trainval
    )
 
    print("4. Áp dụng RobustScaler cho Time và Amount (fit trên train)...")
    scaler_amount = RobustScaler()
    scaler_time   = RobustScaler()
 
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()
 
    # fit chỉ trên train
    X_train["scaled_amount"] = scaler_amount.fit_transform(X_train["Amount"].values.reshape(-1, 1))
    X_train["scaled_time"]   = scaler_time.fit_transform(X_train["Time"].values.reshape(-1, 1))
 
    # transform val và test
    X_val["scaled_amount"]  = scaler_amount.transform(X_val["Amount"].values.reshape(-1, 1))
    X_val["scaled_time"]    = scaler_time.transform(X_val["Time"].values.reshape(-1, 1))
 
    X_test["scaled_amount"] = scaler_amount.transform(X_test["Amount"].values.reshape(-1, 1))
    X_test["scaled_time"]   = scaler_time.transform(X_test["Time"].values.reshape(-1, 1))
 
    X_train.drop(["Time", "Amount"], axis=1, inplace=True)
    X_val.drop(["Time", "Amount"],   axis=1, inplace=True)
    X_test.drop(["Time", "Amount"],  axis=1, inplace=True)
 
    print(f"\nKích thước các tập:")
    print(f"  Train : {X_train.shape} | Fraud rate: {y_train.mean()*100:.3f}%")
    print(f"  Val   : {X_val.shape}   | Fraud rate: {y_val.mean()*100:.3f}%")
    print(f"  Test  : {X_test.shape}  | Fraud rate: {y_test.mean()*100:.3f}%")
 
    return (
        X_train.values, X_val.values, X_test.values,
        y_train.values, y_val.values, y_test.values
    )
 
 
def apply_smote(X_train, y_train, sampling_strategy=1.0, k_neighbors=5):
    """
    Áp dụng SMOTE để cân bằng tập train.
    """
    smote = SMOTE(
        random_state=SEED,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
 
    print(f"Sau SMOTE — Train: {X_resampled.shape} | Fraud rate: {y_resampled.mean()*100:.1f}%")
    return X_resampled, y_resampled

#=================================================
# Unzip file csv.
#=================================================

def load_creditcard_csv(data_dir='../data'):
    """
    Đọc creditcard.csv từ data_dir.
    Nếu chỉ có file .zip thì tự giải nén rồi đọc.
    """
    csv_path = os.path.join(data_dir, 'creditcard.csv')
    zip_path = os.path.join(data_dir, 'creditcard.csv.zip')

    if not os.path.exists(csv_path):
        print("Không tìm thấy CSV, đang giải nén ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract('creditcard.csv', data_dir)
        print("Unzip done!")

    return pd.read_csv(csv_path)

#=================================================
# Chọn threshold tối ưu.
#=================================================


def select_threshold_from_pr_curve(y_true, y_prob, min_precision=0.6):
    """
    Chọn threshold tối ưu từ Precision-Recall curve.
    Mục tiêu:
    1. Chỉ xét các threshold thỏa precision >= min_precision
    2. Trong tập feasible, lấy recall tốt nhất
    3. Trong nhóm có recall tốt nhất, chọn F1 cao nhất
    4. Nếu vẫn hòa, chọn precision cao nhất
    5. Nếu vẫn hòa, chọn threshold cao nhất
    Nếu không có điểm nào thỏa, trả về no_feasible_threshold.
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
 
    if thresholds.size == 0:
        return {
            'threshold'              : None,
            'precision'              : np.nan,
            'recall'                 : np.nan,
            'f1'                     : np.nan,
            'selection_note'         : 'no_thresholds',
            'num_feasible_thresholds': 0,
        }
 
    # precision_recall_curve trả về n+1 điểm, bỏ điểm cuối
    precision_pts  = precision_arr[:-1]
    recall_pts     = recall_arr[:-1]
    threshold_pts  = thresholds
    feasible_idx   = np.where(precision_pts >= min_precision)[0]
 
    if feasible_idx.size == 0:
        return {
            'threshold': None,
            'precision': np.nan,
            'recall': np.nan,
            'f1': np.nan,
            'selection_note': 'no_feasible_threshold',
            'num_feasible_thresholds': 0,
        }
 
    f1_pts = np.divide(
        2 * precision_pts * recall_pts,
        precision_pts + recall_pts,
        out=np.zeros_like(precision_pts),
        where=(precision_pts + recall_pts) > 0,
    )

    best_recall = recall_pts[feasible_idx].max()
    recall_best_idx = feasible_idx[recall_pts[feasible_idx] == best_recall]

    best_f1 = f1_pts[recall_best_idx].max()
    f1_best_idx = recall_best_idx[f1_pts[recall_best_idx] == best_f1]

    best_precision = precision_pts[f1_best_idx].max()
    precision_best_idx = f1_best_idx[precision_pts[f1_best_idx] == best_precision]

    best_idx = precision_best_idx[np.argmax(threshold_pts[precision_best_idx])]

    chosen_threshold = float(threshold_pts[best_idx])
    chosen_pred      = (y_prob >= chosen_threshold).astype(int)
 
    return {
        'threshold'              : chosen_threshold,
        'precision'              : float(precision_score(y_true, chosen_pred, zero_division=0)),
        'recall'                 : float(recall_score(y_true, chosen_pred, zero_division=0)),
        'f1'                     : float(f1_score(y_true, chosen_pred, zero_division=0)),
        'selection_note'         : 'feasible_threshold',
        'num_feasible_thresholds': int(feasible_idx.size),
    }
