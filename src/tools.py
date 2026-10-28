import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

FEATURES = ["r_1", "r_2", "r_3", "r_4", "r_5", "ma5_gap", "ma20_gap", "mom10",
            "rsi14", "vol10", "hl_range", "oc_ret", "vol_chg"]
TRAIN_END = 2022             # huấn luyện cuối cùng: 2016-2022
TEST_START = 2023            # kiểm tra: 2023-2025
VAL_YEARS = [2020, 2021, 2022]   # các năm dùng để chọn tham số


def load_features(path="../data/process/vn30_features.csv"):
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def scale(X_train, X_other):
    #Chuẩn hóa về trung bình 0, độ lệch chuẩn 1. Chỉ học thang đo từ tập train."""
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_other)


def train_test(data):
    train = data.loc[:str(TRAIN_END)]
    test = data.loc[str(TEST_START):]
    X_train, X_test = scale(train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float))
    return X_train, train["y"].to_numpy(), X_test, test["y"].to_numpy(), train, test


def scores(y, pred, decision=None):
    out = {
        "Accuracy": accuracy_score(y, pred),
        "BalAcc": balanced_accuracy_score(y, pred),
        "F1": f1_score(y, pred, pos_label=1, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "UpRate": np.mean(pred == 1),        # tỷ lệ dự đoán "Tăng"
    }
    if decision is not None and len(np.unique(y)) == 2:
        out["AUC"] = roc_auc_score(y, decision)
    return out


def summarize(df, cols):
    #Gộp nhiều lần chạy thành chuỗi 'trung bình ± độ lệch chuẩn'."""
    g = df.groupby("model", sort=False)[cols]
    mean, std = g.mean(), g.std().fillna(0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.4f} ± {s:.4f}" if s > 0 else f"{m:.4f}" for m, s in zip(mean[c], std[c])]
    return out


def expand_grid(grids):
    #[{"C": [0.1, 1]}, ...] -> {"C": 0.1}, {"C": 1}, ... (mọi tổ hợp)."""
    for g in grids:
        keys = list(g)
        for values in itertools.product(*g.values()):
            yield dict(zip(keys, values))


def tune_walk_forward(ModelClass, grids, data, metric="BalAcc"):
    #Với mỗi bộ tham số: train trên mọi năm trước năm v, đo trên năm v (v = 2020, 2021, 2022),
    #lấy trung bình 3 năm. Trả về bộ tốt nhất và bảng điểm của mọi bộ."""
    folds = []
    for year in VAL_YEARS:
        tr = data.loc[:str(year - 1)]
        va = data.loc[str(year):str(year)]
        X_tr, X_va = scale(tr[FEATURES].to_numpy(float), va[FEATURES].to_numpy(float))
        folds.append((X_tr, tr["y"].to_numpy(), X_va, va["y"].to_numpy()))
    rows = []
    for params in expand_grid(grids):
        values = []
        for X_tr, y_tr, X_va, y_va in folds:
            model = ModelClass(**params).fit(X_tr, y_tr)
            values.append(scores(y_va, model.predict(X_va))[metric])
        rows.append({"params": params, metric: float(np.mean(values)), "std": float(np.std(values))})
    best = max(rows, key=lambda row: row[metric])
    return best["params"], rows


def evaluate_on_test(name, ModelClass, params, data, seeds=(0,)):
    #Train 2016-2022 với tham số đã chọn, đo trên test 2023-2025 (cả kỳ và từng năm)."""
    X_train, y_train, X_test, y_test, train, test = train_test(data)
    rows = []
    for seed in seeds:
        model = ModelClass(**params, seed=seed).fit(X_train, y_train)
        decision = model.decision_function(X_test)
        pred = np.where(decision >= 0, 1, -1)
        row = {"model": name, "seed": seed, **scores(y_test, pred, decision),
               "time_s": model.train_time_, "nonzero": model.nonzero_ratio_}
        for year in sorted(set(test.index.year)):
            mask = test.index.year == year
            row[f"BalAcc_{year}"] = scores(y_test[mask], pred[mask])["BalAcc"]
        rows.append(row)
    return pd.DataFrame(rows)


def linear_w(model, d):
    #Mô hình tuyến tính f(x) = w.x + b  =>  b = f(0),  w_j = f(e_j) - b.
    b = model.decision_function(np.zeros((1, d)))[0]
    w = model.decision_function(np.eye(d)) - b
    return w, b


def flip_near_boundary(y, r_next, rate):
    #Đảo nhãn `rate` phần các phiên có |return ngày mai| nhỏ nhất (nhãn mơ hồ quanh biên).
    y = y.copy()
    k = int(round(rate * len(y)))
    if k:
        idx = np.argsort(np.abs(r_next))[:k]
        y[idx] = -y[idx]
    return y


def add_outliers(X, y, rate, direction, shift=15.0, rng=None):
    #Thêm `rate` phần điểm ngoại lai: lấy ngẫu nhiên điểm train, đẩy xa `shift` đơn vị
    #theo hướng `direction` (hướng lớp Tăng) nhưng gán nhãn Giảm."""
    rng = rng or np.random.default_rng(0)
    k = int(round(rate * len(y)))
    if k == 0:
        return X, y
    d = direction / np.linalg.norm(direction)
    idx = rng.choice(len(y), k, replace=False)
    return np.vstack([X, X[idx] + shift * d]), np.r_[y, -np.ones(k, dtype=y.dtype)]
