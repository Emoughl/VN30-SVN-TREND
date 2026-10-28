#Chuyển nguyên trạng Wave_Adam_function.m (github.com/mtanveer1/Wave-SVM) sang Python.
#Giữ nguyên các điểm lệch so với bài báo, đánh dấu [LỆCH x]:
#  [LỆCH 1] chỉ lấy m mẫu ngẫu nhiên MỘT lần, mô hình chỉ học trên m mẫu đó
#  [LỆCH 2] u = y*q - 1, ngược dấu với xi = 1 - y f(x) của bài báo
#  [LỆCH 3] đạo hàm dùng (2 + u) thay vì (2 + a*u)
#  [LỆCH 4] ma trận E (phần gradient của loss) tính một lần TRƯỚC vòng lặp, không cập nhật
#  [LỆCH 5] kernel lúc test exp(-D/2*mew^2) khác lúc train exp(-D/(2*mew^2)), và nhân thêm y_j

import time
import numpy as np


def wave_adam_official(Xtrain, ytrain, Xtest, a=1.5, b=1.0, C=1.0, mew=1.0, m=32,
                       max_iter=1000, beta1=0.9, beta2=0.999, alpha=0.01,
                       epsilon=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    l = len(ytrain)
    idx = rng.permutation(l)[:m]                                   # [LỆCH 1]
    xrand, yrand = Xtrain[idx], ytrain[idx].astype(float)

    XX = (xrand ** 2).sum(1)[:, None] * np.ones((1, m))
    omega = XX + XX.T - 2 * xrand @ xrand.T
    omega = np.exp(-omega / (2 * mew ** 2))

    gamma = 0.01 * np.ones(m)
    r = 0.01 * np.ones(m)
    v = 0.01 * np.ones(m)

    q = np.array([np.sum(gamma * omega[:, i]) for i in range(m)])
    u = yrand * q - 1                                              # [LỆCH 2]

    E = np.zeros((m, m))
    for i in range(m):                                             # [LỆCH 3], [LỆCH 4]
        E[i, :] = ((u[i] * np.exp(a * u[i]) * yrand[i] * (2 + u[i]))
                   / (1 + b * u[i] * u[i] * np.exp(a * u[i])) ** 2) * omega[i, :]

    start = time.perf_counter()
    t = 0
    for _ in range(max_iter):
        t += 1
        gradient = omega @ gamma - C * E.sum(0)
        r = beta1 * r + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2
        r_hat = r / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        gamma = gamma - alpha * r_hat / (np.sqrt(v_hat) + epsilon)

    p = len(Xtest)
    omega1 = -2 * xrand @ Xtest.T
    XK = (xrand ** 2).sum(1)[:, None] * np.ones((1, p))
    Xt = (Xtest ** 2).sum(1)[:, None] * np.ones((1, m))
    omega1 = omega1 + XK + Xt.T
    omega1 = np.exp(-omega1 / 2 * mew ** 2)                        # [LỆCH 5]
    HT = omega1 * yrand[:, None]                                   # [LỆCH 5]
    score = HT.T @ gamma
    return score, time.perf_counter() - start
