#Bốn mô hình của chuỗi SVM -> LS-SVM -> Pin-SVM -> Wave-SVM.
#Cùng khung: min 1/2 ||w||^2 + C * sum c_i * L(u_i),  u_i = 1 - y_i f(x_i)."""
import time
import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import SVC

solvers.options["show_progress"] = False


def kernel_matrix(A, B, kernel="linear", gamma=0.1):
    #Ma trận K[i, j] = K(A_i, B_j). Tuyến tính: tích vô hướng; RBF: exp(-gamma ||a - b||^2)."""
    if kernel == "linear":
        return A @ B.T
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


def class_weights(y):
    #Trọng số cân bằng lớp: c_i = n / (2 * số mẫu cùng lớp với i)."""
    n = len(y)
    n_pos = max(np.sum(y > 0), 1)
    n_neg = max(np.sum(y < 0), 1)
    return np.where(y > 0, n / (2 * n_pos), n / (2 * n_neg))


class BaseModel:
    #Phần chung của 4 mô hình: tính trọng số lớp, bấm giờ, dự đoán bằng dấu của f(x)."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.cw_ = class_weights(y) if self.balanced else np.ones(len(y))
        start = time.perf_counter()
        self._train(X, y)                 # mỗi mô hình tự viết _train
        self.train_time_ = time.perf_counter() - start
        return self

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class SVM(BaseModel):
    #Baseline (Cortes & Vapnik 1995): hinge loss, giải đối ngẫu QP bằng SMO (thư viện libsvm)."""

    def __init__(self, C=1.0, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.seed, self.balanced = seed, balanced          # seed không dùng: SVM tất định

    def _train(self, X, y):
        self.model_ = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, tol=1e-3)
        self.model_.fit(X, y, sample_weight=self.cw_)
        self.nonzero_ratio_ = len(self.model_.support_) / len(y)   # tỷ lệ vector hỗ trợ

    def decision_function(self, X):
        return self.model_.decision_function(X)


class LSSVM(BaseModel):
    #Variant 1 (Suykens & Vandewalle 1999): squared loss + ràng buộc đẳng thức
    #-> giải một hệ tuyến tính (n+1) x (n+1), không cần lặp."""

    def __init__(self, C=1.0, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.kernel, self.gamma = C, kernel, gamma      # C đóng vai trò gamma của bài báo
        self.seed, self.balanced = seed, balanced

    def _train(self, X, y):
        n = len(y)
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        Omega = np.outer(y, y) * K                               # Omega_ij = y_i y_j K_ij
        A = np.zeros((n + 1, n + 1))
        A[0, 1:] = y                                             # hàng đầu: [0, y^T]
        A[1:, 0] = y                                             # cột đầu:  [0; y]
        A[1:, 1:] = Omega + np.diag(1.0 / (self.C * self.cw_))   # khối Omega + D^-1
        rhs = np.r_[0.0, np.ones(n)]                             # vế phải [0; 1]
        solution = np.linalg.solve(A, rhs)
        self.b_, self.alpha_ = solution[0], solution[1:]
        self.X_train_, self.y_train_ = X, y
        self.nonzero_ratio_ = np.mean(np.abs(self.alpha_) > 1e-8)

    def decision_function(self, X):
        K = kernel_matrix(X, self.X_train_, self.kernel, self.gamma)
        return K @ (self.alpha_ * self.y_train_) + self.b_      # f(x) = sum alpha_i y_i K(x_i, x) + b


class PinSVM(BaseModel):
    #Variant 2 (Huang, Shi & Suykens 2014): pinball loss.
    #Đối ngẫu giống SVM, chỉ đổi hộp 0 <= alpha <= C thành -tau*C <= alpha <= C. Giải QP bằng cvxopt."""

    def __init__(self, C=1.0, tau=0.5, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.tau, self.kernel, self.gamma = C, tau, kernel, gamma
        self.seed, self.balanced = seed, balanced

    def _train(self, X, y):
        n = len(y)
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        C_i = self.C * self.cw_
        P = matrix(np.outer(y, y) * K + 1e-8 * np.eye(n))     # cộng 1e-8 cho ma trận ổn định số
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([np.eye(n), -np.eye(n)]))        # lambda <= C_i  và  -lambda <= tau C_i
        h = matrix(np.r_[C_i, self.tau * C_i])
        A = matrix(y.reshape(1, -1))                           # sum lambda_i y_i = 0
        solution = solvers.qp(P, q, G, h, A, matrix(0.0))
        lam = np.ravel(solution["x"])

        # tính b từ các điểm nằm đúng trên lề (lambda nằm hẳn bên trong hộp)
        f_without_b = K @ (lam * y)
        tol = 1e-6 * self.C
        free = (lam > -self.tau * C_i + tol) & (lam < C_i - tol) & (np.abs(lam) > tol)
        idx = free if free.any() else np.abs(lam) > tol
        self.b_ = np.mean(y[idx] - f_without_b[idx])

        self.lam_, self.X_train_, self.y_train_ = lam, X, y
        self.nonzero_ratio_ = np.mean(np.abs(lam) > tol)

    def decision_function(self, X):
        K = kernel_matrix(X, self.X_train_, self.kernel, self.gamma)
        return K @ (self.lam_ * self.y_train_) + self.b_


def wave_loss(u, lam=1.0, a=1.5):
    return (1.0 / lam) * (1 - 1 / (1 + lam * u ** 2 * np.exp(np.clip(a * u, -50, 50))))


def wave_grad(u, lam=1.0, a=1.5):
    #Đạo hàm đúng: L'(u) = u e^{au} (2 + a u) / (1 + lam u^2 e^{au})^2.
    e = np.exp(np.clip(a * u, -50, 50))
    return u * e * (2 + a * u) / (1 + lam * u ** 2 * e) ** 2


class WaveSVM(BaseModel):
    #SOTA (Akhtar, Tanveer & Arshad 2024), theo Algorithm 1 của bài báo, tối ưu bằng Adam.
    #linear: theta = [w, b] trên x mở rộng [x, 1];  rbf: theta = gamma_j trong w = sum gamma_j phi(x_j)."""

    def __init__(self, C=1.0, lam=1.0, a=1.5, kernel="linear", gamma=0.1, k=32, lr=0.01,
                 max_iter=1000, beta1=0.9, beta2=0.999, eps=1e-6, tol=1e-6, seed=0,
                 balanced=True, init=None):
        self.C, self.lam, self.a, self.kernel, self.gamma = C, lam, a, kernel, gamma
        self.k, self.lr, self.max_iter = k, lr, max_iter
        self.beta1, self.beta2, self.eps, self.tol = beta1, beta2, eps, tol
        self.seed, self.balanced, self.init = seed, balanced, init

    def _features(self, X):
        #Linear: thêm cột 1 để gộp b vào w. RBF: mỗi điểm biểu diễn bằng K(x, x_j) với mọi x_j train.
        if self.kernel == "linear":
            return np.c_[X, np.ones(len(X))]
        return kernel_matrix(X, self.X_train_, self.kernel, self.gamma)

    def _train(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.X_train_ = X
        Z = self._features(X)
        p = Z.shape[1]
        theta = np.zeros(p) if self.init is None else np.asarray(self.init, float).copy()
        m, v = np.zeros(p), np.zeros(p)            # moment bậc 1, bậc 2 của Adam
        self.history_ = []
        for t in range(1, self.max_iter + 1):
            B = rng.choice(n, size=min(self.k, n), replace=False)     # lấy k mẫu ngẫu nhiên
            xi = 1 - y[B] * (Z[B] @ theta)                              # eq. (25)
            g_loss = (self.cw_[B] * y[B] * wave_grad(xi, self.lam, self.a)) @ Z[B]
            reg = theta if self.kernel == "linear" else Z @ theta      # gradient của phần điều hòa
            g = reg - self.C * g_loss                                   # eq. (16)/(26)
            m = self.beta1 * m + (1 - self.beta1) * g                   # eq. (17)
            v = self.beta2 * v + (1 - self.beta2) * g ** 2              # eq. (18)
            m_hat = m / (1 - self.beta1 ** t)                           # eq. (19)
            v_hat = v / (1 - self.beta2 ** t)                           # eq. (20)
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            theta = theta - step                                        # eq. (21)
            if t % 50 == 0:
                self.history_.append((t, self.objective(Z, y, theta)))
            if np.linalg.norm(step) < self.tol:                         # điều kiện dừng
                break
        self.theta_, self.n_iter_ = theta, t
        self.nonzero_ratio_ = np.mean(np.abs(theta) > 1e-8) if self.kernel != "linear" else np.nan

    def objective(self, Z, y, theta):
        reg = 0.5 * theta @ theta if self.kernel == "linear" else 0.5 * theta @ (Z @ theta)
        return reg + self.C * (self.cw_ * wave_loss(1 - y * (Z @ theta), self.lam, self.a)).sum()

    def decision_function(self, X):
        return self._features(X) @ self.theta_
