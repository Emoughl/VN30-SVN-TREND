# Triển khai 4 mô hình SVM cho bài toán dự báo
import time
import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import SVC

solvers.options["show_progress"] = False

def kernel_matrix(A, B, kernel="linear", gamma=0.1):
    if kernel == "linear":
        return A @ B.T
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))

def class_weights(y):
    n = len(y)
    n_pos = max(np.sum(y > 0), 1)
    n_neg = max(np.sum(y < 0), 1)
    return np.where(y > 0, n / (2 * n_pos), n / (2 * n_neg))

class BaseModel:
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.cw_ = class_weights(y) if self.balanced else np.ones(len(y))
        start = time.perf_counter()
        self._train(X, y)
        self.train_time_ = time.perf_counter() - start
        return self

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

class SVM(BaseModel):
    def __init__(self, C=1.0, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.kernel, self.gamma, self.seed, self.balanced = C, kernel, gamma, seed, balanced

    def _train(self, X, y):
        self.model_ = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)
        self.model_.fit(X, y, sample_weight=self.cw_)
        self.nonzero_ratio_ = len(self.model_.support_) / len(y)

    def decision_function(self, X):
        return self.model_.decision_function(X)

class LSSVM(BaseModel):
    def __init__(self, C=1.0, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.kernel, self.gamma, self.seed, self.balanced = C, kernel, gamma, seed, balanced

    def _train(self, X, y):
        n = len(y)
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        Omega = np.outer(y, y) * K
        A = np.zeros((n + 1, n + 1))
        A[0, 1:], A[1:, 0] = y, y
        A[1:, 1:] = Omega + np.diag(1.0 / (self.C * self.cw_))
        solution = np.linalg.solve(A, np.r_[0.0, np.ones(n)])
        self.b_, self.alpha_ = solution[0], solution[1:]
        self.X_train_, self.y_train_ = X, y
        self.nonzero_ratio_ = np.mean(np.abs(self.alpha_) > 1e-8)

    def decision_function(self, X):
        K = kernel_matrix(X, self.X_train_, self.kernel, self.gamma)
        return K @ (self.alpha_ * self.y_train_) + self.b_

class PinSVM(BaseModel):
    def __init__(self, C=1.0, tau=0.5, kernel="linear", gamma=0.1, seed=0, balanced=True):
        self.C, self.tau, self.kernel, self.gamma, self.seed, self.balanced = C, tau, kernel, gamma, seed, balanced

    def _train(self, X, y):
        n = len(y)
        K = kernel_matrix(X, X, self.kernel, self.gamma)
        C_i = self.C * self.cw_
        P = matrix(np.outer(y, y) * K + 1e-8 * np.eye(n))
        q = matrix(-np.ones(n))
        G = matrix(np.vstack([np.eye(n), -np.eye(n)]))
        h = matrix(np.r_[C_i, self.tau * C_i])
        solution = solvers.qp(P, q, G, h, matrix(y.reshape(1, -1)), matrix(0.0))
        lam = np.ravel(solution["x"])
        self.b_ = np.mean(y - (K @ (lam * y))) 
        self.lam_, self.X_train_, self.y_train_ = lam, X, y
        self.nonzero_ratio_ = np.mean(np.abs(lam) > 1e-6)

    def decision_function(self, X):
        K = kernel_matrix(X, self.X_train_, self.kernel, self.gamma)
        return K @ (self.lam_ * self.y_train_) + self.b_

def wave_loss(u, lam=1.0, a=1.5):
    return (1.0 / lam) * (1 - 1 / (1 + lam * u ** 2 * np.exp(np.clip(a * u, -50, 50))))

def wave_grad(u, lam=1.0, a=1.5):
    # Đạo hàm chuẩn theo bài báo: d(loss)/du
    e = np.exp(np.clip(a * u, -50, 50))
    return u * e * (2 + a * u) / (1 + lam * u ** 2 * e) ** 2

class WaveSVM(BaseModel):
    def __init__(self, C=1.0, lam=1.0, a=1.5, kernel="linear", gamma=0.1, k=32, lr=0.01,
                 max_iter=1000, beta1=0.9, beta2=0.999, eps=1e-6, tol=1e-6, seed=0,
                 balanced=True, init=None):
        self.C, self.lam, self.a, self.kernel, self.gamma, self.k, self.lr, self.max_iter = C, lam, a, kernel, gamma, k, lr, max_iter
        self.beta1, self.beta2, self.eps, self.tol, self.seed, self.balanced, self.init = beta1, beta2, eps, tol, seed, balanced, init

    def _features(self, X, X_train=None):
        if self.kernel == "linear": 
            return np.c_[X, np.ones(len(X))]
        return kernel_matrix(X, X_train, self.kernel, self.gamma)

    def objective(self, Z, y, theta):
        reg = 0.5 * theta @ theta if self.kernel == "linear" else 0.5 * theta @ (Z @ theta)
        return reg + self.C * (self.cw_ * wave_loss(1 - y * (Z @ theta), self.lam, self.a)).sum()

    def _train(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.X_train_ = X
        
        # Tính Z cho toàn bộ tập train (nếu linear thì nhanh, nếu RBF thì tốn RAM)
        Z = self._features(X, X) 
        theta = np.zeros(Z.shape[1]) if self.init is None else np.asarray(self.init, float).copy()
        m, v = np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
        self.history_ = []

        for t in range(1, self.max_iter + 1):
            # [SỬA LỆCH 1]: Lấy mini-batch ngẫu nhiên MỖI VÒNG LẶP
            B = rng.choice(n, size=min(self.k, n), replace=False)
            
            # Tính u cho batch hiện tại
            u = 1 - y[B] * (Z[B] @ theta) 
            
            # [SỬA LỆCH 4]: Cập nhật Gradient dựa trên u mới nhất
            # g = grad(Reg) + C * sum(cw * grad(loss) * grad(u))
            # grad(u) = -y * Z
            grad_loss = wave_grad(u, self.lam, self.a)
            
            # Regularization gradient
            g_reg = theta if self.kernel == "linear" else Z @ theta
            
            # Loss gradient: C * sum_{i in B} ( cw_i * grad_loss_i * (-y_i * Z_i) )
            g_loss = -self.C * (self.cw_[B] * grad_loss * y[B]) @ Z[B]
            
            g = g_reg + g_loss
            
            # Adam Optimizer
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            theta -= step
            
            if t % 50 == 0:
                self.history_.append((t, self.objective(Z, y, theta)))
            
            if np.linalg.norm(step) < self.tol: break
            
        self.theta_, self.n_iter_ = theta, t
        self.nonzero_ratio_ = np.mean(np.abs(theta) > 1e-8) if self.kernel != "linear" else np.nan

    def decision_function(self, X):
        # Khi predict, X phải được chuyển đổi qua cùng kernel với train
        Z_test = self._features(X, self.X_train_)
        return Z_test @ self.theta_
