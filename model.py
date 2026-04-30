import pickle
from numbers import Integral

import numpy as np


class ReLU:
    """ReLU 激活层。"""

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, dout):
        return dout * (self.x > 0).astype(self.x.dtype, copy=False)


class Sigmoid:
    """Sigmoid 激活层。"""

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        self.out = out
        return self.out

    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)


class Tanh:
    """Tanh 激活层。"""

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)


class Dropout:
    """Dropout 正则化层。"""

    def __init__(self, rate, seed=None):
        self.rate = rate
        self.keep_prob = 1.0 - rate
        self.mask = None
        self.random_state = np.random.RandomState(seed)

    def forward(self, x, training=False):
        if not training or self.rate <= 0.0:
            self.mask = None
            return x

        # 使用反向缩放的 Dropout，训练和推理阶段量级保持一致。
        self.mask = (
            self.random_state.binomial(1, self.keep_prob, size=x.shape).astype(np.float64)
            / self.keep_prob
        )
        return x * self.mask

    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask


class Linear:
    """全连接线性层。"""

    def __init__(self, in_features, out_features, activation_name="relu", seed=None):
        rs = np.random.RandomState(seed)

        # 按激活函数选择初始化缩放系数。
        scale = (
            np.sqrt(2.0 / in_features)
            if activation_name == "relu"
            else np.sqrt(2.0 / (in_features + out_features))
        )
        self.W = rs.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features), dtype=np.float64)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, weight_decay=0.0):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)

        if weight_decay > 0.0:
            self.dW += weight_decay * self.W

        return dout @ self.W.T


class CrossEntropyLoss:
    """带可选 L2 正则项的交叉熵损失。"""

    def __init__(self, weight_decay=0.0):
        self.weight_decay = weight_decay
        self.y_pred = None
        self.y_true_onehot = None

    def forward(self, logits, y_true, model=None):
        n_samples = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        if y_true.ndim == 1:
            self.y_true_onehot = np.zeros_like(logits)
            self.y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
        else:
            self.y_true_onehot = y_true

        clipped = np.clip(self.y_pred, 1e-12, 1.0)
        loss = float(-np.sum(self.y_true_onehot * np.log(clipped)) / n_samples)

        if self.weight_decay > 0.0 and model is not None:
            l2_loss = sum(
                np.sum(layer.W**2) for layer in model.layers if isinstance(layer, Linear)
            )
            loss += 0.5 * self.weight_decay * l2_loss

        return loss

    def backward(self):
        return (self.y_pred - self.y_true_onehot) / self.y_pred.shape[0]


class SGD:
    """支持可选动量的随机梯度下降优化器。"""

    def __init__(self, lr=1e-3, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity_W = {}
        self.velocity_b = {}

    def step(self, model):
        for layer in model.layers:
            if isinstance(layer, Linear):
                lid = id(layer)

                if lid not in self.velocity_W:
                    self.velocity_W[lid] = np.zeros_like(layer.W)
                    self.velocity_b[lid] = np.zeros_like(layer.b)

                if self.momentum > 0:
                    self.velocity_W[lid] = (
                        self.momentum * self.velocity_W[lid] - self.lr * layer.dW
                    )
                    self.velocity_b[lid] = (
                        self.momentum * self.velocity_b[lid] - self.lr * layer.db
                    )
                    layer.W += self.velocity_W[lid]
                    layer.b += self.velocity_b[lid]
                else:
                    layer.W -= self.lr * layer.dW
                    layer.b -= self.lr * layer.db


class Adam:
    """Adam 优化器实现。"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def step(self, model):
        self.t += 1
        for layer in model.layers:
            if isinstance(layer, Linear):
                lid = id(layer)

                if lid not in self.m_W:
                    self.m_W[lid] = np.zeros_like(layer.W)
                    self.v_W[lid] = np.zeros_like(layer.W)
                    self.m_b[lid] = np.zeros_like(layer.b)
                    self.v_b[lid] = np.zeros_like(layer.b)

                self.m_W[lid] = self.beta1 * self.m_W[lid] + (1.0 - self.beta1) * layer.dW
                self.m_b[lid] = self.beta1 * self.m_b[lid] + (1.0 - self.beta1) * layer.db
                self.v_W[lid] = self.beta2 * self.v_W[lid] + (1.0 - self.beta2) * (
                    layer.dW**2
                )
                self.v_b[lid] = self.beta2 * self.v_b[lid] + (1.0 - self.beta2) * (
                    layer.db**2
                )

                m_hat_W = self.m_W[lid] / (1.0 - self.beta1**self.t)
                v_hat_W = self.v_W[lid] / (1.0 - self.beta2**self.t)
                m_hat_b = self.m_b[lid] / (1.0 - self.beta1**self.t)
                v_hat_b = self.v_b[lid] / (1.0 - self.beta2**self.t)

                layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
                layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


class MLP:
    """模块化的多层感知机，支持可变层数、激活函数和 Dropout。"""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation_name="relu",
        dropout_rates=None,
        seed=42,
    ):
        self.input_mean = None
        self.input_std = None
        self.layers = []
        self.activation_name = activation_name.lower()

        # 解析隐藏层配置。
        if hidden_dims is None:
            hidden_dims = []
        elif isinstance(hidden_dims, Integral):
            hidden_dims = [hidden_dims]

        sizes = [input_dim] + list(hidden_dims) + [output_dim]
        hidden_layer_count = len(sizes) - 2

        # 解析每个隐藏层对应的 Dropout 比例。
        if dropout_rates is None:
            dr_rates = [0.0] * hidden_layer_count
        elif isinstance(dropout_rates, (float, int)):
            dr_rates = [float(dropout_rates)] * hidden_layer_count
        else:
            dr_rates = [float(r) for r in dropout_rates]

        # 按“线性层 -> 激活层 -> Dropout”的顺序动态构建网络。
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            self.layers.append(Linear(fan_in, fan_out, self.activation_name, seed + i))

            if i < hidden_layer_count:
                if self.activation_name == "relu":
                    self.layers.append(ReLU())
                elif self.activation_name == "sigmoid":
                    self.layers.append(Sigmoid())
                elif self.activation_name == "tanh":
                    self.layers.append(Tanh())

                if dr_rates[i] > 0.0:
                    self.layers.append(Dropout(dr_rates[i], seed + i))

    def set_input_normalization(self, mean, std):
        """设置输入标准化所需的均值和标准差。"""
        self.input_mean = np.asarray(mean, dtype=np.float64)
        self.input_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-6)

    def forward(self, X, training=False):
        """执行前向传播。"""
        out = X
        if self.input_mean is not None and self.input_std is not None:
            out = (out - self.input_mean) / self.input_std

        for layer in self.layers:
            # Dropout 需要显式知道当前是否处于训练阶段。
            if isinstance(layer, Dropout):
                out = layer.forward(out, training)
            else:
                out = layer.forward(out)

        return out

    def predict_proba(self, X):
        """返回网络前向输出。"""
        return self.forward(X, training=False)

    def predict(self, X):
        """返回预测类别编号。"""
        return np.argmax(self.predict_proba(X), axis=1)

    def backward(self, dout, weight_decay=0.0, clip_norm=None):
        """执行反向传播，并在需要时做梯度裁剪。"""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                dout = layer.backward(dout, weight_decay)
            else:
                dout = layer.backward(dout)

        # 在参数更新前对所有线性层梯度做整体范数裁剪。
        if clip_norm is not None and clip_norm > 0:
            total_norm = 0.0
            for layer in self.layers:
                if isinstance(layer, Linear):
                    total_norm += np.sum(layer.dW**2) + np.sum(layer.db**2)
            total_norm = np.sqrt(total_norm)

            if total_norm > clip_norm:
                scale = clip_norm / (total_norm + 1e-12)
                for layer in self.layers:
                    if isinstance(layer, Linear):
                        layer.dW *= scale
                        layer.db *= scale

    def get_state(self):
        """提取模型参数状态，用于保存最佳模型。"""
        state = {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "layers": [],
        }
        for layer in self.layers:
            if isinstance(layer, Linear):
                state["layers"].append({"W": layer.W.copy(), "b": layer.b.copy()})
        return state

    def set_state(self, state):
        """从状态字典恢复模型参数。"""
        self.input_mean = state.get("input_mean")
        self.input_std = state.get("input_std")
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W = state["layers"][linear_idx]["W"].copy()
                layer.b = state["layers"][linear_idx]["b"].copy()
                linear_idx += 1

    def save(self, path):
        """将模型状态保存到磁盘。"""
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    @classmethod
    def load(cls, path):
        """从磁盘加载模型状态并返回一个构建好的 `MLP` 实例。

        此方法会从保存的状态字典中推断网络层尺寸并构建与之匹配的模型，
        然后恢复参数与输入归一化信息。
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # 推断网络结构：从每个线性层的 W 矩阵形状推断 sizes
        layers_info = state.get("layers", [])
        if not layers_info:
            raise ValueError("Saved state does not contain layer information.")

        sizes = [layers_info[0]["W"].shape[0]] + [li["W"].shape[1] for li in layers_info]
        input_dim = sizes[0]
        output_dim = sizes[-1]
        hidden_dims = sizes[1:-1]

        model = cls(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        model.set_state(state)
        return model
