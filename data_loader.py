import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_int_list(text):
    """将类似 '1024,512,256' 的字符串解析为整数列表。"""
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("至少需要提供一个整数。")
    return values


def parse_float_list(text):
    """将类似 '0.15,0.15,0.15' 的字符串解析为浮点数列表。"""
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("至少需要提供一个浮点数。")
    return values


def load_eurosat_rgb(data_dir, img_size=(64, 64)):
    """加载 EuroSAT RGB 图像及其类别标签。"""
    X, y = [], []
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, fname)
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(img_size)
                X.append(np.asarray(img, dtype=np.float32) / 255.0)
            y.append(class_name)

    return np.stack(X), np.asarray(y), class_names


def preprocess_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """将标签编码后按分层抽样划分为训练、验证和测试集。"""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=test_size + val_size,
        stratify=y_encoded,
        random_state=random_state,
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, le


def load_data_splits(data_dir="EuroSAT_RGB", seed=42):
    """加载 EuroSAT RGB 并返回训练/验证/测试划分和类别名。"""
    X, y, class_names = load_eurosat_rgb(data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = preprocess_data(
        X, y, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_encoder


def flatten_images(X):
    """将图像批量展平为二维特征矩阵，保持样本顺序不变。"""
    return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)


def compute_feature_stats(X, eps=1e-6):
    """从训练集计算特征均值和标准差，用于标准化输入。"""
    X_flat = flatten_images(X)
    mean = X_flat.mean(axis=0, dtype=np.float64)
    std = X_flat.std(axis=0, dtype=np.float64)
    std = np.maximum(std, eps)
    return mean.astype(np.float64), std.astype(np.float64)


def augment_batch(X_batch, rng):
    """对一个批次做轻量增强，包括水平翻转和整数倍旋转。"""
    X_aug = np.array(X_batch, copy=True)

    # 以 50% 概率做水平翻转。
    flip_h = rng.random(X_aug.shape[0]) < 0.5
    X_aug[flip_h] = X_aug[flip_h, :, ::-1, :]

    # 在 0、90、180、270 度之间随机旋转。
    rotations = rng.integers(0, 4, size=X_aug.shape[0])
    for i, k in enumerate(rotations):
        if k:
            X_aug[i] = np.rot90(X_aug[i], k=int(k), axes=(0, 1))

    return X_aug.astype(np.float32, copy=False)


if __name__ == "__main__":
    data_dir = "EuroSAT_RGB"
    X, y, class_names = load_eurosat_rgb(data_dir)
    print(f"数据集样本数: {len(X)}")
    print(f"类别列表: {class_names}")
    X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(X, y)
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
