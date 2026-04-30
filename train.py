import argparse
import json
import os

import numpy as np

from data_loader import (
    augment_batch,
    compute_feature_stats,
    flatten_images,
    load_data_splits,
    parse_float_list,
    parse_int_list,
)
from model import Adam, CrossEntropyLoss, MLP, SGD


def build_model(input_shape, class_count, hidden_dims, activation, dropout, seed):
    """根据实验配置构建 MLP。"""
    return MLP(
        input_dim=int(np.prod(input_shape)),
        hidden_dims=hidden_dims,
        output_dim=class_count,
        activation_name=activation,
        seed=seed,
        dropout_rates=dropout,
    )


def make_optimizer(args, lr):
    """根据命令行参数创建优化器。"""
    if args.optimizer == "adam":
        return Adam(lr=lr)
    momentum = args.momentum if args.optimizer == "momentum" else 0.0
    return SGD(lr=lr, momentum=momentum)


def evaluate(model, criterion, X, y, batch_size=512):
    """分批评估模型，返回 loss、accuracy 和预测标签。"""
    X_flat = flatten_images(X)
    outputs = [
        model.predict_proba(X_flat[i : i + batch_size])
        for i in range(0, X_flat.shape[0], batch_size)
    ]

    y_pred = np.vstack(outputs)
    loss = criterion.forward(y_pred, y, model)
    y_label = np.argmax(y_pred, axis=1)
    acc = float(np.mean(y_label == y))
    return loss, acc, y_label


def train_one_epoch(
    model,
    criterion,
    optimizer,
    X,
    y,
    batch_size,
    augment,
    rng,
    weight_decay,
    clip_norm,
):
    """执行一个 epoch 的 mini-batch 训练。"""
    order = rng.permutation(X.shape[0])
    total_loss, correct, seen = 0.0, 0, 0

    for start in range(0, X.shape[0], batch_size):
        idx = order[start : start + batch_size]
        X_batch, y_batch = X[idx], y[idx]

        if augment:
            X_batch = augment_batch(X_batch, rng)

        y_pred = model.forward(flatten_images(X_batch), training=True)
        loss = criterion.forward(y_pred, y_batch, model)
        model.backward(criterion.backward(), weight_decay=weight_decay, clip_norm=clip_norm)
        optimizer.step(model)

        actual_size = len(y_batch)
        total_loss += loss * actual_size
        correct += np.sum(np.argmax(y_pred, axis=1) == y_batch)
        seen += actual_size

    return total_loss / seen, correct / seen


def _artifact_paths(plot_path):
    if plot_path.endswith(".png"):
        return (
            plot_path.replace(".png", "_loss.png"),
            plot_path.replace(".png", "_val_acc.png"),
            plot_path.replace(".png", "_history.json"),
        )
    return (
        f"{plot_path}_loss.png",
        f"{plot_path}_val_acc.png",
        f"{plot_path}_history.json",
    )


def save_training_artifacts(history, plot_path):
    """保存训练/验证 loss 曲线、验证 accuracy 曲线和 history JSON。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_path, acc_path, history_path = _artifact_paths(plot_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("训练/验证 损失曲线")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"已保存损失曲线到 {loss_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("验证集 准确率")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"已保存验证准确率曲线到 {acc_path}")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"已保存训练历史到 {history_path}")


def train(model, X_train, y_train, X_val, y_val, args, plot_curves=None, plot_path="loss_acc.png"):
    """完整训练流程：训练、验证、学习率衰减、早停和最优权重恢复。"""
    rng = np.random.default_rng(args.seed)

    input_mean, input_std = compute_feature_stats(X_train)
    model.set_input_normalization(input_mean, input_std)

    criterion = CrossEntropyLoss(weight_decay=args.weight_decay)
    current_lr = args.lr
    optimizer = make_optimizer(args, current_lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_state, best_val_acc, best_val_loss = model.get_state(), -np.inf, np.inf
    epochs_since_improve, epochs_since_lr_drop = 0, 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            criterion,
            optimizer,
            X_train,
            y_train,
            args.batch_size,
            args.augment and not args.no_augment,
            rng,
            args.weight_decay,
            args.clip_norm,
        )
        val_loss, val_acc, _ = evaluate(model, criterion, X_val, y_val)

        for key, value in zip(
            history.keys(), [train_loss, train_acc, val_loss, val_acc, current_lr]
        ):
            history[key].append(float(value))

        improved = (val_acc > best_val_acc + 1e-4) or (
            abs(val_acc - best_val_acc) <= 1e-4 and val_loss < best_val_loss - 1e-4
        )
        if improved:
            best_val_acc, best_val_loss, best_state = val_acc, val_loss, model.get_state()
            epochs_since_improve, epochs_since_lr_drop = 0, 0
        else:
            epochs_since_improve += 1
            epochs_since_lr_drop += 1

        if epochs_since_lr_drop >= args.lr_patience and current_lr > args.min_lr:
            current_lr = max(current_lr * args.lr_decay, args.min_lr)
            optimizer.lr = current_lr
            epochs_since_lr_drop = 0

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
            f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
            f"学习率: {current_lr:.6f}"
        )

        if epochs_since_improve >= args.patience:
            print(f"达到早停条件，训练在第 {epoch + 1} 轮停止。")
            break

    model.set_state(best_state)

    should_plot = plot_curves if plot_curves is not None else getattr(args, "plot_curves", False)
    if should_plot:
        try:
            save_training_artifacts(history, plot_path)
        except Exception as exc:
            print(f"保存训练曲线/历史失败: {exc}")

    return history


def get_args():
    parser = argparse.ArgumentParser(description="在 EuroSAT 上训练 MLP")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", type=str, default="256,128")
    parser.add_argument("--dropout", type=str, default="0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--model-path", type=str, default="saved_model.pkl")
    parser.add_argument("--plot-curves", action="store_true")
    parser.add_argument("--plot-path", type=str, default="loss_acc")

    args = parser.parse_args()
    args.hidden_dims = parse_int_list(args.hidden_dims)
    args.dropout = parse_float_list(args.dropout)
    return args


def main():
    args = get_args()
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, _ = load_data_splits(
        args.data_dir, seed=args.seed
    )

    model = build_model(
        input_shape=X_train.shape[1:],
        class_count=len(class_names),
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        dropout=args.dropout,
        seed=args.seed,
    )

    train(model, X_train, y_train, X_val, y_val, args, plot_path=args.plot_path)

    _, test_acc, _ = evaluate(model, CrossEntropyLoss(), X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")

    try:
        model_dir = os.path.dirname(args.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        model.save(args.model_path)
        print(f"已保存模型到 {args.model_path}")
    except OSError as exc:
        print(f"保存模型失败: {exc}")


if __name__ == "__main__":
    main()
