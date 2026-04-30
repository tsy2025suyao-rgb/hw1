import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix

from data_loader import load_data_splits, parse_float_list, parse_int_list
from model import CrossEntropyLoss
from train import build_model, evaluate


def parse_dropout_rates(text):
    values = parse_float_list(text)
    if any(rate < 0.0 or rate >= 1.0 for rate in values):
        raise ValueError("dropout 比例必须在 [0, 1) 区间内。")
    return values


def save_confusion_matrix_csv(cm, class_names, output_path):
    """Save the confusion matrix as a CSV table."""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(class_names))
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.astype(int).tolist())


def plot_confusion_matrix(cm, class_names, output_path, normalize=False, title=None):
    """Save a confusion matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize:
        values = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt = ".2f"
        color_max = 1.0
        default_title = "Normalized Confusion Matrix"
    else:
        values = cm
        fmt = "d"
        color_max = None
        default_title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(values, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=color_max)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title or default_title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = values.max() / 2.0 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = format(values[i, j], fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def get_args():
    """解析测试用命令行参数。"""
    parser = argparse.ArgumentParser(
        description="加载训练好的模型并在测试集上评估。"
    )
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--model-path", type=str, default="saved_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dims", type=str, default="256,128")
    parser.add_argument("--dropout", type=str, default="0.1,0.1")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--save-confusion-csv", type=str, default=None)
    parser.add_argument("--save-confusion-plot", type=str, default=None)
    parser.add_argument("--save-normalized-confusion-plot", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"未找到模型权重文件: {args.model_path}。"
            "请先在 train.py 中训练模型。"
        )

    X_train, _, _, _, X_test, y_test, class_names, _ = load_data_splits(
        args.data_dir, seed=args.seed
    )

    model = build_model(
        input_shape=X_train.shape[1:],
        class_count=len(class_names),
        hidden_dims=parse_int_list(args.hidden_dims),
        activation=args.activation,
        dropout=parse_dropout_rates(args.dropout),
        seed=args.seed,
    )
    with open(args.model_path, "rb") as f:
        model.set_state(pickle.load(f))

    test_loss, test_acc, y_pred = evaluate(
        model,
        CrossEntropyLoss(weight_decay=0.0),
        X_test,
        y_test,
        batch_size=args.eval_batch_size,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("类别顺序:", class_names)
    print("混淆矩阵:")
    print(cm)

    if args.save_confusion_csv:
        save_confusion_matrix_csv(cm, class_names, args.save_confusion_csv)
        print(f"已保存混淆矩阵 CSV 到 {args.save_confusion_csv}")

    if args.save_confusion_plot:
        plot_confusion_matrix(
            cm,
            class_names,
            args.save_confusion_plot,
            normalize=False,
            title=f"Confusion Matrix (acc={test_acc:.4f})",
        )
        print(f"已保存混淆矩阵图到 {args.save_confusion_plot}")

    if args.save_normalized_confusion_plot:
        plot_confusion_matrix(
            cm,
            class_names,
            args.save_normalized_confusion_plot,
            normalize=True,
            title="Normalized Confusion Matrix",
        )
        print(f"已保存归一化混淆矩阵图到 {args.save_normalized_confusion_plot}")
