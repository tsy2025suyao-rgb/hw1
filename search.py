import argparse
import csv
import itertools
import os

import numpy as np

from data_loader import load_data_splits
from model import CrossEntropyLoss
from train import build_model, evaluate, train


INITIAL_GRID = {
    "hidden_dims": [[256, 128], [512, 256], [512, 256, 128]],
    "dropout": [0.1, 0.2, 0.3],
    "activation": ["relu"],
    "optimizer": ["sgd"],
    "epochs": [15],
    "batch_size": [128, 256],
    "lr": [1e-3, 5e-4, 1e-4],
    "momentum": [0.9],
    "weight_decay": [1e-4, 1e-3],
    "clip_norm": [5.0],
    "patience": [8],
    "lr_patience": [3],
    "lr_decay": [0.5],
    "min_lr": [1e-5],
    "augment": [True, False],
    "seed": [42],
}


FINETUNE_GRID = {
    "hidden_dims": [[512, 256, 128], [1024, 512, 256]],
    "dropout": [0.05, 0.1, 0.15],
    "activation": ["relu"],
    "optimizer": ["sgd"],
    "epochs": [30],
    "batch_size": [128],
    "lr": [0.0005, 0.001, 0.002],
    "momentum": [0.9],
    "weight_decay": [1e-4, 5e-4],
    "clip_norm": [5.0],
    "patience": [10],
    "lr_patience": [4],
    "lr_decay": [0.5],
    "min_lr": [1e-5],
    "augment": [True, False],
    "seed": [42],
}


class HyperArgs:
    """将超参数字典包装成 train() 可读取的属性对象。"""

    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.no_augment = not self.augment


def normalize_dropout(dropout, hidden_dims):
    """如果 dropout 是标量，则扩展到每个隐藏层。"""
    if isinstance(dropout, (float, int)):
        return [float(dropout)] * len(hidden_dims)
    return [float(rate) for rate in dropout]


def save_result_to_csv(csv_path, trial_id, config, val_loss, val_acc):
    """将一次超参数试验结果追加写入 CSV。"""
    file_exists = os.path.exists(csv_path)
    row_data = {"trial_id": trial_id}
    for key, value in config.items():
        row_data[key] = "-".join(map(str, value)) if isinstance(value, list) else value
    row_data["val_loss"] = val_loss
    row_data["val_acc"] = val_acc

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def sample_configs(param_grid, n_iter, seed):
    """从参数网格笛卡尔积中随机采样配置。"""
    keys, values = zip(*param_grid.items())
    all_configs = [dict(zip(keys, values_)) for values_ in itertools.product(*values)]
    rng = np.random.default_rng(seed)
    sample_size = min(n_iter, len(all_configs))
    return list(rng.choice(all_configs, size=sample_size, replace=False)), len(all_configs)


def random_search(
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    input_shape,
    output_dim,
    n_iter,
    csv_path,
    seed,
    title,
):
    """执行随机超参数搜索并记录验证集性能。"""
    chosen_configs, total_configs = sample_configs(param_grid, n_iter, seed)
    best_acc, best_config = -np.inf, None

    print(f"\n{'=' * 50}")
    print(f"开启{title} (采样数: {len(chosen_configs)} / 总组合数: {total_configs})")
    print(f"{'=' * 50}")

    for trial_id, config in enumerate(chosen_configs, start=1):
        config = dict(config)
        config["dropout"] = normalize_dropout(config["dropout"], config["hidden_dims"])
        print(f"\nTrial {trial_id}/{len(chosen_configs)} | Config: {config}")

        model = build_model(
            input_shape=input_shape,
            class_count=output_dim,
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dropout=config["dropout"],
            seed=config["seed"],
        )

        args = HyperArgs(**config)
        train(model, X_train, y_train, X_val, y_val, args)

        criterion = CrossEntropyLoss(weight_decay=config["weight_decay"])
        val_loss, val_acc, _ = evaluate(model, criterion, X_val, y_val, batch_size=512)
        print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")
        save_result_to_csv(csv_path, trial_id, config, val_loss, val_acc)

        if val_acc > best_acc:
            best_acc, best_config = val_acc, config
            print(f"新的最佳纪录: {best_acc:.4f}")

    print(f"\n{'=' * 50}")
    print(f"最佳配置: {best_config}")
    print(f"最佳准确率: {best_acc:.4f}")
    print(f"{'=' * 50}")
    return best_config, best_acc


def get_args():
    parser = argparse.ArgumentParser(description="运行 MLP 超参数随机搜索")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--mode", choices=["initial", "finetune"], default="initial")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    param_grid = INITIAL_GRID if args.mode == "initial" else FINETUNE_GRID
    csv_path = args.csv_path or (
        "search_results.csv" if args.mode == "initial" else "finetune_results.csv"
    )

    if args.overwrite and os.path.exists(csv_path):
        os.remove(csv_path)

    X_train, y_train, X_val, y_val, _, _, class_names, _ = load_data_splits(
        args.data_dir, seed=42
    )

    random_search(
        param_grid,
        X_train,
        y_train,
        X_val,
        y_val,
        input_shape=X_train.shape[1:],
        output_dim=len(class_names),
        n_iter=args.n_iter,
        csv_path=csv_path,
        seed=args.seed,
        title="初始随机搜索" if args.mode == "initial" else "精细化随机搜索",
    )


if __name__ == "__main__":
    main()
