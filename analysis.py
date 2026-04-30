import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import flatten_images
from data_loader import load_data_splits
from model import MLP, Linear

def visualize_first_layer_weights(model, num_neurons=16, img_shape=(64, 64, 3)):
    """
    可视化第一层隐藏层的权重矩阵。
    将 (12288, hidden_dim) 的权重中的前 num_neurons 列提取出来，
    恢复成 (64, 64, 3) 的图像格式进行展示。
    """
    print("\n[1] Generating first layer weight visualizations...")
    
    # 获取第一层线性层 (Linear)
    first_layer = None
    for layer in model.layers:
        if isinstance(layer, Linear):
            first_layer = layer
            break
            
    if first_layer is None:
        raise ValueError("模型中未找到 Linear 层。")

    weights = first_layer.W  # 形状应为 (12288, hidden_dims[0])
    
    # 创建 4x4 的画布 (假设 num_neurons=16)
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle("First Hidden Layer Weights Visualization", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < num_neurons and i < weights.shape[1]:
            # 提取第 i 个神经元的权重并 reshape
            w = weights[:, i].reshape(img_shape)
            
            # Min-Max 归一化到 [0, 1] 以便 matplotlib 显示为 RGB 图像
            w_min, w_max = w.min(), w.max()
            w_norm = (w - w_min) / (w_max - w_min + 1e-8)
            
            ax.imshow(w_norm)
            ax.set_title(f"Neuron {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("weight_visualization.png", dpi=150)
    plt.show()
    print("✅ Saved weight visualization as 'weight_visualization.png'")


def analyze_misclassifications(model, X_test, y_test, class_names, num_samples=5):
    """
    错例分析：找出预测错误的样本，展示原图、真实标签和预测标签。
    注意：X_test 应当是未被 flatten 的原始图像形状 (N, 64, 64, 3)。
    """
    print("\n[2] Running misclassification analysis...")
    
    # 推理时需要将图像拉平
    X_test_flat = flatten_images(X_test)
    y_pred = model.predict(X_test_flat)
    
    # 找出所有分错的索引
    error_indices = np.where(y_pred != y_test)[0]
    print(f"测试集中共有 {len(error_indices)} 个分类错误的样本。")
    
    if len(error_indices) == 0:
        print("🎉 模型在测试集上准确率为 100%，没有错例！")
        return

    # 随机挑选几个错例进行展示
    rng = np.random.default_rng(42) # 固定种子以便复现
    selected_indices = rng.choice(error_indices, min(num_samples, len(error_indices)), replace=False)

    fig, axes = plt.subplots(1, len(selected_indices), figsize=(15, 4))
    fig.suptitle("Misclassified Images Analysis", fontsize=16)

    # 兼容处理单样本情况
    if len(selected_indices) == 1: axes = [axes]

    for i, idx in enumerate(selected_indices):
        img = X_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # 将归一化的图像（如果有）恢复到可视范围内
        if img.max() <= 1.0:
            img_display = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_display = img.astype(np.uint8)

        axes[i].imshow(img_display)
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", 
              color="red" if true_label != pred_label else "green")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("misclassified_analysis.png", dpi=150)
    plt.show()
    print("✅ Saved misclassified analysis as 'misclassified_analysis.png'")


def get_args():
    parser = argparse.ArgumentParser(description="运行第一层权重可视化和错例分析")
    parser.add_argument("--model-path", type=str, default="final_best_model.pkl")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--num-neurons", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    
    # 1. 加载数据
    print("Loading data...")
    _, _, _, _, X_test, y_test, class_names, _ = load_data_splits(args.data_dir, seed=42)
    
    # 2. 加载预训练模型
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}. Please run train.py first.")
        exit(1)
        
    print(f"Loading model from {model_path}...")
    model = MLP.load(model_path)
    
    # 3. 运行分析
    visualize_first_layer_weights(model, num_neurons=args.num_neurons)
    analyze_misclassifications(model, X_test, y_test, class_names, num_samples=args.num_samples)
