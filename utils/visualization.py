import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

class EngineVisualizer:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
    def plot_predictions(self, outputs, labels, title, num_samples = 5, save_path = None):
        """
        绘制预测值与真实值的对比散点图
        参数:
            outputs: 模型预测输出 [n_samples]
            labels: 真实目标值 [n_samples]
            num_samples: 随机显示的样本数
            save_path: 图片保存路径
        """
        plt.figure(figsize=(8, 4))
        outputs = np.array(outputs).squeeze()
        labels = np.array(labels).squeeze()

        # 随机选择样本
        sample_indices = np.random.choice(len(outputs), size = num_samples, replace = False)
        
        # 所有样本的散点
        plt.scatter(labels, outputs, alpha=0.3, label='All samples')
        
        # 突出显示随机样本
        plt.scatter(labels[sample_indices], 
                    outputs[sample_indices],
                    s = 100, 
                    edgecolors = 'k', 
                    label = 'Selected samples')
        
        # 绘制理想线
        min_val = min(labels.min(), outputs.min())
        max_val = max(labels.max(), outputs.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha = 0.5)
        
        # 计算R2分数
        r2 = r2_score(labels, outputs)
        plt.title(title + f'Prediction (R2={r2:.3f})')
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()

    def plot_RUL(self, outputs, labels, title, save_path=None):
        """
        绘制RUL变化曲线
        Args:
            outputs: 模型预测输出
            labels: 真实目标值
            save_path: 图片保存路径
        """
        outputs = np.array(outputs).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(np.mean((outputs - labels) ** 2))

        plt.figure(figsize=(12, 6))
        plt.plot(labels, 'b-', linewidth=2, label='Real RUL')
        plt.plot(outputs, 'r--', linewidth=1.5, label='Predicted RUL')

        # 在图表上添加RMSE文本框
        textstr = f'RMSE = {rmse:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.12, textstr, transform=plt.gca().transAxes, fontsize=20,
                verticalalignment='top', bbox=props)

        # 设置图表属性
        plt.title(title + f'Real RUL vs Predicted RUL (RMSE={rmse:.4f})', fontsize=14)
        plt.xlabel('Sample Index', fontsize=16)
        plt.ylabel('Cycle', fontsize=16)
        plt.legend()

        # 添加图例
        plt.legend(fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # plt.show()

    def plot_loss_curves(self, loss_list, legend_list, save_path=None):
        """
        绘制训练和验证损失曲线
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 图片保存路径
        """
        plt.figure(figsize=(8, 4))
        for loss, legend in zip(loss_list, legend_list):
            plt.plot(loss, label=legend)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()