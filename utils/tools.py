import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        '''
        早停机制
        Args:
            patience: 当验证损失在连续patience个epoch内没有改善时，触发早停
            min_delta: 验证损失改善的最小变化量，只有当验证损失减少超过min_delta时才算作改善
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # 当作函数执行，记录原参数
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # reset if improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class GRL(torch.autograd.Function):
    '''
    梯度反转层
    前向传播时GRL层的输出与输入相同; 反向传播时GRL层将梯度乘-lambda_实现对特征提取器的梯度反转
    '''
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None