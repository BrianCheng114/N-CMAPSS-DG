from torch import nn

class EngineGateMLP(nn.Module):
    def __init__(self, 
                 input_channels, 
                 hidden_dims,
                 output_channels, 
                 dropout):
        super().__init__()

        # 不同通道数据数值差异大，使用BatchNorm1d进行归一化
        self.layer1 = nn.Sequential(
            nn.Linear(input_channels, hidden_dims[0]),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
            )

        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], output_channels),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        # x shape: [batch, features]
        x = self.layer1(x) # [batch, hidden_dims[0]]
        x = self.layer2(x) # [batch, output_channels]

        return x

