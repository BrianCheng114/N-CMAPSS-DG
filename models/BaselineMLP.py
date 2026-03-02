from torch import nn

class EngineBaselineMLP(nn.Module):
    def __init__(self, 
                 input_channels, 
                 seq_length,
                 hidden_dims,
                 output_channels, 
                 dropout):
        super().__init__()

        # 不同通道数据数值差异大，使用BatchNorm1d进行归一化
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(input_channels * seq_length),
            nn.Linear(input_channels * seq_length, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )

        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], output_channels),
            nn.ReLU()
            )

    def forward(self, x):
        # x shape: [batch, seq, features]
        x = x.permute(0, 2, 1) # [batch, features, seq]
        x = self.layer1(x) # [batch, hidden_dims[0]]
        x = self.layer2(x) # [batch, hidden_dims[1]]
        x = self.layer3(x) # [batch, hidden_dims[2]]
        x = self.layer4(x) # [batch, output_channels]

        return x

