from torch import nn

class EngineBaselineLSTM(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 hidden_size, 
                 num_layers,
                 dropout):
        super().__init__()

        self.layer1 = nn.BatchNorm1d(input_channels)
        
        self.layer2 = nn.LSTM(input_size=input_channels, 
                              hidden_size=hidden_size, 
                              num_layers=num_layers, 
                              batch_first=True)
        
        self.layer3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_channels)
            )
        


    def forward(self, x):
        # x shape: [batch, seq, features]
        batch, seq, features = x.shape
        x = x.contiguous().view(batch * seq, features)
        x = self.layer1(x)
        x = x.view(batch, seq, features)
        
        out, _ = self.layer2(x) # [batch, seq, hidden_size]
        out = out[:, -1, :]  # 取最后一个时间步的输出 [batch, hidden_size]
        out = self.layer3(out) # [batch, output_channels]

        return out

