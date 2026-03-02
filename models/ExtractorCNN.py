from torch import nn

class EngineExtractorCNN(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 conv_channels, 
                 kernel_size, 
                 dropout):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(input_channels),
            nn.Conv1d(input_channels, conv_channels[0], kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )

        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(conv_channels[0]),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
            )

        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(conv_channels[1]),
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout)
            )

        self.layer4 = nn.Sequential(
            nn.BatchNorm1d(conv_channels[2]),
            nn.Conv1d(conv_channels[2], conv_channels[3], kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Dropout(dropout)
            )

        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[3] * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, output_channels),
            nn.ReLU()
            )

    def forward(self, x):
        # x shape: [batch, seq, features]
        x = x.permute(0, 2, 1)  # [batch, features, seq]
        x = self.layer1(x) # [batch, conv_channels[0], seq]
        x = self.layer2(x) # [batch, conv_channels[1], seq//2]
        x = self.layer3(x) # [batch, conv_channels[2], seq//4]
        x = self.layer4(x) # [batch, conv_channels[3], 4]
        x = self.layer5(x) # [batch, output_channels]

        return x

