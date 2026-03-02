import copy

import torch
import torch.nn as nn


class MoEModel(nn.Module):
    def __init__(self, num_domains,
                 Extractor, Regressor, GatingNetwork):
        super().__init__()
        self.num_domains = num_domains
        self.extractor = Extractor
        self.regressors = nn.ModuleList([
            copy.deepcopy(Regressor) for _ in range(num_domains)
        ])
        self.gate = GatingNetwork

    def forward_hard(self, x, domain):
        """
        第一阶段使用
        """
        z = self.extractor(x)
        outputs = torch.zeros(x.size(0), 1, device=x.device)

        for k in range(self.num_domains):
            mask = (domain == k+1)
            if mask.sum() > 0:
                outputs[mask] = self.regressors[k](z[mask])

        return outputs

    def forward_soft(self, x):
        """
        第二阶段使用
        """
        z = self.extractor(x)
        weights = self.gate(z)  # [batch_size, num_domains]

        preds = torch.stack(
            [head(z) for head in self.regressors],
            dim=2
        )  # [batch_size, 1, num_domains]

        weights = weights.unsqueeze(1)  # [batch_size, 1, num_domains]

        y_hat = (weights * preds).sum(dim=2)  # [batch_size, 1]

        return y_hat, weights.squeeze(1)