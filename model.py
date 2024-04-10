from mamba_py.mamba import MambaConfig, Mamba
import torch.nn as nn


class MambaClassification(nn.Module):
    """
    各時刻でクラス分類をするmamba
    """

    def __init__(self, input_size: int, d_model: int, n_layers: int, num_classes: int):
        super().__init__()
        config = MambaConfig(d_model=d_model, n_layers=n_layers)
        self.in_head = nn.Linear(input_size, config.d_model)
        self.mamba = Mamba(config=config)
        self.class_head = nn.Linear(config.d_model, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        # y: (B, T, num_classes)
        x = self.in_head(x)
        x = self.mamba(x)
        y = self.class_head(x)
        return y
