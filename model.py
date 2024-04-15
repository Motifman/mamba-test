from mamba_py.mamba import MambaConfig, Mamba
import torch.nn as nn
import torch.nn.functional as F
from mem import AttentionRedFB, AttnNoNormLMSTM


class MambaClassification(nn.Module):
    """
    classification mamba model
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


class TransformerClassification(nn.Module):
    def __init__(self, input_size: int, d_model: int, n_layers: int, num_classes: int):
        super().__init__()
        self.in_head = nn.Linear(input_size, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.class_hid = nn.Linear(d_model, d_model)
        self.class_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.in_head(x)
        x = self.transformer(x)
        x = F.relu(self.class_hid(x))
        y = self.class_head(x)
        return y


def make_model(
    model_name: str,
    input_size: int,
    d_model: int,
    n_layers: int,
    num_classes: int,
    parallel: bool,
    activation: str,
):
    if model_name == "mamba":
        model = MambaClassification(input_size, d_model, n_layers, num_classes)
    elif model_name == "transformer":
        model = TransformerClassification(input_size, d_model, n_layers, num_classes)
    elif model_name == "redfb":
        model = AttentionRedFB(
            input_size,
            d_model,
            num_classes,
            num_rnn=6,
            num_heads=4,
            nonlinear=activation,
            parallel=parallel,
        )
    elif model_name == "preredfb":
        model = AttnNoNormLMSTM(input_size, d_model, num_classes, 6, 4, activation)
    else:
        raise ValueError("select mamba or transformer")
    return model
