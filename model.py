from mamba_py.mamba import MambaConfig, Mamba
from palm import PaLM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, 0.5)
        encoder = nn.TransformerEncoderLayer(
            d_model,
            nhead=4,
            dim_feedforward=2 * d_model,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.class_head = nn.Linear(d_model, num_classes)
        self.init_weights()

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        T = x.shape[1]
        device = x.device
        x_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        x = self.transformer(x, x_mask)
        y = self.class_head(x)
        return y

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.class_head.bias.data.zero_()
        self.class_head.weight.data.uniform_(-init_range, init_range)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 学習不可能

    def forward(self, x: torch.Tensor):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


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
    elif model_name == "palm":
        model = PaLM(
            input_size, d_model, num_classes, n_layers, dim_head=40, heads=2, ff_mult=2
        )
    else:
        raise ValueError("select mamba or transformer")
    return model
