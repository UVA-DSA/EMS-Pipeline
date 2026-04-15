import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self, data_format: str = "channels_last") -> None:
        super().__init__()
        self.step_axis = 1 if data_format == "channels_last" else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=self.step_axis).values


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=512, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
            nn.Conv1d(in_channels=256, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.encoder(x)


class TransformerInferenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        kernel_size: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=input_dim, out_channels=d_model, kernel_size=kernel_size)
        self.pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=batch_first,
            ),
            num_layers=num_layers,
        )
        self.out = nn.Linear(d_model, output_dim)
        self.max_pool = GlobalMaxPooling1D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pe(x)
        x = self.transformer(x)
        x = self.out(x)
        x = self.max_pool(x)
        return x
