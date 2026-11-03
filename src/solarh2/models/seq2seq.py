"""Encoder–Decoder (Seq2Seq) RNN — LSTM veya GRU hücresiyle, doğrudan çok adımlı tahmin.

LSTM ile GRU aynı mimaride, yalnızca `cell` parametresiyle ayrışır; böylece kıyaslama
adildir (aynı girdiler, aynı decoder, aynı eğitim döngüsü).

Neden bu mimari?
- Decoder, her ufuk adımında o ana ait *bilinen gelecek* değişkenleri (saat kodlaması,
  hava tahmini) alır; bu, gün öncesi tahminin gerçekliğine uygundur.
- Decoder kendi çıktısını geri beslemez -> hata birikimi (exposure bias) yoktur.
"""
from __future__ import annotations

import torch
from torch import nn

CELLS = {"lstm": nn.LSTM, "gru": nn.GRU}


class Seq2SeqRNN(nn.Module):
    def __init__(self, n_past: int, n_future: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2, cell: str = "lstm"):
        super().__init__()
        if cell not in CELLS:
            raise ValueError(f"cell {list(CELLS)} içinden olmalı, gelen: {cell!r}")
        rnn = CELLS[cell]
        drop = dropout if num_layers > 1 else 0.0
        self.cell = cell
        self.encoder = rnn(n_past, hidden_size, num_layers, batch_first=True, dropout=drop)
        self.decoder = rnn(n_future, hidden_size, num_layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        # x_past: (B, L, n_past)  x_future: (B, H, n_future)  ->  (B, H)
        # LSTM durumu (h, c) demetidir, GRU durumu yalnızca h; ikisi de doğrudan aktarılır.
        _, state = self.encoder(x_past)
        out, _ = self.decoder(x_future, state)
        return self.head(out).squeeze(-1)


class Seq2SeqLSTM(Seq2SeqRNN):
    """Geriye dönük uyumluluk: önceki sürümlerdeki ad."""

    def __init__(self, n_past, n_future, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__(n_past, n_future, hidden_size, num_layers, dropout, cell="lstm")
