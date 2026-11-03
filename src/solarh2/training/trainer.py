"""Erken durdurmalı, en iyi checkpoint'i saklayan eğitim döngüsü."""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from solarh2.data.windowing import WindowSet


def _loader(ws: WindowSet, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(ws.x_past), torch.from_numpy(ws.x_future),
                       torch.from_numpy(ws.y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train_model(model: nn.Module, train: WindowSet, val: WindowSet, cfg: dict,
                on_epoch: Callable[[int, float], None] | None = None, verbose: bool = True):
    """on_epoch(epoch, val_loss): her epoch sonunda çağrılır (ör. Optuna pruning için;
    TrialPruned fırlatabilir)."""
    t = cfg["training"]
    device = resolve_device(t["device"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    tr_dl = _loader(train, t["batch_size"], shuffle=True)   # pencereler bağımsız örnek
    va_dl = _loader(val, t["batch_size"], shuffle=False)

    best, best_state, wait, history = np.inf, None, 0, []
    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for xp, xf, y in tr_dl:
            xp, xf, y = xp.to(device), xf.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xp, xf), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(y)
        tr_loss /= len(train)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xp, xf, y in va_dl:
                va_loss += loss_fn(model(xp.to(device), xf.to(device)), y.to(device)).item() * len(y)
        va_loss /= len(val)
        sched.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        if verbose:
            print(f"epoch {epoch:3d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if on_epoch is not None:
            on_epoch(epoch, va_loss)

        if va_loss < best - 1e-5:
            best, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= t["patience"]:
                if verbose:
                    print(f"Erken durdurma: epoch {epoch}, en iyi val {best:.4f}")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model: nn.Module, ws: WindowSet, y_scaler, device: str = "auto") -> np.ndarray:
    dev = resolve_device(device)
    model.to(dev).eval()
    out = model(torch.from_numpy(ws.x_past).to(dev), torch.from_numpy(ws.x_future).to(dev))
    pred = y_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).reshape(out.shape)
    return np.clip(pred, 0, None)  # fiziksel kısıt: güç negatif olamaz
