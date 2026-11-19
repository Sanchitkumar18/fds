"""Train and score matched baseline and spectral-prior classifiers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F

from src.data.preprocessing import StandardScaler1D
from src.models.factory import build_model
from src.physics.physics_losses import build_frequency_masks, spectral_consistency_loss

from .data import WIN, ratios_for, spectral_evidence

CONFIG = {"preprocess": {"stft_size": 64}, "model": {}}


def spectral_loss_v2(logits, evidence):
    """Eq. (6); the healthy logit has exactly zero gradient."""
    weight = logits.softmax(dim=1)[:, 1:].sum(dim=1).detach()
    aligned = (logits[:, 1:].softmax(dim=1) * evidence[:, 1:].detach()).sum(dim=1)
    return (weight * (1 - aligned)).sum() / (weight.sum() + 1e-12)


def grad_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** .5


def metrics(y, predictions):
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "windows": len(y)}


def experiment(dataset, protocol, data, stress, model_name, prior, seed, epochs, device):
    train = np.flatnonzero(data[protocol] == "train")
    val = np.flatnonzero(data[protocol] == "val")
    test = np.flatnonzero(data[protocol] == "test")
    y = data["y"]
    if model_name in ("rf", "svm"):
        scaler = StandardScaler1D().fit(data["handcrafted"][train])
        estimator = (RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
                     if model_name == "rf" else SVC(C=10))
        estimator.fit(scaler.transform(data["handcrafted"][train]), y[train])

        def predict(other, idx):
            return estimator.predict(scaler.transform(other["handcrafted"][idx]))

        best_epoch, weight = None, 0.0
    else:
        n_classes = len(ratios_for(dataset))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = build_model(model_name, CONFIG, n_classes).to(device)
        scaler = StandardScaler1D().fit(data["core"][train])
        features = scaler.transform(data["core"])
        labels = torch.as_tensor(y, device=device, dtype=torch.long)
        counts = np.bincount(y[train], minlength=n_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(
            counts.sum() / (n_classes * counts), dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        params = list(model.backbone.parameters()) if prior != "none" else []
        adaptive = prior in ("raw", "envelope", "v1", "target")
        weight = float(prior.split("_")[1]) if prior.startswith("fixed_") else float(adaptive)

        def logits(other, other_features, idx):
            maps = torch.as_tensor(other["stft"][idx, None, None], device=device)
            stats = torch.as_tensor(other_features[idx], device=device)
            return model(maps, stats)["logits"]

        def predict(other, idx):
            other_features = scaler.transform(other["core"])
            with torch.no_grad():
                return torch.cat([
                    logits(other, other_features, batch).argmax(1).cpu()
                    for batch in np.array_split(idx, max(1, (len(idx) + 63) // 64))
                ]).numpy()

        if prior == "v1":
            freq = np.linspace(0, data["sample_rate"][0] / 2, 64)
            masks = np.stack([build_frequency_masks(freq, ratios_for(dataset) * hz)
                              for hz in data["shaft_hz"]])
            masks = torch.as_tensor(masks, device=device)
            energy = torch.as_tensor((data["stft"] ** 2).mean(-1), device=device)
            valid = torch.as_tensor(ratios_for(dataset) > 0, device=device)

            def physics(prediction, idx):
                return spectral_consistency_loss(energy[idx], prediction.softmax(1), masks[idx], valid)

        elif prior != "none":
            fs = data["sample_rate"][0]
            freq = np.fft.rfftfreq(WIN, 1 / fs)
            view = "raw" if prior == "raw" else "envelope"
            evidence = spectral_evidence(data[view], freq[freq <= 1000], data["shaft_hz"],
                                         ratios_for(dataset))
            evidence = torch.as_tensor(evidence, device=device)

            def physics(prediction, idx):
                return spectral_loss_v2(prediction, evidence[idx])

        best_loss, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = rng.permutation(train)
            for begin in range(0, len(shuffled), 64):
                idx = shuffled[begin:begin + 64]
                prediction = logits(data, features, idx)
                supervised = criterion(prediction, labels[idx])
                loss = supervised
                if prior != "none":
                    spec_loss = physics(prediction, idx)
                    if prior == "target":
                        targets = rng.choice(test, len(idx), replace=len(test) < len(idx))
                        spec_loss = (spec_loss + physics(logits(data, features, targets), targets)) / 2
                    if adaptive:
                        target_weight = .25 * grad_norm(supervised, params) / (grad_norm(spec_loss, params) + 1e-8)
                        weight = float(np.clip(.88 * weight + .12 * target_weight, 1e-3, 1e2))
                    loss = loss + weight * spec_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            schedule.step()
            model.eval()
            with torch.no_grad():
                outputs = torch.cat([logits(data, features, val[i:i + 64])
                                     for i in range(0, len(val), 64)])
                val_loss = criterion(outputs, labels[val]).item()
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch = val_loss, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if epoch - best_epoch >= 8:
                break
        if best_state is None:
            raise RuntimeError("Training did not produce finite validation loss")
        model.load_state_dict(best_state)

    robustness = {}
    for variant, other in stress.items():
        idx = np.arange(len(other["y"]))
        robustness[variant] = metrics(other["y"], predict(other, idx))
    return {"dataset": dataset, "protocol": protocol, "model": model_name, "prior": prior,
            "seed": seed, "epochs": epochs, "best_epoch": best_epoch, "final_weight": weight,
            "split_windows": {"train": len(train), "val": len(val), "test": len(test)},
            "test": metrics(y[test], predict(data, test)), "stress": robustness}
