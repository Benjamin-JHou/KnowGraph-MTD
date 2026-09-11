"""Training machinery: DWA, gradient surgery, and the multitask loop.

* **Dynamic weight averaging (DWA)** -- per-task loss weights are adjusted from
  the loss descent rates across consecutive epochs:
  ``r_i(t-1) = L_i(t-1) / L_i(t-2)``,
  ``w_i(t) = K * exp(r_i(t-1) / T) / sum_j exp(r_j(t-1) / T)``.

* **Gradient surgery (GS)** -- for each pair (i, j) with ``cos(g_i, g_j) < 0``,
  task i's gradient is projected to remove its component along g_j. Projections
  are applied task-by-task in descending order of task loss magnitude
  (configurable: desc / asc / random).

* Optimization: AdamW with differential learning rates (1e-4 shared layers,
  2e-4 task heads), weight decay 1e-5, gradient clipping (max norm 1.0),
  ReduceLROnPlateau (factor 0.5, patience 5) and early stopping (patience 15).
"""

import itertools
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .metrics import all_metrics


# --------------------------------------------------------------------------- #
# Dynamic weight averaging
# --------------------------------------------------------------------------- #
class DynamicWeightAveraging:
    """Per-epoch DWA loss weighting (Liu et al., 2019)."""

    def __init__(self, n_tasks: int, temperature: float = 2.0, relative_clamp: float = 20.0):
        self.n_tasks = n_tasks
        self.temperature = max(float(temperature), 1e-6)
        self.relative_clamp = float(relative_clamp)
        self.history: List[np.ndarray] = []
        self.weights = np.ones(n_tasks, dtype=np.float64) / n_tasks

    def update(self, epoch_losses: Sequence[float]) -> np.ndarray:
        """Update weights from this epoch's mean per-task losses (unweighted)."""
        current = np.asarray(epoch_losses, dtype=np.float64)
        if len(self.history) >= 1:
            previous = self.history[-1]
            # Descent rate: ratio of the loss of the previous epoch to the
            # loss of the epoch before it.
            relative = previous / (current + 1e-8)
            relative = np.clip(relative, 1.0 / self.relative_clamp, self.relative_clamp)
            logits = relative / self.temperature
            logits = logits - logits.max()
            exp_w = np.exp(logits)
            self.weights = self.n_tasks * exp_w / (exp_w.sum() + 1e-8)
        self.history.append(current)
        return self.weights.copy()


# --------------------------------------------------------------------------- #
# Gradient surgery (PCGrad-style projection)
# --------------------------------------------------------------------------- #
def gradient_surgery(
    task_grads: List[Dict[nn.Parameter, torch.Tensor]],
    loss_values: Sequence[float],
    order: str = "desc",
    seed: Optional[int] = None,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Project conflicting per-task gradients and return their sum.

    ``task_grads[t]`` holds the gradient of task t with respect to the *shared*
    parameters. For each pair (i, j) with a negative cosine similarity, task i's
    gradient is projected to remove its component along task j's gradient.
    Projection order follows ``order``:
      - "desc":   descending task loss magnitude (default)
      - "asc":    ascending task loss magnitude
      - "random": fixed random order (seeded)
    """
    n = len(task_grads)
    if n == 0:
        return {}
    if n == 1:
        return {k: v.clone() for k, v in task_grads[0].items()}

    if order == "desc":
        order_idx = sorted(range(n), key=lambda i: -loss_values[i])
    elif order == "asc":
        order_idx = sorted(range(n), key=lambda i: loss_values[i])
    elif order == "random":
        order_idx = list(range(n))
        random.Random(seed).shuffle(order_idx)
    else:
        raise ValueError(f"Unknown gs_order '{order}' (use desc/asc/random).")

    params = list(task_grads[0].keys())
    result: Dict[nn.Parameter, torch.Tensor] = {
        p: torch.zeros_like(task_grads[0][p]) for p in params
    }

    for i in order_idx:
        g_i = {p: task_grads[i][p].clone() for p in params}
        for j in order_idx:
            if j == i:
                continue
            g_j = task_grads[j]
            # Conflict criterion: cos(g_i, g_j) < 0  <=>  dot(g_i, g_j) < 0.
            dot = sum(torch.dot(g_i[p].reshape(-1), g_j[p].reshape(-1)) for p in params)
            norm2_j = sum((g_j[p] ** 2).sum() for p in params) + 1e-12
            if float(dot) < 0.0:
                coef = dot / norm2_j
                for p in params:
                    g_i[p] = g_i[p] - coef * g_j[p]
        for p in params:
            result[p] = result[p] + g_i[p]

    return result


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Multitask trainer supporting the full component set and ablations."""

    def __init__(self, config: Config, model: nn.Module, task_names: List[str], device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.task_names = list(task_names)
        self.device = device
        self.criterion = nn.MSELoss()

        groups = model.parameter_groups(
            config.lr_shared, config.lr_head, config.weight_decay
        )
        self.optimizer = torch.optim.AdamW(groups)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=config.lr_factor,
            patience=config.lr_patience,
        )

        self.use_dwa = True
        self.use_gs = True
        self.use_adapters = True
        self.dwa: Optional[DynamicWeightAveraging] = None
        self.dwa_weights = np.ones(len(self.task_names)) / len(self.task_names)

    def configure(self, use_dwa: bool, use_gs: bool, use_adapters: bool):
        self.use_dwa = use_dwa
        self.use_gs = use_gs
        self.use_adapters = use_adapters
        if use_dwa:
            self.dwa = DynamicWeightAveraging(
                len(self.task_names), self.config.dwa_temperature,
                self.config.dwa_relative_change_clamp,
            )
        self.dwa_weights = np.ones(len(self.task_names)) / len(self.task_names)

    # ------------------------------------------------------------------ step
    def _conflict_parameters(self) -> List[nn.Parameter]:
        """Parameters shared by all tasks (gradient-surgery target).

        Only the projection layer and the transformer encoder blocks are used
        by every task; task adapters and heads are task-specific and therefore
        excluded from cross-task gradient projection.
        """
        return list(self.model.projection.parameters()) + list(self.model.blocks.parameters())

    def train_step(
        self, batches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> Tuple[float, np.ndarray]:
        """One optimizer step over a batch per task.

        Returns (weighted total loss, unweighted per-task losses).
        """
        cfg = self.config
        self.optimizer.zero_grad()
        conflict_params = self._conflict_parameters()
        task_grads: List[Dict[nn.Parameter, torch.Tensor]] = []
        loss_values: List[float] = []
        per_task_loss = np.zeros(len(self.task_names), dtype=np.float64)
        total_loss = torch.zeros((), device=self.device)

        present = [(t, b) for t, b in enumerate(batches) if b is not None]
        n_present = len(present)

        for k, (t, (x, y)) in enumerate(present):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model.forward_task(x, t)
            loss = self.criterion(pred, y)
            per_task_loss[t] = float(loss.detach().cpu())
            weighted = loss * float(self.dwa_weights[t])
            total_loss = total_loss + weighted
            retain = k < n_present - 1
            weighted.backward(retain_graph=retain)
            loss_values.append(float(weighted.detach()))
            if self.use_gs:
                task_grads.append({p: p.grad.clone() for p in conflict_params})

        # Resolve shared-layer gradient conflicts (gradient surgery).
        if self.use_gs and n_present > 1:
            projected = gradient_surgery(
                task_grads, loss_values, order=cfg.gs_order, seed=cfg.seed
            )
            for p in conflict_params:
                p.grad = projected[p]

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        return float(total_loss.detach().cpu()), per_task_loss

    # ------------------------------------------------------------- evaluate
    def evaluate(self, loaders: Dict[str, DataLoader]) -> Tuple[Dict[str, dict], Dict[str, float]]:
        """Evaluate all tasks on the given loaders; returns (per-task, aggregate)."""
        self.model.eval()
        per_task: Dict[str, dict] = {}
        with torch.no_grad():
            for t in self.task_names:
                loader = loaders.get(t)
                if loader is None or len(loader) == 0:
                    continue
                y_true, y_pred = [], []
                for x, y in loader:
                    pred = self.model.forward_task(x.to(self.device), self.task_names.index(t))
                    y_true.append(y.numpy())
                    y_pred.append(pred.detach().cpu().numpy())
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                per_task[t] = all_metrics(y_true, y_pred, self.config.active_threshold)
        return per_task, self._aggregate(per_task)

    @staticmethod
    def _aggregate(per_task: Dict[str, dict]) -> Dict[str, float]:
        keys = ["RMSE", "MAE", "R2", "Pearson r", "ROC-AUC", "PR-AUC"]
        out = {}
        for k in keys:
            vals = [m[k] for m in per_task.values() if np.isfinite(m[k])]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        test_loaders: Dict[str, DataLoader],
        out_prefix: str,
    ) -> Tuple[Dict[str, dict], Dict[str, float], Dict[str, np.ndarray]]:
        """Train with early stopping; returns test metrics, aggregates, predictions."""
        import csv
        import json
        import os

        cfg = self.config
        best_val_rmse = float("inf")
        best_state = None
        patience_counter = 0
        history = []
        epoch_task_losses: List[np.ndarray] = []

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            epoch_loss = np.zeros(len(self.task_names), dtype=np.float64)
            iters = {t: itertools.cycle(ld) for t, ld in train_loaders.items()}
            max_batches = max((len(ld) for ld in train_loaders.values()), default=1)
            for _ in range(max_batches):
                ordered = [None] * len(self.task_names)
                for pos, t in enumerate(self.task_names):
                    ordered[pos] = next(iters[t])
                step_loss, step_task_loss = self.train_step(ordered)
                total_loss += step_loss
                epoch_loss += step_task_loss
                n_batches += 1

            epoch_task_losses.append(epoch_loss / max(n_batches, 1))
            if self.dwa is not None:
                self.dwa_weights = self.dwa.update(epoch_task_losses[-1])

            val_metrics, val_agg = self.evaluate(val_loaders)
            val_rmse = val_agg["RMSE"]
            self.scheduler.step(val_rmse)
            history.append({
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "val_RMSE": val_rmse,
                "val_Pearson_r": val_agg["Pearson r"],
            })

            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stop_patience:
                    print(f"[train] Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_metrics, test_agg = self.evaluate(test_loaders)

        # Persist predictions, metrics, and history.
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        with open(out_prefix + "_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_prefix + "_metrics.json", "w") as f:
            json.dump({"per_task": test_metrics, "aggregate": test_agg}, f, indent=2)
        with open(out_prefix + "_predictions.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target", "pIC50_true", "pIC50_pred"])
            with torch.no_grad():
                for t in self.task_names:
                    loader = test_loaders.get(t)
                    if loader is None or len(loader) == 0:
                        continue
                    for x, y in loader:
                        pred = self.model.forward_task(x.to(self.device), self.task_names.index(t))
                        for a, b in zip(y.numpy(), pred.detach().cpu().numpy()):
                            writer.writerow([t, f"{a:.4f}", f"{b:.4f}"])

        return test_metrics, test_agg, {}
