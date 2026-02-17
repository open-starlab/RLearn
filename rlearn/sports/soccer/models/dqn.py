from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from rlearn.sports.soccer.constant import UNAVAILABLE_ACTION_QVALUE
from rlearn.sports.soccer.models.q_model_base import QModelBase
from rlearn.sports.soccer.modules.optimizer import LRScheduler, Optimizer
from rlearn.sports.soccer.modules.seq2seq_encoder import Seq2SeqEncoder
from rlearn.sports.soccer.torch.metrics.classification import get_classification_full_metrics


class Prediction:
    def __init__(self, q_values: torch.Tensor) -> None:
        self.q_values = q_values


@QModelBase.register("dqn_attacker")
class AttackerDQNModel(QModelBase):
    def __init__(
        self,
        observation_dim: int,
        sequence_encoder: Dict[str, Any],
        optimizer: Dict[str, Any] | None = None,
        scheduler: Dict[str, Any] | None = None,
        vocab_size: int = 16,
        pad_token_id: int = 15,
        gamma: float = 1.0,
        lambda_: float = 0.0,
        lambda2_: float = 0.0,
        class_weights: torch.Tensor | None = None,
        offball_action_idx: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        onball_action_idx: list[int] = [9, 10, 11, 12, 13],
        defensive_action_idx: list[int] = [14],
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.vocab_size = vocab_size
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self.pad_token_id = pad_token_id
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.offball_action_idx = offball_action_idx
        self.onball_action_idx = onball_action_idx
        self.defensive_action_idx = defensive_action_idx

        self.encoder = Seq2SeqEncoder.from_params(sequence_encoder)
        self.fc1 = nn.Linear(self.observation_dim, self.encoder.get_input_dim())
        self.fc2 = nn.Linear(self.encoder.get_output_dim(), self.vocab_size)

        self.train_metrics = get_classification_full_metrics(
            num_classes=self.vocab_size,
            average_method="macro",
            ignore_index=self.pad_token_id,
            prefix="train_",
        )
        self.val_metrics = get_classification_full_metrics(
            num_classes=self.vocab_size,
            average_method="macro",
            ignore_index=self.pad_token_id,
            prefix="val_",
        )
        self.test_metrics = get_classification_full_metrics(
            num_classes=self.vocab_size,
            average_method="macro",
            ignore_index=self.pad_token_id,
            prefix="test_",
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = F.relu(self.fc1(batch["observation"]))
        out = self.encoder(x)
        q_values = self.fc2(out)
        return q_values

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mask = batch["mask"]
        reward = batch["reward"]
        action = batch["action"]
        on_ball = batch["onball_mask"]

        q_values = self.forward(batch)
        device = q_values.device

        q_sa = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.forward(batch)[:, 1:].max(dim=2).values
            td_target = reward[:, :-1] + self.gamma * next_q_values

        td_error = td_target - q_sa[:, :-1]
        td_loss = (td_error**2 * mask[:, :-1]).sum() / mask[:, :-1].sum()
        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.log("train_td_loss", td_loss, sync_dist=True)
        self.log("train_l1_loss", l1_loss, sync_dist=True)

        q_values_masked = q_values.clone()
        unavailable_action_value = torch.tensor(UNAVAILABLE_ACTION_QVALUE, device=device)

        off_ball_action_mask = on_ball.unsqueeze(-1).expand(-1, -1, len(self.offball_action_idx)).to(device)
        on_ball_action_mask = ~on_ball.unsqueeze(-1).expand(-1, -1, len(self.onball_action_idx)).to(device)

        q_values_masked[:, :, self.offball_action_idx] = torch.where(
            off_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.offball_action_idx]
        )
        q_values_masked[:, :, self.onball_action_idx] = torch.where(
            on_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.onball_action_idx]
        )

        action_loss = F.cross_entropy(
            input=q_values_masked.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values_masked.device) if self.class_weights is not None else None,
        )

        self.train_metrics(
            q_values_masked.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )

        self.log("train_action_loss", action_loss, sync_dist=True)
        self.log_dict(self.train_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.log("train_total_loss", total_loss, sync_dist=True)
        logging.info(
            f"[Epoch {self.current_epoch} | Batch {batch_idx}] train total loss: {total_loss.item():.6f}, td loss: {td_loss.item():.6f}, action loss: {action_loss.item():.6f}, l1 loss: {l1_loss.item():.6f}"
        )
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mask = batch["mask"]
        reward = batch["reward"]
        action = batch["action"]
        on_ball = batch["onball_mask"]

        q_values = self.forward(batch)
        device = q_values.device

        q_sa = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.forward(batch)[:, 1:].max(dim=2).values
            td_target = reward[:, :-1] + self.gamma * next_q_values

        td_error = td_target - q_sa[:, :-1]
        td_loss = (td_error**2 * mask[:, :-1]).sum() / mask[:, :-1].sum()
        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.log("val_td_loss", td_loss, sync_dist=True)
        self.log("val_l1_loss", l1_loss, sync_dist=True)

        q_values_masked = q_values.clone()
        unavailable_action_value = torch.tensor(UNAVAILABLE_ACTION_QVALUE, device=device)

        off_ball_action_mask = on_ball.unsqueeze(-1).expand(-1, -1, len(self.offball_action_idx)).to(device)
        on_ball_action_mask = ~on_ball.unsqueeze(-1).expand(-1, -1, len(self.onball_action_idx)).to(device)

        q_values_masked[:, :, self.offball_action_idx] = torch.where(
            off_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.offball_action_idx]
        )
        q_values_masked[:, :, self.onball_action_idx] = torch.where(
            on_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.onball_action_idx]
        )

        action_loss = F.cross_entropy(
            input=q_values_masked.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values_masked.device) if self.class_weights is not None else None,
        )
        self.val_metrics(
            q_values_masked.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )
        self.log("val_action_loss", action_loss, sync_dist=True)
        self.log_dict(self.val_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.log("val_loss", total_loss, sync_dist=True)
        logging.info(
            f"[Epoch {self.current_epoch} | Batch {batch_idx}] validation total loss: {total_loss.item():.6f}, td loss: {td_loss.item():.6f}, action loss: {action_loss.item():.6f}, l1 loss: {l1_loss.item():.6f}"
        )

        pred_actions = q_values_masked.argmax(dim=2)[batch["mask"]]
        class_counts = torch.bincount(pred_actions, minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Predicted Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Predicted Class Ratios (validation)", fig, self.current_epoch)

        gold_actions = batch["action"][batch["mask"]]
        class_counts = torch.bincount(gold_actions.flatten(), minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Gold Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Gold Class Ratios (validation)", fig, self.current_epoch)

        return total_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mask = batch["mask"]
        reward = batch["reward"]
        action = batch["action"]
        on_ball = batch["onball_mask"]

        q_values = self.forward(batch)
        device = q_values.device

        q_sa = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.forward(batch)[:, 1:].max(dim=2).values
            td_target = reward[:, :-1] + self.gamma * next_q_values

        td_error = td_target - q_sa[:, :-1]
        td_loss = (td_error**2 * mask[:, :-1]).sum() / mask[:, :-1].sum()
        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.log("test_td_loss", td_loss, sync_dist=True)
        self.log("test_l1_loss", l1_loss, sync_dist=True)

        q_values_masked = q_values.clone()
        unavailable_action_value = torch.tensor(UNAVAILABLE_ACTION_QVALUE, device=device)

        off_ball_action_mask = on_ball.unsqueeze(-1).expand(-1, -1, len(self.offball_action_idx)).to(device)
        on_ball_action_mask = ~on_ball.unsqueeze(-1).expand(-1, -1, len(self.onball_action_idx)).to(device)

        q_values_masked[:, :, self.offball_action_idx] = torch.where(
            off_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.offball_action_idx]
        )
        q_values_masked[:, :, self.onball_action_idx] = torch.where(
            on_ball_action_mask, unavailable_action_value, q_values_masked[:, :, self.onball_action_idx]
        )

        action_loss = F.cross_entropy(
            input=q_values_masked.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values_masked.device) if self.class_weights is not None else None,
        )

        self.test_metrics(
            q_values_masked.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )

        self.log("test_action_loss", action_loss, sync_dist=True)
        self.log_dict(self.test_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.log("test_loss", total_loss, sync_dist=True)

        pred_actions = q_values_masked.argmax(dim=2)[batch["mask"]]
        class_counts = torch.bincount(pred_actions, minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Predicted Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Predicted Class Ratios (test)", fig, self.current_epoch)

        gold_actions = batch["action"][batch["mask"]]
        class_counts = torch.bincount(gold_actions.flatten(), minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Gold Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Gold Class Ratios (test)", fig, self.current_epoch)

        return total_loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]] | Optimizer:
        self._optimizer = Optimizer.from_params(params_=self._optimizer_config, params=self.parameters())
        if self._scheduler_config is not None:
            self._scheduler = LRScheduler.from_params(params_=self._scheduler_config, optimizer=self._optimizer)
            return [self._optimizer], [self._scheduler]
        return self._optimizer
