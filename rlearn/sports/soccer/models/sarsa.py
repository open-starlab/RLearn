from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlearn.sports.soccer.models.q_model_base import QModelBase
from rlearn.sports.soccer.modules.optimizer import LRScheduler, Optimizer
from rlearn.sports.soccer.modules.seq2seq_encoder import Seq2SeqEncoder
from rlearn.sports.soccer.modules.token_embedder import TokenEmbedder
from rlearn.sports.soccer.torch.metrics.classification import get_classification_full_metrics


class Prediction:
    def __init__(self, q_values: torch.Tensor) -> None:
        self.q_values = q_values


@QModelBase.register("sarsa_attacker")
class AttacckerSARSAModel(QModelBase):
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
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.vocab_size = vocab_size
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self.pad_token_id = pad_token_id
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lambda2_ = lambda2_
        self.class_weights = class_weights
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None

        self.encoder = Seq2SeqEncoder.from_params(sequence_encoder)
        self.fc1 = nn.Linear(self.observation_dim, self.encoder.get_input_dim())
        self.fc2 = nn.Linear(self.encoder.get_output_dim(), self.vocab_size)

        # evaluation
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
        # observation: (batch_size, seq_len, obs_dim)
        x = F.relu(self.fc1(batch["observation"]))  # (batch_size, input_length, encoder_input_dim)
        out = self.encoder(x)
        q_values = self.fc2(out)  # (batch_size, input_length, vocab_size)

        # if q_values contains NaN values, print the indices of the NaN values and the values themselves
        if torch.isnan(q_values).any():
            print("q_values contains NaN values")
            nan_indices = torch.nonzero(torch.isnan(q_values)).squeeze()
            for idx in nan_indices:
                idx_tuple = tuple(idx.tolist())
                print(f"NaN at index: {idx_tuple}")
                print(f"q_values[{idx_tuple}] = {q_values[idx_tuple]}")
                print(f"batch['observation'][{idx_tuple[:-1]}] = {batch['observation'][idx_tuple[:-1]]}")
                print(f"batch['action'][{idx_tuple[:-1]}] = {batch['action'][idx_tuple[:-1]]}")
                print(f"batch['reward'][{idx_tuple[:-1]}] = {batch['reward'][idx_tuple[:-1]]}")
                print(f"batch['mask'][{idx_tuple[:-1]}] = {batch['mask'][idx_tuple[:-1]]}")
        return q_values

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # avail_actions = []
        mask = batch["mask"]  # (batch_size, input_length)
        reward = batch["reward"]  # (batch_size, input_length)
        action = batch["action"]  # (batch_size, input_length)

        q_values = self.forward(batch)  # (batch_size, input_length, vocab_size)
        pred_q_values = q_values.argmax(dim=2)  # (batch_size, input_length)
        td_loss = (reward[:, 1:] + self.gamma * pred_q_values[:, 1:] - pred_q_values[:, :-1]) ** 2
        td_loss = (td_loss * mask[:, 1:]).sum() / mask[:, 1:].sum()
        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.log("train_td_loss", td_loss)
        self.log("train_l1_loss", l1_loss)

        action_loss = F.cross_entropy(
            input=q_values.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values.device) if self.class_weights is not None else None,
        )

        self.train_metrics(
            q_values.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )
        self.log("train_action_loss", action_loss)
        self.log_dict(self.train_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        mask = batch["mask"]  # (batch_size, input_length)
        reward = batch["reward"]  # (batch_size, input_length)
        action = batch["action"]  # (batch_size, input_length)

        q_values = self.forward(batch)  # (batch_size, input_length, vocab_size)
        pred_q_values = q_values.argmax(dim=2)  # (batch_size, input_length)
        td_loss = (reward[:, 1:] + self.gamma * pred_q_values[:, 1:] - pred_q_values[:, :-1]) ** 2
        td_loss = (td_loss * mask[:, 1:]).sum() / mask[:, 1:].sum()
        # L1 loss for parameter regularization
        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.log("val_td_loss", td_loss)
        self.log("val_l1_loss", l1_loss)

        action_loss = F.cross_entropy(
            input=q_values.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values.device) if self.class_weights is not None else None,
        )
        self.val_metrics(
            q_values.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )
        self.log("val_action_loss", action_loss)
        self.log_dict(self.val_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.log("val_loss", total_loss)

        # log prediction count as a histogram
        pred_actions = q_values.argmax(dim=2)[batch["mask"]]
        class_counts = torch.bincount(pred_actions, minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Predicted Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Predicted Class Ratios (validation)", fig, self.current_epoch)

        # log gold count as a histogram
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
        mask = batch["mask"]  # (batch_size, input_length)
        reward = batch["reward"]  # (batch_size, input_length)
        action = batch["action"]  # (batch_size, input_length)

        q_values = self.forward(batch)  # (batch_size, input_length, vocab_size)
        self.check_for_nan_inf(q_values, "q_values")

        pred_q_values = q_values.argmax(dim=2)  # (batch_size, input_length)
        self.check_for_nan_inf(pred_q_values, "pred_q_values")

        td_loss = (reward[:, 1:] + self.gamma * pred_q_values[:, 1:] - pred_q_values[:, :-1]) ** 2
        self.check_for_nan_inf(td_loss, "td_loss before mask")
        td_loss = (td_loss * mask[:, 1:]).sum() / mask[:, 1:].sum()
        self.check_for_nan_inf(td_loss, "td_loss after mask")

        l1_loss = sum([torch.norm(param, p=1) for param in self.parameters() if param.requires_grad])
        self.check_for_nan_inf(l1_loss, "l1_loss")
        self.log("test_td_loss", td_loss)
        self.log("test_l1_loss", l1_loss)

        action_loss = F.cross_entropy(
            input=q_values.reshape(-1, self.vocab_size),
            target=action.reshape(-1),
            reduction="mean",
            ignore_index=self.pad_token_id,
            weight=self.class_weights.to(q_values.device) if self.class_weights is not None else None,
        )
        self.check_for_nan_inf(action_loss, "action_loss")

        # import pdb; pdb.set_trace()

        self.test_metrics(
            q_values.reshape(-1, self.vocab_size),
            action.reshape(-1),
        )
        self.log("test_action_loss", action_loss)
        self.log_dict(self.test_metrics)

        total_loss = td_loss + self.lambda_ * l1_loss + self.lambda2_ * action_loss
        self.check_for_nan_inf(total_loss, "total_loss")
        self.log("test_loss", total_loss)

        # log prediction count as a histogram
        pred_actions = q_values.argmax(dim=2)[batch["mask"]]
        class_counts = torch.bincount(pred_actions, minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Predicted Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Predicted Class Ratios (test)", fig, self.current_epoch)

        # log gold count as a histogram
        gold_actions = batch["action"][batch["mask"]]
        class_counts = torch.bincount(gold_actions, minlength=self.vocab_size).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        fig, ax = plt.subplots()
        ax.bar(range(self.vocab_size), class_ratios)
        ax.set_xticks(range(self.vocab_size))
        ax.set_xlabel("Classes")
        ax.set_ylabel("Gold Ratio")
        if self.logger is not None:
            self.logger.experiment.add_figure("Gold Class Ratios (test)", fig, self.current_epoch)

        return total_loss

    def check_for_nan_inf(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values")

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]] | Optimizer:
        self._optimizer = Optimizer.from_params(params_=self._optimizer_config, params=self.parameters())
        if self._scheduler_config is not None:
            self._scheduler = LRScheduler.from_params(params_=self._scheduler_config, optimizer=self._optimizer)
            return [self._optimizer], [self._scheduler]
        return self._optimizer
