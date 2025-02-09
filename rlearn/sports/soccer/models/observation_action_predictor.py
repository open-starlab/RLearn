from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tango.common import Registrable

from ..dataclass import ObservationActionForLMBatch
from ..modules.optimizer import LRScheduler, Optimizer
from ..modules.sequence_decoder import SequenceDecoder
from ..modules.token_embedder import TokenEmbedder


class ObservationActionPredictorBase(pl.LightningModule, Registrable):
    def training_step(self, batch: ObservationActionForLMBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self, batch: ObservationActionForLMBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def test_step(self, batch: ObservationActionForLMBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]] | Optimizer:
        raise NotImplementedError


@ObservationActionPredictorBase.register("simple")
class SimpleObservationActionPredictor(ObservationActionPredictorBase):
    def __init__(
        self,
        token_embedder: Dict[str, Any],
        sequence_decoder: Dict[str, Any],
        optimizer: Dict[str, Any],
        scheduler: Dict[str, Any] | None = None,
        action_coeff: float = 1.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.token_embedder = TokenEmbedder.from_params(token_embedder)
        self.sequence_decoder = SequenceDecoder.from_params(sequence_decoder)
        self.token_projection = torch.nn.Linear(
            self.sequence_decoder.get_output_dim(),
            self.token_embedder.get_input_dim(),
            bias=False,
        )
        self.token_projection.weight = self.token_embedder.embedding.weight
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self.action_coeff = action_coeff
        self.class_weights = class_weights

    def forward(self, batch: ObservationActionForLMBatch) -> torch.Tensor:
        """
        Args:
            batch: ObservationActionForLMBatch
        Returns:
            output: (batch_size, seq_len, input_dim)
        """
        mask = batch['mask']
        inputs = self.token_embedder(batch['input_ids'])
        output = self.sequence_decoder(inputs, mask)
        output = self.token_projection(output)
        return output

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self(batch)
        target = batch['input_ids'][:, 1:].contiguous()
        loss = F.cross_entropy(
            output[:, :-1].contiguous().view(-1, output.shape[-1]),
            target.view(-1),
            reduction='mean',
            ignore_index=self.token_embedder.padding_idx,
            weight=self.class_weights.to(output.device) if self.class_weights is not None else None,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self(batch)
        target = batch['input_ids'][:, 1:].contiguous()
        action_mask = batch['action_mask'][:, 1:].contiguous()
        loss = F.cross_entropy(
            output[:, :-1].contiguous().view(-1, output.shape[-1]),
            target.view(-1),
            reduction='mean',
            ignore_index=self.token_embedder.padding_idx,
            weight=self.class_weights.to(output.device) if self.class_weights is not None else None,
        )
        self.log("val_loss", loss)

        num_classes = self.token_embedder.get_input_dim()
        output = output[:, :-1].contiguous().view(-1, output.shape[-1])  # (batch_size * (seq_len - 1), vocab_size)
        pred = output.argmax(dim=-1)  # (batch_size * (seq_len - 1), )
        pred_actions = pred[action_mask.view(-1) == 1]  # (num_actions, )
        class_counts = torch.bincount(pred_actions.flatten(), minlength=num_classes).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        table_text = "| Class | Ratio |\n|-------|-------|\n"
        for i, ratio in enumerate(class_ratios):
            table_text += f"| {i} | {ratio:.2f} |\n"
        self.logger.experiment.add_text("Predicted Class Ratios (valid)", table_text, self.current_epoch)

        gold_actions = batch['input_ids'][batch['action_mask'] == 1]  # (num_actions, )
        gold_class_counts = torch.bincount(gold_actions.flatten(), minlength=num_classes).cpu().numpy()
        gold_class_ratios = gold_class_counts / gold_class_counts.sum()
        table_text = "| Class | Ratio |\n|-------|-------|\n"
        for i, ratio in enumerate(gold_class_ratios):
            table_text += f"| {i} | {ratio:.2f} |\n"
        self.logger.experiment.add_text("Gold Class Ratios (valid)", table_text, self.current_epoch)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        output = self(batch)
        target = batch['input_ids'][:, 1:].contiguous()
        action_mask = batch['action_mask'][:, 1:].contiguous()
        loss = F.cross_entropy(
            output[:, :-1].contiguous().view(-1, output.shape[-1]),
            target.view(-1),
            reduction='mean',
            ignore_index=self.token_embedder.padding_idx,
            weight=self.class_weights.to(output.device) if self.class_weights is not None else None,
        )
        self.log("test_loss", loss)

        num_classes = self.token_embedder.get_input_dim()
        output = output[:, :-1].contiguous().view(-1, output.shape[-1])  # (batch_size * (seq_len - 1), vocab_size)
        pred = output.argmax(dim=-1)  # (batch_size * (seq_len - 1), )
        pred_actions = pred[action_mask.view(-1) == 1]  # (num_actions, )
        class_counts = torch.bincount(pred_actions.flatten(), minlength=num_classes).cpu().numpy()
        class_ratios = class_counts / class_counts.sum()
        table_text = "| Class | Ratio |\n|-------|-------|\n"
        for i, ratio in enumerate(class_ratios):
            table_text += f"| {i} | {ratio:.2f} |\n"
        self.logger.experiment.add_text("Predicted Class Ratios (test)", table_text, self.current_epoch)

        gold_actions = batch['input_ids'][batch['action_mask'] == 1]  # (num_actions, )
        gold_class_counts = torch.bincount(gold_actions.flatten(), minlength=num_classes).cpu().numpy()
        gold_class_ratios = gold_class_counts / gold_class_counts.sum()
        table_text = "| Class | Ratio |\n|-------|-------|\n"
        for i, ratio in enumerate(gold_class_ratios):
            table_text += f"| {i} | {ratio:.2f} |\n"
        self.logger.experiment.add_text("Gold Class Ratios (test)", table_text, self.current_epoch)

        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]] | Optimizer:
        self._optimizer = Optimizer.from_params(params_=self._optimizer_config, params=self.parameters())
        if self._scheduler_config is not None:
            self._scheduler = LRScheduler.from_params(params_=self._scheduler_config, optimizer=self._optimizer)
            return [self._optimizer], [self._scheduler]
        return self._optimizer
