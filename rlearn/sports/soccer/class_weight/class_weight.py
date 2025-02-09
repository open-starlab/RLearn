from typing import List, Optional

import torch
from tango.common import Registrable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ClassWeightBase(Registrable):
    def __init__(self, cache_path: str | Path | None = None, **kwargs):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.class_weights: torch.Tensor | None = None
        if cache_path is not None:
            if self.cache_path.exists():
                logger.info(f"Loading class weights from {self.cache_path}")
                self.class_weights = torch.load(self.cache_path)
            else:
                logger.info(f"No cache found at {self.cache_path}")

    def calculate(
        self,
        class_counts: List[int],
        ignore_indices: List[int] | None = None,
    ) -> torch.Tensor:
        if self.class_weights is not None:
            return self.class_weights

        self.class_weights = self._calculate(class_counts, ignore_indices)
        if self.cache_path is not None:
            logger.info(f"Saving class weights to {self.cache_path}")
            self.cache_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.class_weights, self.cache_path)
        return self.class_weights


    def _calculate(
        self,
        class_counts: List[int],
        ignore_indices: List[int] | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

@ClassWeightBase.register("reciprocal")
class ClassWeightReciprocal(ClassWeightBase):
    def _calculate(
        self,
        class_counts: List[int],
        ignore_indices: List[int] | None = None,
    ) -> torch.Tensor:
        class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = 1 / class_weights[class_idx]

        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0

        return class_weights


@ClassWeightBase.register("sklearn")
class ClassWeightSklearn(ClassWeightBase):
    def _calculate(
        self,
        class_counts: List[int], # (num_classes,)
        ignore_indices: List[int] | None = None,
    ) -> torch.Tensor:
        class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
        event_len = class_weights.sum().item()
        num_classes = len(class_weights)
        # cf: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = event_len / (
                class_weights[class_idx] * num_classes
            )
        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0
        return class_weights



@ClassWeightBase.register("exponential")
class ClassWeightExponential(ClassWeightBase):
    def __init__(self, cache_path: str | Path | None = None, beta: float = 0.9) -> None:
        super().__init__(cache_path)
        self.beta = beta

    def _calculate(
        self,
        class_counts: List[int],
        ignore_indices: List[int] | None = None,
    ) -> torch.Tensor:
        class_weights = torch.tensor(class_counts, dtype=torch.float) + 1
        sum_exponentials = ((1 / class_weights) ** self.beta).sum().item()
        for class_idx in range(len(class_weights)):
            class_weights[class_idx] = (
                (1 / class_weights[class_idx]) ** self.beta
            ) / sum_exponentials

        # ignore_indices
        if ignore_indices is not None:
            for ignore_idx in ignore_indices:
                class_weights[ignore_idx] = 0
        return class_weights
