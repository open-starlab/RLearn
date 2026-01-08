import logging
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from rlearn.sports.soccer.dataclass import (
    SimpleObservation_PVS,
    SimpleObservation_EDMS,
)
from rlearn.sports.soccer.constant import ONBALL_ACTION_INDICES
from rlearn.sports.soccer.modules.datamodule.datamodule import DataModule
from rlearn.sports.soccer.modules.state_action_tokenizer.state_action_tokenizer import StateActionTokenizerBase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@DataModule.register("rl_attacker")
class RLAttackerDataModule(DataModule):
    def __init__(
        self,
        state_action_tokenizer: Dict,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_action_tokenizer = StateActionTokenizerBase.from_params(state_action_tokenizer)

        self.onball_action = ONBALL_ACTION_INDICES

    @classmethod
    def preprocess_data(
        cls,
        dataset: torch.utils.data.Dataset,
        state_def: str,
        state_action_tokenizer: Dict[str, Any],
        num_workers: int = 8,
        preprocess_batch_size: int = 32,
    ) -> torch.utils.data.Dataset:
        state_action_tokenizer = StateActionTokenizerBase.from_params(deepcopy(state_action_tokenizer))
        dataset = dataset.map(
            cls._preprocess_data,
            batched=True,
            num_proc=num_workers,
            batch_size=preprocess_batch_size,
            fn_kwargs={"state_action_tokenizer": state_action_tokenizer, "state_def": state_def},
        )
        return dataset

    @classmethod
    def _preprocess_data(
        cls, examples, state_action_tokenizer: StateActionTokenizerBase, state_def: str
    ) -> Dict[str, torch.Tensor]:
        observation_in_examples = []
        action_in_examples = []
        reward_in_examples = []
        for sequence in examples["sequence"]:
            observation_in_example = []  # (seq_len, obs_dim)
            action_in_example = []  # (seq_len,)
            reward_in_example = []  # (seq_len,)
            for observation_action in sequence:
                observation = observation_action["observation"]
                action = observation_action["action"]
                reward = observation_action["reward"]
                assert isinstance(observation, dict), f"observation is not dict: {observation}"
                assert isinstance(action, str), f"action is not str: {action}"
                assert isinstance(reward, float), f"reward is not float: {reward}"
                if state_def == "PVS":
                    observation_in_example.append(SimpleObservation_PVS.from_dict(observation).to_tensor())
                elif state_def == "EDMS":
                    direction = 1
                    observation_in_example.append(SimpleObservation_EDMS.from_dict(observation).to_tensor(direction))
                if action in {"ball_recovery", "interception", "clearance", "pressure", "block"}:
                    action = "defensive_action"
                elif action == "goal":
                    action = "shot"
                action_in_example.append(state_action_tokenizer.encode(action))
                reward_in_example.append(reward)
            observation_in_examples.append(torch.stack(observation_in_example, dim=0))
            action_in_examples.append(torch.tensor(action_in_example, dtype=torch.long))
            reward_in_examples.append(torch.tensor(reward_in_example, dtype=torch.float32))
        return {
            "observation": observation_in_examples,  # List of np.array with (seq_len, obs_dim)
            "action": action_in_examples,  # List of np.array with (seq_len,)
            "reward": reward_in_examples,  # List of np.array with (seq_len,)
        }

    def train_dataloader(
        self,
        batch_size: int | None = None,
        shuffle: bool = True,
        num_workers: int | None = None,
    ) -> torch.utils.data.DataLoader:
        return self.build_dataloader(
            dataset=self.train_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers or self.num_workers,
        )

    def val_dataloader(
        self,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int | None = None,
    ) -> torch.utils.data.DataLoader:
        assert self.valid_dataset is not None, f"valid dataset is not found: {self.valid_dataset}"
        return self.build_dataloader(
            dataset=self.valid_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers or self.num_workers,
        )

    def test_dataloader(
        self,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int | None = None,
    ) -> torch.utils.data.DataLoader:
        assert self.test_dataset is not None, f"test dataset is not found: {self.test_dataset}"
        return self.build_dataloader(
            dataset=self.test_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers or self.num_workers,
        )

    def build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int | None = None,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def batch_collator(self, instances) -> Dict[str, torch.Tensor]:
        max_length = max([len(instance["observation"]) for instance in instances])
        observation_dim = len(instances[0]["observation"][0])
        observation = torch.zeros((len(instances), max_length, observation_dim), dtype=torch.float32)
        action = torch.full(
            (len(instances), max_length),
            self.state_action_tokenizer.encode("[PAD]"),
            dtype=torch.long,
        )
        reward = torch.full(
            (len(instances), max_length),
            0,
            dtype=torch.float32,
        )
        mask = torch.zeros((len(instances), max_length), dtype=torch.bool)
        onball_mask = torch.zeros((len(instances), max_length), dtype=torch.bool)

        for i, instance in enumerate(instances):
            length = len(instance["observation"])
            observation[i, :length] = torch.tensor(instance["observation"], dtype=torch.float32)
            action[i, :length] = torch.tensor(instance["action"], dtype=torch.long)
            reward[i, :length] = torch.tensor(instance["reward"], dtype=torch.float32)
            mask[i, :length] = 1
            for j in range(length):
                action_idx = instance["action"][j]
                onball_mask[i, j] = 1 if action_idx in self.onball_action else 0

        batch = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "mask": mask,
            "onball_mask": onball_mask,
        }

        return batch
