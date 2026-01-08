import logging
from copy import deepcopy
from typing import Any, Dict, cast

import torch

from rlearn.sports.soccer.dataclass import (
    SimpleObservationAction,
)
from rlearn.sports.soccer.modules.datamodule.datamodule import DataModule
from rlearn.sports.soccer.modules.state_action_tokenizer.state_action_tokenizer import StateActionTokenizerBase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@DataModule.register("observation_action_sequence_with_eos")
class SimpleObservationActionSequenceWithEOSDataModule(DataModule):
    def __init__(
        self,
        state_action_tokenizer: Dict[str, Any],
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset | None = None,
        test_dataset: torch.utils.data.Dataset | None = None,
        batch_size: int = 128,
        max_sequence_length: int = 4096,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.state_action_tokenizer = StateActionTokenizerBase.from_params(state_action_tokenizer)
        self.max_sequence_length = max_sequence_length

    @classmethod
    def _preprocess_data(cls, examples, state_action_tokenizer: StateActionTokenizerBase):
        sequence_in_examples = []
        action_mask_in_examples = []
        for sequence in examples["sequence"]:
            sequence_in_example = []
            action_mask_in_example = []
            for observation_action in sequence:
                observation_action = SimpleObservationAction.from_dict(observation_action)
                # [SEP] position velocity position velocity ... [ACTION_SEP] action [SEP] position velocity position velocity ...
                sub_sequence = []
                sub_sequence.append(state_action_tokenizer.encode("[SEP]"))
                for player in observation_action.observation.players:
                    if state_action_tokenizer.encode_position_separately:
                        sub_sequence.extend(
                            [
                                state_action_tokenizer.encode(player.position.x),
                                state_action_tokenizer.encode(player.position.y),
                                state_action_tokenizer.encode(player.velocity),
                            ]
                        )
                    else:
                        sub_sequence.extend(
                            [
                                state_action_tokenizer.encode(player.position),
                                state_action_tokenizer.encode(player.velocity),
                            ]
                        )

                if state_action_tokenizer.encode_position_separately:
                    sub_sequence.extend(
                        [
                            state_action_tokenizer.encode(observation_action.observation.ball.position.x),
                            state_action_tokenizer.encode(observation_action.observation.ball.position.y),
                            state_action_tokenizer.encode(observation_action.observation.ball.velocity),
                            state_action_tokenizer.encode(observation_action.observation.ego_player.position.x),
                            state_action_tokenizer.encode(observation_action.observation.ego_player.position.y),
                        ]
                    )
                else:
                    sub_sequence.extend(
                        [
                            state_action_tokenizer.encode(observation_action.observation.ball.position),
                            state_action_tokenizer.encode(observation_action.observation.ball.velocity),
                            state_action_tokenizer.encode(observation_action.observation.ego_player.position),
                        ]
                    )
                sub_sequence.append(state_action_tokenizer.encode("[ACTION_SEP]"))
                sub_sequence.append(state_action_tokenizer.encode(observation_action.action))
                sequence_in_example.extend(sub_sequence)
                action_mask_in_example.extend([0] * (len(sub_sequence) - 1) + [1])

            sequence_in_example.append(state_action_tokenizer.encode("[EOS]"))
            action_mask_in_example.append(0)
            sequence_in_examples.append(sequence_in_example)
            action_mask_in_examples.append(action_mask_in_example)
        assert len(sequence) != 0, f"sequence is empty: {sequence}"
        return {
            "input_ids": sequence_in_examples,
            "action_mask": action_mask_in_examples,
        }

    @classmethod
    def preprocess_data(
        cls,
        dataset: torch.utils.data.Dataset,
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
            fn_kwargs={
                "state_action_tokenizer": state_action_tokenizer,
            },
        )
        # dataset = dataset.map(
        #     self._truncate_data,
        #     batched=True,
        #     num_proc=self.num_workers,
        # )
        return dataset

    # def _truncate_data(self, examples):
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, and if the total_length < max_seq_len  we exclude this batch and return an empty dict. # noqa
    #     # We could add padding if the model supported it instead of this drop, you can customize this part to your needs. # noqa
    #     total_length = (total_length // self.max_sequence_length) * self.max_sequence_length
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + self.max_sequence_length] for i in range(0, total_length, self.max_sequence_length)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     return result

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
        max_length = max(len(instance["input_ids"]) for instance in instances)
        input_ids = cast(
            torch.LongTensor, torch.full((len(instances), max_length), self.state_action_tokenizer.encode("[PAD]"))
        )
        mask = torch.zeros((len(instances), max_length), dtype=torch.long)
        action_mask = torch.zeros((len(instances), max_length), dtype=torch.long)

        for i, instance in enumerate(instances):
            length = len(instance["input_ids"])
            input_ids[i, :length] = torch.tensor(instance["input_ids"])
            mask[i, :length] = 1
            action_mask[i, :length] = torch.tensor(instance["action_mask"])

        batch = {
            "input_ids": input_ids,
            "mask": mask,
            "action_mask": action_mask,
        }
        return batch
