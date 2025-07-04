import logging
import re
from pathlib import Path
import os
from typing import Any, Dict
from copy import deepcopy
import time
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch
from lightning_lite.utilities.seed import seed_everything


from ..dataclass import (
    Events_PVS,
    Events_EDMS,
    SimpleObservation_PVS,
    SimpleObservation_EDMS,
    SimpleObservationAction_PVS,
    SimpleObservationAction_EDMS,
    SimpleObservationActionSequence_PVS,
    SimpleObservationActionSequence_EDMS,
)
from ..utils.file_utils import load_json, save_formatted_json
from ..env import OUTPUT_DIR, PROJECT_DIR
from ..models.q_model_base import QModelBase
from ..modules.datamodule import DataModule
from ..class_weight.class_weight import ClassWeightBase
from ..application.q_values_movie import create_movie, save_q_values_to_csv


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class rlearn_model_soccer:
    def __init__(
        self,
        state_def,
        model_name=None,
        config=None,
        seed=42,
        num_process=4,
        input_path=None,
        output_path=None,
    ):
        self.model_name = model_name
        self.state_def = state_def
        self.config = config
        self.seed = seed
        self.num_process = num_process
        self.input_path = input_path
        self.output_path = output_path

    def split_train_test(self, pytest=False):
        # Load data into a Dataset

        output_dir = Path(self.output_path)
        # Set output directory
        if output_dir is None:
            output_dir = self.input_path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if pytest:
            game_ids = [str(p.name) for p in Path(self.input_path).glob("*") if re.match(r"\d{4}", p.name)]
            train_dataset = load_dataset(
                "json",
                data_files=[str(Path(self.input_path) / f"{game_id}" / "events.jsonl") for game_id in game_ids],
                split="train",
                num_proc=self.num_process,
            )
        else:
            game_ids = [str(p.name) for p in Path(self.input_path).glob("*") if re.match(r"\d{7}", p.name)]
            train_game_ids, test_val_game_ids = train_test_split(game_ids, test_size=0.5, random_state=self.seed)
            test_game_ids, val_game_ids = train_test_split(test_val_game_ids, test_size=0.1, random_state=self.seed)

            train_dataset = load_dataset(
                "json",
                data_files=[str(Path(self.input_path) / f"{game_id}" / "events.jsonl") for game_id in train_game_ids],
                split="train",
                num_proc=self.num_process,
            )
            valid_dataset = load_dataset(
                "json",
                data_files=[str(Path(self.input_path) / f"{game_id}" / "events.jsonl") for game_id in val_game_ids],
                split="train",
                num_proc=self.num_process,
            )
            test_dataset = load_dataset(
                "json",
                data_files=[str(Path(self.input_path) / f"{game_id}" / "events.jsonl") for game_id in test_game_ids],
                split="train",
                num_proc=self.num_process,
            )

            # Save the splits
            for split_name, split_dataset in zip(
                ["train", "validation", "test"],
                [train_dataset, valid_dataset, test_dataset],
            ):
                split_dataset.save_to_disk(output_dir / split_name)

            logging.info(f"Data splits saved in {output_dir}:")
            logging.info(f"Train: {len(train_dataset)}")
            logging.info(f"Validation: {len(valid_dataset)}")
            logging.info(f"Test: {len(test_dataset)}")

        # for debugging
        train_dataset.select(range(5)).save_to_disk(output_dir / "mini")

    def events2attacker_simple_observation_action_sequence(
        self, examples, min_frame_len_threshold: int = 30, max_frame_len_threshold: int = 500, num_offball_players: int = 3
    ) -> Dict[str, Any]:
        if self.state_def == "PVS":
            Events = Events_PVS
            SimpleObservation = SimpleObservation_PVS
            SimpleObservationAction = SimpleObservationAction_PVS
            SimpleObservationActionSequence = SimpleObservationActionSequence_PVS
        elif self.state_def == "EDMS":
            Events = Events_EDMS
            SimpleObservation = SimpleObservation_EDMS
            SimpleObservationAction = SimpleObservationAction_EDMS
            SimpleObservationActionSequence = SimpleObservationActionSequence_EDMS
        else:
            raise ValueError(f"Unsupported state definition: {self.state_def}")

        events_list = [Events.from_dict(dict(zip(examples, v))) for v in zip(*examples.values())]
        for events in events_list:
            assert min_frame_len_threshold <= len(events.events) <= max_frame_len_threshold, (
                f"len(events.events): {len(events.events)}"
            )
        attacker_observation_action_sequence = []
        for events in events_list:
            if self.state_def == "PVS":
                valid_attack_player_ids = [
                    player.player_id
                    for player in events.events[0].state.players
                    if player.player_role != "GK" and player.player_id > 0 and player.team_name == events.team_name_attack
                ]
            elif self.state_def == "EDMS":
                valid_attack_player_ids = [
                    player.player_id
                    for player in events.events[0].state.raw_state.players
                    if player.player_role != "GK" and player.player_id > 0 and player.team_name == events.team_name_attack
                ]
            else:
                raise ValueError(f"Unsupported state definition: {self.state_def}")

            if len(valid_attack_player_ids) != 10:
                logger.warning(
                    f"Found onlu {len(valid_attack_player_ids)} valid attack players in game_id: {events.game_id}, half: {events.half}, sequence_id: {events.sequence_id}. "
                )

            if self.state_def == "EDMS":
                onball_list = None
                previous_attack_team = events.team_name_attack
                onball_list = [0] * len(events.events)
                attack_action = ["pass", "dribble", "shot", "through_pass", "cross"]
                last_attack_index = -1

                for i, event in enumerate(events.events):
                    for player_index, player in enumerate(event.state.raw_state.players):
                        if player.action in attack_action:
                            onball_list[i] = player_index
                            current_team = player.team_name
                            if current_team == previous_attack_team and last_attack_index >= 0:
                                last_player_index = onball_list[last_attack_index]
                                for j in range(last_attack_index + 1, i):
                                    onball_list[j] = last_player_index
                            previous_attack_team = current_team
                            last_attack_index = i
                            break

            for number_of_player, target_player_id in enumerate(valid_attack_player_ids):
                attacker_observation_action_sequence_in_event = []

                for number_of_event, event in enumerate(events.events):
                    try:
                        if self.state_def == "PVS":
                            target_player = [
                                player for player in event.state.attack_players if player.player_id == target_player_id
                            ][0]
                        elif self.state_def == "EDMS":
                            target_player = [
                                player
                                for player in event.state.raw_state.attack_players
                                if player.player_id == target_player_id
                            ][0]
                        else:
                            raise ValueError(f"Unsupported state definition: {self.state_def}")
                    except (IndexError, Exception):
                        logger.warning(
                            f"target_player_id: {target_player_id} not found in game_id: {events.game_id} half: {events.half} seq_id: {events.sequence_id}"
                        )
                        continue

                    if self.state_def == "PVS":
                        observation = SimpleObservation.from_state(event.state, target_player)
                    elif self.state_def == "EDMS":
                        gk_idx = next(
                            (i for i, player in enumerate(event.state.raw_state.attack_players) if player.player_role == "GK"),
                            -1,
                        )
                        observation = SimpleObservation.from_state(
                            event.state,
                            target_player,
                            target_player_id,
                            gk_idx,
                            number_of_player,
                            num_offball_players,
                            onball_list,
                            number_of_event,
                            self.state_def,
                        )
                    else:
                        raise ValueError(f"Unsupported state definition: {self.state_def}")

                    action = target_player.action
                    observation_action = SimpleObservationAction(
                        player=target_player, observation=observation, action=action, reward=event.reward
                    )
                    attacker_observation_action_sequence_in_event.append(observation_action)

                attacker_observation_action_sequence.append(attacker_observation_action_sequence_in_event)

        if self.state_def == "EDMS":
            events_team = [events.team_name_attack for events in events_list]
            for observation_action_sequence in attacker_observation_action_sequence:
                if (
                    observation_action_sequence
                    and observation_action_sequence[0].observation.ego_player.team_name not in events_team
                ):
                    logger.warning(
                        f"ego player is not attacker: {observation_action_sequence[0].observation.ego_player.team_name}"
                    )

        num_attacker = len(attacker_observation_action_sequence)

        attacker_observation_action_sequence = [
            SimpleObservationActionSequence(
                game_id=events_list[i // (num_attacker // len(events_list))].game_id,
                half=events_list[i // (num_attacker // len(events_list))].half,
                sequence_id=events_list[i // (num_attacker // len(events_list))].sequence_id,
                team_name_attack=events_list[i // (num_attacker // len(events_list))].team_name_attack,
                team_name_defense=events_list[i // (num_attacker // len(events_list))].team_name_defense,
                sequence=observation_action_sequence,
            ).to_dict()
            for i, observation_action_sequence in enumerate(attacker_observation_action_sequence)
        ]

        assert len(attacker_observation_action_sequence) <= len(events_list) * num_attacker, (
            f"len(attacker_observation_action_sequence): {len(attacker_observation_action_sequence)} (len(events_list): {len(events_list)})"
        )

        return {
            key: [item[key] for item in attacker_observation_action_sequence]
            for key in attacker_observation_action_sequence[0].keys()
        }

    def preprocess_observation(self, batch_size):
        logging.info(f"input_path: {self.input_path}")
        start = time.time()
        config = load_json(self.config)
        dataset = load_from_disk(str(self.input_path))
        logger.info("Length of dataset: {}".format(len(dataset)))
        dataset = dataset.map(
            self.events2attacker_simple_observation_action_sequence,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_process,
            batch_size=batch_size,
            fn_kwargs={
                "min_frame_len_threshold": config["min_frame_len_threshold"],
                "max_frame_len_threshold": config["max_frame_len_threshold"],
                "num_offball_players": config["num_offball_players"],
            },
        )
        logger.info("Length of dataset after processing: {}".format(len(dataset)))
        dataset.save_to_disk(str(self.output_path))
        logging.info(f"output_path: {self.output_path} (elapsed: {time.time() - start:.2f} sec)")

    def train(
        self,
        exp_name,
        run_name,
        accelerator,
        devices,
        strategy,
        save_q_values_csv=True,
        max_games_csv=1,
        max_sequences_per_game_csv=5,
        mlflow=True,
    ):
        seed_everything(self.seed)
        exp_config = load_json(self.config)
        config_copy = deepcopy(exp_config)
        logger.info(f"exp_config: {exp_config}")
        output_dir = OUTPUT_DIR / exp_name / run_name
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info("loading dataset...")
        train_dataset = load_from_disk(Path(exp_config["dataset"]["train_filename"]).resolve())
        valid_dataset = load_from_disk(Path(exp_config["dataset"]["valid_filename"]).resolve())
        test_dataset = load_from_disk(Path(exp_config["dataset"]["test_filename"]).resolve())
        logger.info("Preprocessing dataset...")
        start = time.time()
        train_dataset = DataModule.by_name(exp_config["datamodule"]["type"]).preprocess_data(
            train_dataset, self.state_def, **exp_config["dataset"]["preprocess_config"]
        )
        valid_dataset = DataModule.by_name(exp_config["datamodule"]["type"]).preprocess_data(
            valid_dataset, self.state_def, **exp_config["dataset"]["preprocess_config"]
        )
        test_dataset = DataModule.by_name(exp_config["datamodule"]["type"]).preprocess_data(
            test_dataset, self.state_def, **exp_config["dataset"]["preprocess_config"]
        )
        logger.info(f"Preprocessing dataset is done. {time.time() - start} sec")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Valid dataset size: {len(valid_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")

        datamodule = DataModule.from_params(
            params_=exp_config["datamodule"],
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )
        # count tokens and calculate class weights (the inverse of the frequency of each class)
        # cache the class weights so that we need not calculate them every time
        if "class_weight_fn" in exp_config:
            logger.info("Prepare class weights...")
            start = time.time()
            class_weight_fn = ClassWeightBase.from_params(exp_config["class_weight_fn"])
            if class_weight_fn.class_weights is not None:
                class_weights = class_weight_fn.class_weights
            else:
                logger.info("Calculating class weights...")
                class_counts = torch.zeros(datamodule.state_action_tokenizer.num_tokens)
                for batch in tqdm(datamodule.train_dataloader(batch_size=512), desc="calculating class weights"):
                    valid_actions = torch.masked_select(batch["action"], batch["mask"].bool())
                    class_counts += torch.bincount(valid_actions, minlength=datamodule.state_action_tokenizer.num_tokens)
                class_weights = class_weight_fn.calculate(class_counts=class_counts)
            assert class_weights.shape[0] == datamodule.state_action_tokenizer.num_tokens, (
                f"Class weights shape mismatch: {class_weights.shape[0]} != {datamodule.state_action_tokenizer.num_tokens}"
            )
            logger.info(f"Prepare class weights is done. {time.time() - start} sec")
        else:
            class_weights = None

        if mlflow:
            mlflow_logger = pl.loggers.MLFlowLogger(
                experiment_name=exp_name,
                run_name=run_name,
                save_dir=str(PROJECT_DIR / "mlruns"),
            )
        else:
            mlflow_logger = False

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.01,
        )
        trainer = pl.Trainer(
            max_epochs=exp_config["max_epochs"],
            logger=mlflow_logger,
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            deterministic=True,
            val_check_interval=exp_config["val_check_interval"] if "val_check_interval" in exp_config else None,
            detect_anomaly=True,
            accumulate_grad_batches=exp_config["accumulate_grad_batches"] if "accumulate_grad_batches" in exp_config else 1,
            gradient_clip_val=None,
        )

        params_ = {
            "type": exp_config["model"]["type"],
            "observation_dim": exp_config["model"]["observation_dim"],
            "sequence_encoder": exp_config["model"]["sequence_encoder"],
            "optimizer": exp_config["model"]["optimizer"],
            "vocab_size": datamodule.state_action_tokenizer.num_tokens,
            "pad_token_id": datamodule.state_action_tokenizer.encode("[PAD]"),
            "gamma": exp_config["model"]["gamma"],
            "lambda_": exp_config["model"]["lambda_"],
            "lambda2_": exp_config["model"]["lambda2_"],
            "class_weights": class_weights,
        }
        params_["class_weights"] = params_["class_weights"].tolist()

        print(f"params_: {params_}")
        print("class_weights:", params_["class_weights"])

        model = QModelBase.from_params(params_=params_)

        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())
        save_formatted_json(config_copy, output_dir / "config.json")

        # Save Q-values to CSV if requested
        if save_q_values_csv:
            model_name = exp_config.split("/")[-1].split(".")[0]
            save_dir = OUTPUT_DIR / "figures" / model_name
            save_q_values_to_csv(
                model=model,
                datamodule=datamodule,
                state_def=self.state_def,
                output_dir=save_dir,
                max_games=max_games_csv,
                max_sequences_per_game=max_sequences_per_game_csv,
            )

    def visualize_data(self, model_name, checkpoint_path, match_id, sequence_id):
        exp_config_path = os.getcwd() + f"/test/config/{model_name}.json"
        exp_config = load_json(exp_config_path)
        test_file_path = Path(os.getcwd() + "/" + exp_config["dataset"]["test_filename"])
        test_dataset = load_from_disk(test_file_path)
        test_dataset = DataModule.by_name(exp_config["datamodule"]["type"]).preprocess_data(
            test_dataset, **exp_config["dataset"]["preprocess_config"]
        )

        # unique game_ids
        game_ids = set(test_dataset["game_id"])

        for game_id_to_find in game_ids:
            indices = [i for i, game_id in enumerate(test_dataset["game_id"]) if game_id == game_id_to_find]
            sequence_ids = [test_dataset["sequence_id"][i] for i in indices]
            unique_sequence_ids = set(sequence_ids)

            print(f"Unique sequence_ids for game_id {game_id_to_find}: {unique_sequence_ids}")

        # import pdb; pdb.set_trace()

        print(f"start loading {match_id} {sequence_id}")

        datamodule = DataModule.from_params(
            exp_config["datamodule"],
            train_dataset=test_dataset,
            valid_dataset=None,
            test_dataset=None,
        )

        type_ = exp_config["model"]["type"]
        observation_dim = exp_config["model"]["observation_dim"]
        sequence_encoder = exp_config["model"]["sequence_encoder"]
        optimizer = exp_config["model"]["optimizer"]
        model = QModelBase.from_params(
            params_={
                "type": type_,
                "observation_dim": observation_dim,
                "sequence_encoder": sequence_encoder,
                "vocab_size": datamodule.state_action_tokenizer.num_tokens,
                "optimizer": optimizer,
                "gamma": exp_config["model"]["gamma"],
                "lambda_": exp_config["model"]["lambda_"],
                "lambda2_": exp_config["model"]["lambda2_"],
            }
        )
        state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        model.to("cuda")

        q_values_list = []

        for data, batch in tqdm(
            zip(datamodule.train_dataset, datamodule.train_dataloader(batch_size=1, shuffle=False)),
            total=len(datamodule.train_dataset),
        ):
            q_values_df = pd.DataFrame(
                index=range(len(data["sequence"])),
                columns=[
                    "game_id",
                    "sequence_id",
                    "team_name",
                    "player_name",
                    "q_value",
                    "action_idx",
                    "q_values_for_actions",
                ],
            )
            q_values_df_path = (
                PROJECT_DIR / f"output/figures/{model_name}/players_q_state/q_values_{match_id}_{sequence_id}.csv"
            )
            q_values_df_path.parent.mkdir(parents=True, exist_ok=True)

            if data["game_id"] == match_id and data["sequence_id"] == sequence_id:
                player = data["sequence"][0]["player"]
                q_values = (
                    model(datamodule.transfer_batch_to_device(batch, "cuda", 0)).squeeze(0).detach().cpu()
                )  # (seq_len, num_actions)
                action_idx = batch["action"].squeeze(0)  # (seq_len, )

                # gather q_values for the actions taken
                q_values_for_actions = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1).tolist()  # len = seq_len

                for i, _ in enumerate(data["sequence"]):
                    q_values_df.loc[i, "game_id"] = data["game_id"]
                    q_values_df.loc[i, "sequence_id"] = data["sequence_id"]
                    q_values_df.loc[i, "player_name"] = player["player_name"]
                    q_values_df.loc[i, "q_value"] = q_values[i, :]
                    q_values_df.loc[i, "action_idx"] = action_idx[i]
                    q_values_df.loc[i, "q_values_for_actions"] = q_values_for_actions[i]

            else:
                continue

            q_values_list.append(q_values_df)

            final_q_values_df = pd.concat(q_values_list, ignore_index=True)
            final_q_values_df.to_csv(q_values_df_path, index=False)

        create_movie(
            q_values_path=q_values_df_path,
            match_id=match_id,
            sequence_id=sequence_id,
        )


if __name__ == "__main__":
    import os

    # # split data into train and test (PVS)
    # rlearn_model_soccer(
    #     state_def="PVS",
    #     input_path=os.getcwd() + "/tests/data/datastadium/",
    #     output_path=os.getcwd() + "/tests/data/datastadium/split/",
    # ).split_train_test()

    # # preprocess observation (PVS)
    # rlearn_model_soccer(
    #     state_def="PVS",
    #     config=os.getcwd() + "/tests/config/preprocessing_dssports2020.json",
    #     input_path=os.getcwd() + "/tests/data/datastadium/split/mini",
    #     output_path=os.getcwd() + "/tests/data/datastadium_simple_obs_action_seq/split/mini",
    #     num_process=5,
    # ).preprocess_observation(batch_size=64)

    # # train model (PVS)
    # rlearn_model_soccer(state_def="PVS", config=os.getcwd() + "/tests/config/exp_config.json").train(
    #     exp_name="sarsa_attacker",
    #     run_name="test",
    #     accelerator="gpu",
    #     devices=1,
    #     strategy="ddp",
    # )

    # # visualize data (PVS)
    # rlearn_model_soccer(
    #     state_def="PVS",
    # ).visualize_data(
    #     model_name="exp_config",
    #     checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
    #     match_id="2022100106",
    #     sequence_id=0,
    # )

    # # split data into train and test (EDMS)
    # rlearn_model_soccer(
    #     state_def="EDMS",
    #     input_path=os.getcwd() + "/tests/data/datastadium/",
    #     output_path=os.getcwd() + "/tests/data/datastadium/split/",
    # ).split_train_test()

    # # preprocess observation (EDMS)
    # rlearn_model_soccer(
    #     state_def="EDMS",
    #     config=os.getcwd() + "/tests/config/preprocessing_dssports2020.json",
    #     input_path=os.getcwd() + "/tests/data/datastadium/split/mini",
    #     output_path=os.getcwd() + "/tests/data/datastadium_simple_obs_action_seq/split/mini",
    #     num_process=5,
    # ).preprocess_observation(batch_size=64)

    # # train model (EDMS)
    # rlearn_model_soccer(state_def="EDMS", config=os.getcwd() + "/tests/config/exp_config.json").train(
    #     exp_name="sarsa_attacker",
    #     run_name="test",
    #     accelerator="gpu",
    #     devices=1,
    #     strategy="ddp",
    # )

    # # visualize data (EDMS)
    # rlearn_model_soccer(
    #     state_def="EDMS",
    # ).visualize_data(
    #     model_name="exp_config",
    #     checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
    #     match_id="2022100106",
    #     sequence_id=0,
    # )

    print("Done")
