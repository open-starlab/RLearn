import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
import torch

from ..class_weight.class_weight import ClassWeightBase
from ..env import OUTPUT_DIR
from ..utils.file_utils import load_json, save_formatted_json
from .run_rlearn_config import PreprocessObservationConfig


def resolve_training_runtime(
    accelerator: str | None,
    devices: Any,
    strategy: str | None,
    exp_config: dict[str, Any],
    test_mode: bool,
) -> tuple[str, int, str]:
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if devices is None:
        devices = 1
    if strategy is None or accelerator == "cpu":
        strategy = "auto"

    if test_mode:
        exp_config["max_epochs"] = 1
        exp_config["datamodule"]["batch_size"] = min(exp_config["datamodule"]["batch_size"], 32)
        accelerator = "cpu"
        strategy = "auto"

    return accelerator, devices, strategy


def prepare_class_weights(
    exp_config: dict[str, Any],
    datamodule: Any,
    use_class_weights: bool | None,
    logger: logging.Logger,
) -> torch.Tensor | None:
    should_use_class_weights = "class_weight_fn" in exp_config if use_class_weights is None else use_class_weights
    if not should_use_class_weights:
        logger.info("Skipping class weights.")
        return None

    if "class_weight_fn" not in exp_config:
        raise ValueError("use_class_weights=True requires class_weight_fn in the experiment config.")

    logger.info("Prepare class weights...")
    start = time.time()
    class_weight_fn = ClassWeightBase.from_params(exp_config["class_weight_fn"])
    if class_weight_fn.class_weights is not None:
        class_weights = class_weight_fn.class_weights
    else:
        logger.info("Calculating class weights...")
        class_counts = torch.zeros(datamodule.state_action_tokenizer.num_tokens)
        class_weight_batch_size = exp_config.get("class_weight_batch_size", 512)
        for batch in tqdm(
            datamodule.train_dataloader(batch_size=class_weight_batch_size),
            desc="calculating class weights",
        ):
            valid_actions = torch.masked_select(batch["action"], batch["mask"].bool())
            class_counts += torch.bincount(valid_actions, minlength=datamodule.state_action_tokenizer.num_tokens)
        class_weights = class_weight_fn.calculate(class_counts=class_counts)

    assert class_weights.shape[0] == datamodule.state_action_tokenizer.num_tokens, (
        f"Class weights shape mismatch: {class_weights.shape[0]} != {datamodule.state_action_tokenizer.num_tokens}"
    )
    logger.info(f"Prepare class weights is done. {time.time() - start} sec")
    return class_weights


def resolve_resume_checkpoint_path(
    resume_checkpoint_path: str | None,
    logger: logging.Logger,
) -> str | None:
    if resume_checkpoint_path is None:
        return None

    checkpoint_path = Path(resume_checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_output_base_dir(output_base_dir: str | None) -> Path:
    if output_base_dir is None:
        return OUTPUT_DIR
    return Path(output_base_dir).expanduser().resolve()


def resolve_preprocess_datasets(test_mode: bool) -> list[str]:
    if test_mode:
        return ["mini"]
    return ["train", "validation", "test", "mini"]


def preprocess_split_datasets(
    model: Any,
    original_output_path: str,
    preprocess_config: PreprocessObservationConfig,
    logger: logging.Logger,
) -> str:
    datasets_to_process = resolve_preprocess_datasets(preprocess_config.test_mode)
    base_output_dir = Path(original_output_path).parent / f"{Path(original_output_path).name}_simple_obs_action_seq"

    for dataset_name in datasets_to_process:
        model.input_path = str(Path(original_output_path) / dataset_name)
        model.output_path = str(base_output_dir / dataset_name)

        if Path(model.input_path).exists():
            logger.info(f"Processing {dataset_name} dataset...")
            model.preprocess_observation(batch_size=preprocess_config.batch_size)
        else:
            logger.warning(f"Dataset {dataset_name} not found at {model.input_path}, skipping...")

    return str(base_output_dir)


def build_training_config_for_preprocessed_data(
    exp_config_path: str,
    preprocessed_output_base: str,
    test_mode: bool,
) -> str:
    config_data = load_json(exp_config_path)
    if test_mode:
        mini_path = str(Path(preprocessed_output_base) / "mini")
        config_data["dataset"]["train_filename"] = mini_path
        config_data["dataset"]["valid_filename"] = mini_path
        config_data["dataset"]["test_filename"] = mini_path
    else:
        config_data["dataset"]["train_filename"] = str(Path(preprocessed_output_base) / "train")
        config_data["dataset"]["valid_filename"] = str(Path(preprocessed_output_base) / "validation")
        config_data["dataset"]["test_filename"] = str(Path(preprocessed_output_base) / "test")

    updated_config_path = Path(preprocessed_output_base) / "updated_exp_config.json"
    updated_config_path.parent.mkdir(parents=True, exist_ok=True)
    save_formatted_json(config_data, updated_config_path)
    return str(updated_config_path)


def resolve_latest_checkpoint_path(
    exp_name: str | None,
    run_name: str | None,
    output_base_dir: str | None,
    logger: logging.Logger,
) -> str | None:
    if exp_name is None or run_name is None:
        return None

    checkpoint_dir = resolve_output_base_dir(output_base_dir) / exp_name / run_name / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        return None

    latest_checkpoint_path = str(max(checkpoint_files, key=lambda checkpoint_file: checkpoint_file.stat().st_mtime))
    logger.info(f"Auto-detected checkpoint: {latest_checkpoint_path}")
    return latest_checkpoint_path
