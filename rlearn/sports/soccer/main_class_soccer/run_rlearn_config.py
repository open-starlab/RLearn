from dataclasses import dataclass
from typing import Any


@dataclass
class SplitTrainTestConfig:
    test_mode: bool = False


@dataclass
class PreprocessObservationConfig:
    batch_size: int = 64
    test_mode: bool = False


@dataclass
class TrainAndTestConfig:
    exp_name: str | None = None
    run_name: str | None = None
    exp_config_path: str | None = None
    accelerator: str | None = None
    devices: Any = None
    strategy: str | None = None
    save_q_values_csv: bool = False
    max_games_csv: int = 1
    max_sequences_per_game_csv: int = 5
    save_intermediate_checkpoints: bool = False
    checkpoint_every_n_epochs: int = 1
    checkpoint_save_top_k: int = -1
    test_mode: bool = False
    use_class_weights: bool | None = None
    resume_checkpoint_path: str | None = None
    output_base_dir: str | None = None
    class_weight_only: bool = False


@dataclass
class VisualizeDataConfig:
    model_name: str | None = None
    exp_config_path: str | None = None
    checkpoint_path: str | None = None
    tracking_file_path: str | None = None
    match_id: str | None = None
    sequence_id: int | None = None
    test_mode: bool = False
    viz_style: str = "radar"
    movie_output_dir: str | None = None
    keep_frames: bool = True
