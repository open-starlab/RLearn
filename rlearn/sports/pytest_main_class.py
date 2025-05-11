# === File: openstarlab/main.py ===
from pathlib import Path
from openstarlab.Event.event.sports.soccer.main_class_soccer.main import \
    rlearn_model_soccer as RLearn_Model

# PROJECT_ROOT points to the root of your project (where manage.py or setup.py lives)
PROJECT_ROOT = Path(__file__).resolve().parent


# === File: openstarlab/cli.py ===
import argparse
from pathlib import Path
from openstarlab.main import RLearn_Model, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="RLearn Soccer pipeline entrypoint"
    )
    parser.add_argument(
        "--stage",
        choices=["split", "preprocess", "train", "all"],
        default="all",
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for preprocessing"
    )
    parser.add_argument(
        "--num_process", type=int, default=5,
        help="Number of worker processes for preprocessing"
    )
    args = parser.parse_args()

    root = PROJECT_ROOT
    paths = {
        "split_in":  root / "test" / "data" / "dss" / "preprocess_data",
        "split_out": root / "test" / "data" / "dss" / "preprocess_data" / "split",
        "mini_in":   root / "test" / "data" / "dss" / "preprocess_data" / "split" / "mini",
        "mini_out":  root / "test" / "data" / "dss_simple_obs_action_seq" / "split" / "mini",
    }

    if args.stage in ("split", "all"):
        RLearn_Model(
            input_path=str(paths["split_in"]),
            output_path=str(paths["split_out"]),
        ).split_train_test()

    if args.stage in ("preprocess", "all"):
        RLearn_Model(
            config=str(root / "test" / "config" / "preprocessing_dssports2020.json"),
            input_path=str(paths["mini_in"]),
            output_path=str(paths["mini_out"]),
            num_process=args.num_process,
        ).preprocess_observation(batch_size=args.batch_size)

    if args.stage in ("train", "all"):
        RLearn_Model(
            config=str(root / "test" / "config" / "exp_config.json")
        ).train(
            exp_name='sarsa_attacker',
            run_name='cli',
            accelerator="cpu",
            devices=1,
            strategy='ddp',
            mlflow=False
        )


if __name__ == "__main__":
    main()


# === File: tests/test_rlearn_soccer.py ===
"""
Pytest suite for RLearn Soccer.

- Use `pytest -m "not slow"` to skip the training test by default.
- To parallelize across CPUs, install pytest-xdist and run:
    pytest -n auto -m "not slow"
"""
import pytest
from pathlib import Path
from openstarlab.main import RLearn_Model, PROJECT_ROOT


@pytest.fixture(scope="session")
def paths():
    root = PROJECT_ROOT
    return {
        "split_in":  root / "test" / "data" / "dss" / "preprocess_data",
        "split_out": root / "test" / "data" / "dss" / "preprocess_data" / "split",
        "mini_in":   root / "test" / "data" / "dss" / "preprocess_data" / "split" / "mini",
        "mini_out":  root / "test" / "data" / "dss_simple_obs_action_seq" / "split" / "mini",
        "cfg_pp":    root / "test" / "config" / "preprocessing_dssports2020.json",
        "cfg_tr":    root / "test" / "config" / "exp_config.json",
    }


def test_split_train_test(paths):
    model = RLearn_Model(
        input_path=str(paths["split_in"]),
        output_path=str(paths["split_out"]),
    )
    model.split_train_test(pytest=True)


@pytest.mark.dependency(depends=["test_split_train_test"])
def test_preprocess_observation(paths):
    model = RLearn_Model(
        config=str(paths["cfg_pp"]),
        input_path=str(paths["mini_in"]),
        output_path=str(paths["mini_out"]),
        num_process=paths.get("num_process", 5),
    )
    model.preprocess_observation(batch_size=64)


@pytest.mark.dependency(depends=["test_preprocess_observation"])
@pytest.mark.slow
def test_train_model(paths):
    model = RLearn_Model(config=str(paths["cfg_tr"]))
    model.train(
        exp_name='sarsa_attacker',
        run_name='test',
        accelerator="cpu",
        devices=1,
        strategy='ddp',
        mlflow=False
    )
