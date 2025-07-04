# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
import os
from .soccer.main_class_soccer.main import rlearn_model_soccer


class RLearn_Model:
    state_def_list = ["PVS", "EDMS"]

    def __new__(cls, state_def, *args, **kwargs):
        if state_def in cls.state_def_list:
            return rlearn_model_soccer(state_def, *args, **kwargs)
        else:
            raise ValueError(f"Invalid state_def '{state_def}'. Supported values are: {', '.join(cls.state_def_list)}")


def test_PVS_split_mini_data():
    # test split_data
    RLearn_Model(
        state_def="PVS",
        input_path=os.getcwd() + "/test/data/dss/preprocess_data/",
        output_path=os.getcwd() + "/test/data/dss/preprocess_data/split/",
    ).split_train_test(pytest=True)


def test_PVS_preprocess_data():
    # test preprocess observation data
    RLearn_Model(
        state_def="PVS",
        config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
        input_path=os.getcwd() + "/test/data/dss/preprocess_data/split/mini",
        output_path=os.getcwd() + "/test/data/dss_simple_obs_action_seq/split/mini",
        num_process=5,
    ).preprocess_observation(batch_size=64)


def test_PVS_train_data():
    # test train model
    RLearn_Model(state_def="PVS", config=os.getcwd() + "/test/config/exp_config.json").train(
        exp_name="sarsa_attacker", run_name="test", accelerator="cpu", devices=1, strategy="ddp", mlflow=False
    )


def test_PVS_visualize_data():
    # test visualize
    RLearn_Model(state_def="PVS").visualize_data(
        model_name="exp_config",
        checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
        match_id="0001",
        sequence_id=0,
    )


def test_EDMS_split_mini_data():
    # test split_data
    RLearn_Model(
        state_def="EDMS",
        input_path=os.getcwd() + "/test/data/dss/preprocess_data/",
        output_path=os.getcwd() + "/test/data/dss/preprocess_data/split/",
    ).split_train_test(pytest=True)


def test_EDMS_preprocess_data():
    # test preprocess observation data
    RLearn_Model(
        state_def="EDMS",
        config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
        input_path=os.getcwd() + "/test/data/dss/preprocess_data/split/mini",
        output_path=os.getcwd() + "/test/data/dss_simple_obs_action_seq/split/mini",
        num_process=5,
    ).preprocess_observation(batch_size=64)


def test_EDMS_train_data():
    # test train model
    RLearn_Model(state_def="EDMS", config=os.getcwd() + "/test/config/exp_config.json").train(
        exp_name="sarsa_attacker", run_name="test", accelerator="cpu", devices=1, strategy="ddp", mlflow=False
    )


def test_EDMS_visualize_data():
    # test visualize
    RLearn_Model(state_def="EDMS").visualize_data(
        model_name="exp_config",
        checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
        match_id="0001",
        sequence_id=0,
    )


if __name__ == "__main__":
    test_PVS_split_mini_data()
    test_PVS_preprocess_data()
    test_PVS_train_data()
    test_PVS_visualize_data()
    test_EDMS_split_mini_data()
    test_EDMS_preprocess_data()
    test_EDMS_train_data()
    test_EDMS_visualize_data()
    print("Done")
