# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
from .soccer.main_class_soccer.main import rlearn_model_soccer


class RLearn_Model:
    state_list = ["PVS", "EDMS"]

    def __new__(cls, state_def, *args, **kwargs):
        if state_def in cls.state_list:
            return rlearn_model_soccer(state_def, *args, **kwargs)
        else:
            raise ValueError(f"Invalid state_def '{state_def}'. Supported values are: {', '.join(cls.state_list)}")


if __name__ == "__main__":
    import os

    # test split_data
    RLearn_Model(
        state_def="PVS",
        input_path=os.getcwd() + "/test/data/datastadium/",
        output_path=os.getcwd() + "/test/data/datastadium/split/",
    ).split_train_test()

    # test preprocess observation data
    RLearn_Model(
        state_def="PVS",
        config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
        input_path=os.getcwd() + "/test/data/datastadium/split/mini",
        output_path=os.getcwd() + "/test/data/datastadium_simple_obs_action_seq/split/mini",
        num_process=5,
    ).preprocess_observation(batch_size=64)

    # test train model
    RLearn_Model(state_def="PVS", config=os.getcwd() + "/test/config/exp_config.json").train_and_test(
        exp_name="sarsa_attacker",
        run_name="test",
        accelerator="gpu",
        devices=1,
        strategy="ddp",
    )

    # test visualize
    RLearn_Model(
        state_def="PVS",
    ).visualize_data(
        model_name="exp_config",
        checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
        match_id="2022100106",
        sequence_id=0,
    )

    print("Done")
