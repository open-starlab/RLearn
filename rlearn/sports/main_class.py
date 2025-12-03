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
    ).run_rlearn(run_split_train_test=True)

    # test preprocess observation data
    RLearn_Model(
        state_def="PVS",
        config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
        input_path=os.getcwd() + "/test/data/datastadium/split/mini",
        output_path=os.getcwd() + "/test/data/datastadium_simple_obs_action_seq/split/mini",
        num_process=5,
    ).run_rlearn(run_preprocess_observation=True, batch_size=64)

    # test train model
    RLearn_Model(state_def="PVS", config=os.getcwd() + "/test/config/exp_config.json").run_rlearn(
        run_train_and_test=True,
        exp_name="sarsa_attacker",
        run_name="test",
        accelerator="gpu",
        devices=1,
        strategy="ddp",
    )

    # test visualize
    RLearn_Model(
        state_def="PVS",
    ).run_rlearn(
        run_visualize_data=True,
        model_name="exp_config",
        exp_config_path=os.getcwd() + "/test/config/exp_config.json",
        checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
        events_file_path=os.getcwd() + "/test/data/dss/preprocess_data/2022100106/events.jsonl",
        match_id="2022100106",
        sequence_id=0,
    )

    print("Individual tests")
    print("=" * 50)

    # Full pipeline: split -> preprocess -> train -> visualize
    print("Running full pipeline...")
    RLearn_Model(
        state_def="PVS",
        config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
        input_path=os.getcwd() + "/test/data/datastadium/",
        output_path=os.getcwd() + "/test/data/datastadium/split/",
        num_process=5,
    ).run_rlearn(
        run_split_train_test=True,
        run_preprocess_observation=True,
        run_train_and_test=True,
        run_visualize_data=True,
        batch_size=64,
        exp_name="sarsa_attacker",
        run_name="full_pipeline_test",
        accelerator="gpu",
        devices=1,
        strategy="ddp",
        save_q_values_csv=True,
        max_games_csv=1,
        max_sequences_per_game_csv=5,
        model_name="exp_config",
        exp_config_path=os.getcwd() + "/test/config/exp_config.json",
        events_file_path=os.getcwd() + "/test/data/dss/preprocess_data/2022100106/events.jsonl",
        match_id="2022100106",
        sequence_id=0,
    )

    print("Full pipeline completed successfully!")
