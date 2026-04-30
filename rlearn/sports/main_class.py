try:
    from .soccer.main_class_soccer.main import rlearn_model_soccer
    from .soccer.main_class_soccer.run_rlearn_config import (
        PreprocessObservationConfig,
        SplitTrainTestConfig,
        TrainAndTestConfig,
        VisualizeDataConfig,
    )
except ImportError:
    from rlearn.sports.soccer.main_class_soccer.main import rlearn_model_soccer


class RLearn_Model:
    state_list = ["PVS", "EDMS"]

    def __new__(cls, state_def, *args, **kwargs):
        if state_def in cls.state_list:
            return rlearn_model_soccer(state_def, *args, **kwargs)
        else:
            raise ValueError(f"Invalid state_def '{state_def}'. Supported values are: {', '.join(cls.state_list)}")


if __name__ == "__main__":
    pass

    # test split_data
    # RLearn_Model(
    #     state_def="PVS",
    #     input_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/",
    #     output_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/split/",
    # ).run_rlearn(
    #     run_split_train_test=True,
    #     split_config=SplitTrainTestConfig(),
    # )

    # # test preprocess observation data
    # RLearn_Model(
    #     state_def="PVS",
    #     config=os.getcwd() + "/test/config/preprocessing_datafactory.json",
    #     input_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/split/mini",
    #     output_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory_simple_obs_action_seq/split/mini",
    #     num_process=5,
    # ).run_rlearn(
    #     run_preprocess_observation=True,
    #     preprocess_config=PreprocessObservationConfig(batch_size=64),
    # )

    # # test train model
    # RLearn_Model(state_def="PVS", config=os.getcwd() + "/test/config/exp_config.json").run_rlearn(
    #     run_train_and_test=True,
    #     train_and_test_config=TrainAndTestConfig(
    #         exp_name="sarsa_attacker",
    #         run_name="test",
    #         exp_config_path=os.getcwd() + "/test/config/exp_config.json",
    #         accelerator="gpu",
    #         devices=1,
    #         strategy="ddp",
    #     ),
    # )

    # # test visualize
    # RLearn_Model(
    #     state_def="PVS",
    # ).run_rlearn(
    #     run_visualize_data=True,
    #     visualize_config=VisualizeDataConfig(
    #         model_name="exp_config",
    #         exp_config_path=os.getcwd() + "/test/config/exp_config.json",
    #         checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
    #         tracking_file_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/0001/events.jsonl",
    #         match_id="0001",
    #         sequence_id=0,
    #     ),
    # )

    # print("Individual tests")
    # print("=" * 50)

    # # Full pipeline: split -> preprocess -> train -> visualize
    # print("Running full pipeline...")
    # RLearn_Model(
    #     state_def="PVS",
    #     config=os.getcwd() + "/test/config/preprocessing_datafactory.json",
    #     input_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/",
    #     output_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/split_pipeline/",
    #     num_process=5,
    # ).run_rlearn(
    #     run_split_train_test=True,
    #     run_preprocess_observation=True,
    #     run_train_and_test=True,
    #     run_visualize_data=True,
    #     split_config=SplitTrainTestConfig(),
    #     preprocess_config=PreprocessObservationConfig(batch_size=64),
    #     train_and_test_config=TrainAndTestConfig(
    #         exp_name="sarsa_attacker",
    #         run_name="full_pipeline_test",
    #         exp_config_path=os.getcwd() + "/test/config/exp_config_datafactory.json",
    #         accelerator="cpu",
    #         devices=1,
    #         strategy=None,
    #         save_q_values_csv=True,
    #         max_games_csv=1,
    #         max_sequences_per_game_csv=5,
    #     ),
    #     visualize_config=VisualizeDataConfig(
    #         model_name="exp_config",
    #         tracking_file_path=os.getcwd() + "/test/sports/rlearn_data/data/datafactory/preprocess_data/0001/events.jsonl",
    #         match_id="0001",
    #         sequence_id=0,
    #     ),
    # )

    # print("Full pipeline completed successfully!")
