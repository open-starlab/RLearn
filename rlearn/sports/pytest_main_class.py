from .soccer.main_class_soccer.main import rlearn_model_soccer


class RLearn_Model:
    state_def_list = ["PVS", "EDMS"]

    def __new__(cls, state_def, *args, **kwargs):
        if state_def in cls.state_def_list:
            return rlearn_model_soccer(state_def, *args, **kwargs)
        else:
            raise ValueError(f"Invalid state_def '{state_def}'. Supported values are: {', '.join(cls.state_def_list)}")


# def test_PVS_split_only():
#     # test split data only
#     RLearn_Model(
#         state_def="PVS",
#         input_path="test/data/dss/preprocess_data/",
#         output_path="test/data/dss/preprocess_data/split/",
#     ).run_rlearn(run_split_train_test=True, test_mode=True)


# def test_PVS_preprocess_only():
#     # test preprocess observation data only
#     RLearn_Model(
#         state_def="PVS",
#         config="test/config/preprocessing_dssports2020.json",
#         input_path="test/data/dss/preprocess_data/split/mini",
#         output_path="test/data/dss_simple_obs_action_seq/split/mini",
#         num_process=3,
#     ).run_rlearn(run_preprocess_observation=True, batch_size=64, test_mode=True)


# def test_PVS_train_only():
#     # test train model only
#     RLearn_Model(state_def="PVS", config="test/config/exp_config.json").run_rlearn(
#         run_train_and_test=True,
#         exp_name="sarsa_attacker",
#         run_name="test_train_only",
#         exp_config_path="test/config/exp_config.json",
#         test_mode=True,
#     )


def test_PVS_full_pipeline():
    # test full pipeline: split -> preprocess -> train -> visualize

    RLearn_Model(
        state_def="PVS",
        config="test/config/preprocessing_dssports2020.json",
        input_path="test/data/dss/preprocess_data/",
        output_path="test/data/dss/preprocess_data/split_pipeline/",
        num_process=3,
    ).run_rlearn(
        run_split_train_test=True,
        run_preprocess_observation=True,
        run_train_and_test=True,
        run_visualize_data=True,
        batch_size=32,
        exp_name="sarsa_attacker",
        run_name="full_pipeline_test",
        exp_config_path="test/config/exp_config.json",
        model_name="exp_config",
        tracking_file_path="test/data/dss/preprocess_data/2022100106/events.jsonl",
        match_id="2022100106",
        sequence_id=0,
        test_mode=True,
    )


# def test_EDMS_split_mini_data():
#     # test split_data
#     RLearn_Model(
#         state_def="EDMS",
#         input_path=os.getcwd() + "/test/data/dss/preprocess_data/",
#         output_path=os.getcwd() + "/test/data/dss/preprocess_data/split/",
#     ).split_train_test(test_mode=True)


# def test_EDMS_preprocess_data():
#     # test preprocess observation data
#     RLearn_Model(
#         state_def="EDMS",
#         config=os.getcwd() + "/test/config/preprocessing_dssports2020.json",
#         input_path=os.getcwd() + "/test/data/dss/preprocess_data/split/mini",
#         output_path=os.getcwd() + "/test/data/dss_simple_obs_action_seq/split/mini",
#         num_process=5,
#     ).preprocess_observation(batch_size=64)


# def test_EDMS_train_data():
#     # test train model
#     RLearn_Model(state_def="EDMS", config=os.getcwd() + "/test/config/exp_config.json").train(
#         exp_name="sarsa_attacker", run_name="test", accelerator="cpu", devices=1, strategy="ddp", mlflow=False
#     )


# def test_EDMS_visualize_data():
#     # test visualize
#     RLearn_Model(state_def="EDMS").visualize_data(
#         model_name="exp_config",
#         checkpoint_path=os.getcwd() + "/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt",
#         match_id="2022100106",
#         sequence_id=0,
#     )


if __name__ == "__main__":
    # Individual step tests
    # test_PVS_split_only()
    # test_PVS_preprocess_only()
    # test_PVS_train_only()

    # Full pipeline test
    test_PVS_full_pipeline()

    # test_EDMS_split_mini_data()
    # test_EDMS_preprocess_data()
    # test_EDMS_train_data()
    # test_EDMS_visualize_data()
    print("All tests completed successfully!")
