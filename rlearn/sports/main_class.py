# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
from .soccer.main_class_soccer.main import rlearn_model_soccer

class RLearn_Model:
    state_list = ["PVS", "EDMS", "PVSS"]

    def __new__(cls, state_def, *args, **kwargs):
        if state_def in cls.state_list:
            return rlearn_model_soccer(state_def, *args, **kwargs)
        else:
            raise ValueError(
                f"Invalid state_def '{state_def}'. Supported values are: {', '.join(cls.state_list)}"
            )


if __name__ == "__main__":
    import os

    base_path = os.getcwd()

    # test split_data
    RLearn_Model(
        state_def="PVSS",
        input_path=os.path.join(base_path, "test/data/datastadium/"),
        output_path=os.path.join(base_path, "test/data/datastadium/split/"),
    ).split_train_test()

    # test preprocess observation data
    RLearn_Model(
        state_def="PVSS",
        config=os.path.join(base_path, "test/config/preprocessing_pvss.json"),
        input_path=os.path.join(base_path, "test/data/datastadium/split/mini"),
        output_path=os.path.join(base_path, "test/data/datastadium_pvss_obs/split/mini"),
        num_process=5,
    ).preprocess_observation(batch_size=64)

    # test train model
    RLearn_Model(
        state_def="PVSS",
        config=os.path.join(base_path, "test/config/exp_config_qmix.json"),
    ).train(
        exp_name="qmix_pvss",
        run_name="test",
        accelerator="gpu",
        devices=1,
        strategy="ddp",
    )

    # test visualize
    RLearn_Model(
        state_def="PVSS",
    ).visualize_data(
        model_name="exp_config_qmix",
        checkpoint_path=os.path.join(base_path, "rlearn/sports/output/qmix_pvss/test/checkpoints/epoch=1-step=2.ckpt"),
        match_id="2022100106",
        sequence_id=0,
    )

    print("Done")
