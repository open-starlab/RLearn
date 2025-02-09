# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
from .soccer.main_class_soccer.main import rlearn_model_soccer

class RLearn_Model:
    def __new__(cls, *args, **kwargs):
        return rlearn_model_soccer(*args, **kwargs)


if __name__ == '__main__':
    import os
    # test split_data
    RLearn_Model(
        input_path=os.getcwd()+'/tests/data/datastadium/',
        output_path=os.getcwd()+'/tests/data/datastadium/split/'
    ).train_test_split()

    # test preprocess observation data
    RLearn_Model(
        config=os.getcwd()+'/tests/config/preprocessing_dssports2020.json',
        input_path=os.getcwd()+'/tests/data/datastadium/split/mini',
        output_path=os.getcwd()+'/tests/data/datastadium_simple_obs_action_seq/split/mini',
        num_process=5,
    ).preprocess_observation(batch_size=64)

    # test train model
    RLearn_Model(
        config=os.getcwd()+'/tests/config/exp_config.json'
    ).train(
        exp_name='sarsa_attacker',
        run_name='test',
        accelerator="gpu",
        devices=1,
        strategy="ddp",
    )

    # test visualize
    RLearn_Model().visualize_data(
        model_name='exp_config',
        checkpoint_path=os.getcwd()+'/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt',
        match_id='2022100106',
        sequence_id=0,
    )

    print('Done')
