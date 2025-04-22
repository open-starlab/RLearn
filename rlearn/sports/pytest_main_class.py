# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
import os
from .soccer.main_class_soccer.main import rlearn_model_soccer

class RLearn_Model:
    def __new__(cls, *args, **kwargs):
        return rlearn_model_soccer(*args, **kwargs)


# def test_datastadium_split_mini_data():
#     # test split_data
#     RLearn_Model(
#         input_path=os.getcwd()+'/test/data/dss/preprocess_data/',
#         output_path=os.getcwd()+'/test/data/dss/preprocess_data/split/'
#     ).split_train_test(pytest=True)

# def test_datastadium_preprocess_data():
#     # test preprocess observation data
#     RLearn_Model(
#         config=os.getcwd()+'/test/config/preprocessing_dssports2020.json',
#         input_path=os.getcwd()+'/test/data/dss/preprocess_data/split/mini',
#         output_path=os.getcwd()+'/test/data/dss_simple_obs_action_seq/split/mini',
#         num_process=5,
#     ).preprocess_observation(batch_size=64)

def test_datastadium_train_data():
    # test train model
    RLearn_Model(
        config=os.getcwd()+'/test/config/exp_config.json'
    ).train(
        exp_name='sarsa_attacker',
        run_name='test',
        accelerator="cpu",
        devices=1,
        strategy='ddp'
    )
    

# def test_datastadium_visualize_data():
#     # test visualize
#     RLearn_Model().visualize_data(
#         model_name='exp_config',
#         checkpoint_path=os.getcwd()+'/rlearn/sports/output/sarsa_attacker/test/checkpoints/epoch=1-step=2.ckpt',
#         match_id='0001',
#         sequence_id=0,
#     )

# if __name__ == '__main__':
#     test_datastadium_split_mini_data()
#     test_datastadium_preprocess_data()
#     test_datastadium_train_data()
#     test_datastadium_visualize_data()
#     print('Done')