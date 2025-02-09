# from /home/c_yeung/workspace6/python/openstarlab/Event/event/sports/soccer/main_class_soccer/main.py
from .soccer.main_class_soccer.main import rlearn_model_soccer

class RLearn_Model:
    def __new__(cls, *args, **kwargs):
        return rlearn_model_soccer(*args, **kwargs)


if __name__ == '__main__':
    import os
    # test split_data
    RLearn_Model.rlearn_model_soccer(
        input_path=os.getcwd()+'/tests/data/datastadium/',
        output_path=os.getcwd()+'/tests/data/datastadium/split/',
    ).train_test_split()

    # test preprocess observation data
    rlearn_model_soccer(
        config=os.getcwd()+'/tests/config/preprocessing_datastadium2020.json',
        input_path=os.getcwd()+'/tests/data/datastadium/split/mini',
        output_path=os.getcwd()+'/tests/data/datastadium_simple_obs_action_seq/split/mini',
        num_process=1,
        batch_size=64,
    ).preprocess_observation()

    # test train model
    rlearn_model_soccer(
        config=os.getcwd()+'/tests/config/exp_config.json',
        exp_name='sarsa_attacker',
        run_name='test',
        accelerator='gpu',
        device=1,
        seed=42,
    ).train()

    # test visualize
    rlearn_model_soccer(
        data_dir=os.path.join(os.path.dirname(__file__), 'data'),
        save_dir=os.path.join(os.path.dirname(__file__), 'save'),
        config_path=os.path.join(os.path.dirname(__file__), 'config.json'),
        league='EPL',
        match_id='1',
        split_ratio=0.8,
        seed=42
    ).visualize_data()
    print('Done')