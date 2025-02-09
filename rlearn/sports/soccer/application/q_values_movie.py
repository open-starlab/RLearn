import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotsoccer as mps
import os
from pathlib import Path
import japanize_matplotlib
from io import StringIO
import cv2
import ast
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
import glob
from ast import literal_eval
from math import pi
from matplotlib.colors import Normalize
import matplotlib.cm as cm

onball_action_names = [
    'pass',
    'through_pass',
    'shot',
    'cross',
    'dribble',
    'defense'
]

offball_action_names = [
    'idle',
    'up',
    'up_right',
    'right',
    'down_right',
    'down',
    'down_left',
    'left',
    'up_left'
]

offball_action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
onball_action_idx = [9, 10, 11, 12, 13, 14]


def preprocess_tracking_data(df, sequence_id):
    df_sequence = df[df['sequence_id'] == sequence_id].copy()
    df_sequence.loc[:, 'raw_state'] = df_sequence['events'].apply(lambda x: x['state'])

    df_sequence = df_sequence.drop(columns=['events'])
    df_sequence.reset_index(drop=True, inplace=True)

    df_sequence['ball_x'] = df_sequence['raw_state'].apply(lambda x: x['ball']['position']['x'])
    df_sequence['ball_y'] = df_sequence['raw_state'].apply(lambda x: x['ball']['position']['y'])


    df_sequence['attack_team_position'] = df_sequence['raw_state'].apply(lambda x: [player['position'] for player in x['attack_players']])
    df_sequence['attack_team_player'] = df_sequence['raw_state'].apply(lambda x: [player['player_name'] for player in x['attack_players']])
    df_sequence['attack_team_action'] = df_sequence['raw_state'].apply(lambda x: [player['action'] for player in x['attack_players']])
    df_sequence['defence_team_position'] = df_sequence['raw_state'].apply(lambda x: [player['position'] for player in x['defense_players']])
    df_sequence['defence_team_player'] = df_sequence['raw_state'].apply(lambda x: [player['player_name'] for player in x['defense_players']])

    return df_sequence


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        val = val.replace('nan', 'math.nan')
        val = eval(val)
        return val


def preprocess_q_values(q_values):
    q_values['q_value'] = q_values['q_value'].apply(lambda x: x[7:-1])
    q_values['q_value'] = q_values['q_value'].apply(lambda x: np.array(safe_literal_eval(x.replace('\n', ''))) if isinstance(x, str) else x)
    return q_values


def normalize_list(values):
    scaler = MinMaxScaler()
    values = np.array(values).reshape(-1, 1)
    return scaler.fit_transform(values).flatten()


def split_q_values(values, offball_action_idx, onball_action_idx): 
    offball_values = values[offball_action_idx] 
    onball_values = values[onball_action_idx] 
    return offball_values, onball_values


def offside(attackers, defenders):
    attackers_x = [attacker['x'] for attacker in attackers]
    offside_attackers = [0]*len(attackers)
    defenders_min_x = min([defender['x'] for defender in defenders])
    offside_line = 52.5 if defenders_min_x < 52.5 else defenders_min_x

    for i, attacker_x in enumerate(attackers_x):
        if attacker_x > offside_line:
            offside_attackers[i] = 1

    return offside_attackers


def plot_q_values(df_sequence, q_values, name, team_name, match_id, sequence_id):
    offball_directions = ["U", "U-L", "L", "B-L", "B", "B-R", "R", "U-R"]
    onball_actions = onball_action_names
    offball_q_values = 0
    angles_offball = np.linspace(0, 2 * np.pi, len(offball_directions) + 1)
    angles_onball = np.linspace(0, 2 * np.pi, len(onball_actions) + 1)

    cmap = cm.viridis  # colormap (e.g., viridis)
    yticks_offball, yticks_onball = [], []
    for idx, (data, q_value) in enumerate(zip(df_sequence.iterrows(), q_values.iterrows())):
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1], polar=True)
        mps.field(ax=ax1, show=False, color="green")
        ball_x, ball_y = data[1]['ball_x'] + 52.5, data[1]['ball_y'] + 34

        if ball_x - 25 < -2:
            field_x_min = -2
            field_x_max = 55
        elif ball_x + 25 > 107:
            field_x_min = 45
            field_x_max = 80
        else:
            field_x_min = ball_x-25
            field_x_max = ball_x+25
        ax1.plot(ball_x, ball_y, 'ko', markersize=7)
        ax1.set_xlim(field_x_min, field_x_max)
        ax1.set_ylim(-2, 70)
        ax1.set_title(f'{team_name}_{sequence_id}_{name}_{idx:04d}', fontsize=15)
        
        for position, player in zip(data[1]['attack_team_position'], data[1]['attack_team_player']):
            ax1.plot(position['x'] + 52.5, position['y'] + 34, 'ro', markersize=7)
            if player == name and field_x_min <= position['x'] + 52.5 <= field_x_max:
                ax1.text(position['x'] + 52, position['y'] + 35, player, fontsize=15)
        for position, player in zip(data[1]['defence_team_position'], data[1]['defence_team_player']):
            ax1.plot(position['x'] + 52.5, position['y'] + 34, 'bo', markersize=7)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'{data[1]["team_name_attack"]} vs {data[1]["team_name_defense"]}', fontsize=15)

        idle_q_values = q_value[1]['q_value_offball'][0].copy()
        offball_q_values = q_value[1]['q_value_offball'][1:].copy()

        offball_q_values = np.nan_to_num(offball_q_values, nan=0.0, posinf=0.0, neginf=0.0)
        idle_q_values = np.nan_to_num(idle_q_values, nan=0.0, posinf=0.0, neginf=0.0)

        ax2.set_xticks(angles_offball[:-1])
        ax2.set_xticklabels(offball_directions, color='grey', fontsize=15)
        yticks_offball = np.round(np.arange(-1, 1.2, 0.5), 2)
        ax2.set_yticks(yticks_offball)
        ax2.set_yticklabels([f'{ytick:.2f}' for ytick in yticks_offball], color='grey', fontsize=8)
        ax2.set_ylim(-1, max(offball_q_values)+0.3)        
        # ax2.set_theta_offset(np.pi / 2)
        ax2.set_title(f'Q Values for Offball Actions ({name})', fontsize=15)
        
        norm = Normalize(vmin=min(offball_q_values), vmax=max(offball_q_values))
        colors = cmap(norm(offball_q_values))

        for i in range(len(offball_q_values)):
            ax2.bar(
                angles_offball[i], 
                offball_q_values[i], 
                width=(angles_offball[i + 1]-0.2 - angles_offball[i]),
                color=colors[i],
                edgecolor='black',
                alpha=0.7,
            )

            ax2.text(
                angles_offball[i], 
                max(offball_q_values), 
                f'{offball_q_values[i]:.3g}'
            )

        scatter_size = max(idle_q_values * 1000, 0)
        ax2.scatter(0, -1, s=scatter_size, c='blue', alpha=0.7, label=f'{idle_q_values}')
        ax2.text(0, -1, f"{idle_q_values:.3g}", ha='center', va='center', color='white', fontsize=8)
        


        # onball_q_values = q_value[1]['q_value_onball'].copy()
        # norm = Normalize(vmin=min(onball_q_values), vmax=max(onball_q_values))
        # colors = cmap(norm(onball_q_values))

        # for i in range(len(onball_q_values)):
        #     ax3.bar(
        #         angles_onball[i],
        #         onball_q_values[i], 
        #         width=(angles_onball[i + 1]-0.2 - angles_onball[i]), 
        #         color=colors[i],
        #         edgecolor='black',
        #         alpha=0.7,
        #     )

        #     ax3.text(
        #         angles_onball[i],
        #         max(onball_q_values),
        #         f'{onball_q_values[i]:.3g}'
        #     )

        # ax3.set_xticks(angles_onball[:-1]) 
        # ax3.set_xticklabels(onball_actions, color='black', fontsize=10)
        # yticks_onball = np.round(np.arange(-1, 1.2, 0.5), 2)
        # # print(yticks_onball)
        # ax3.set_yticks(yticks_onball)
        # ax3.set_yticklabels([f'{ytick:.2f}' for ytick in yticks_onball], color='grey', fontsize=8)
        # ax3.set_ylim(-1, max(onball_q_values)+0.3)
        # ax3.set_title(f'Q Values for Onball Actions ({name})', fontsize=12)
        # ax3.set_theta_offset(np.pi/2)

        output_path = os.getcwd()+f'/tests/data/figures/{match_id}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.tight_layout()
        plt.show()
        plt.savefig(output_path+f'frame_{name}_{sequence_id}_{idx:04d}.png')
        plt.close(fig)

        if idx == 10:
            break


def movie_from_images(image_files, output_file):
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 1, (width, height))

    for image in image_files:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for image in image_files:
        os.remove(image)


def create_movie(q_values_path, match_id, sequence_id):
    target_file_path = q_values_path
    tracking_data_path = os.getcwd()+f'/tests/data/datastadium/{match_id}/events.jsonl' # tracking data path preprocessed in SAR Package

    df = pd.DataFrame()
    data_list = []
    with open(tracking_data_path, 'r') as file:
        for line in file:
            data_list.append(pd.read_json(StringIO(line)))

    df = pd.concat(data_list, axis=0)

    df_sequence = preprocess_tracking_data(df, sequence_id)

    q_values = pd.read_csv(target_file_path)

    q_values = preprocess_q_values(q_values)

    # create new columns for offball and onball q_values
    q_values[['q_value_offball', 'q_value_onball']] = pd.DataFrame(q_values['q_value'].apply(lambda x: split_q_values(x, offball_action_idx, onball_action_idx)).tolist(), index=q_values.index)

    # standardize q_values
    q_values['q_value_offball'] = q_values['q_value_offball'].apply(normalize_list)
    q_values['q_value_onball'] = q_values['q_value_onball'].apply(normalize_list)

    player_name = q_values['player_name'].unique()
    print(player_name)
    for i in range(len(player_name)):
        name = player_name[i]
        team_name = q_values['team_name'][q_values['player_name'] == name].values[0]

        print(f"player name: {name}")
        plot_q_values(df_sequence, q_values, name, team_name, match_id, sequence_id)
        image_files = sorted(glob.glob(os.getcwd()+f'/tests/data/figures/{match_id}/frame_{name}_{sequence_id}*.png'))
        output_dir = os.getcwd()+f'/tests/data/movies/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{match_id}_{sequence_id}_{name}.avi')
        # movie_from_images(image_files, output_file)
        break
