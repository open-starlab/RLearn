import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotsoccer as mps
import os
from io import StringIO
import cv2
import ast
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
import glob

onball_action_names = ["pass", "through_pass", "shot", "cross", "dribble", "defense"]

offball_action_names = ["idle", "up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"]

offball_action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
onball_action_idx = [9, 10, 11, 12, 13, 14]


def preprocess_tracking_data(df, sequence_id, state_def="PVS"):
    df_sequence = df[df["sequence_id"] == sequence_id].copy()
    if state_def == "PVS":
        df_sequence.loc[:, "raw_state"] = df_sequence["events"].apply(lambda x: x["state"])

        df_sequence = df_sequence.drop(columns=["events"])
        df_sequence.reset_index(drop=True, inplace=True)

        df_sequence["ball_x"] = df_sequence["raw_state"].apply(lambda x: x["ball"]["position"]["x"])
        df_sequence["ball_y"] = df_sequence["raw_state"].apply(lambda x: x["ball"]["position"]["y"])

        df_sequence["attack_team_position"] = df_sequence["raw_state"].apply(
            lambda x: [player["position"] for player in x["attack_players"]]
        )
        df_sequence["attack_team_player"] = df_sequence["raw_state"].apply(
            lambda x: [player["player_name"] for player in x["attack_players"]]
        )
        df_sequence["attack_team_action"] = df_sequence["raw_state"].apply(
            lambda x: [player["action"] for player in x["attack_players"]]
        )
        df_sequence["defence_team_position"] = df_sequence["raw_state"].apply(
            lambda x: [player["position"] for player in x["defense_players"]]
        )
        df_sequence["defence_team_player"] = df_sequence["raw_state"].apply(
            lambda x: [player["player_name"] for player in x["defense_players"]]
        )
    elif state_def == "EDMS":
        df_sequence["ball_x"] = df_sequence["raw_state"].apply(lambda x: x["ball"]["position"]["x"])
        df_sequence["ball_y"] = df_sequence["raw_state"].apply(lambda x: x["ball"]["position"]["y"])

        df_sequence["attack_team_position"] = df_sequence["raw_state"].apply(
            lambda x: [player["position"] for player in x["attack_players"]]
        )
        df_sequence["attack_team_player"] = df_sequence["raw_state"].apply(
            lambda x: [player["player_name"] for player in x["attack_players"]]
        )
        df_sequence["attack_team_action"] = df_sequence["raw_state"].apply(
            lambda x: [player["action"] for player in x["attack_players"]]
        )
        df_sequence["defence_team_position"] = df_sequence["raw_state"].apply(
            lambda x: [player["position"] for player in x["defense_players"]]
        )
        df_sequence["defence_team_player"] = df_sequence["raw_state"].apply(
            lambda x: [player["player_name"] for player in x["defense_players"]]
        )

    return df_sequence


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        val = val.replace("nan", "math.nan")
        val = eval(val)
        return val


def preprocess_q_values(q_values):
    q_values["q_value"] = q_values["q_value"].apply(lambda x: x[7:-1])
    q_values["q_value"] = q_values["q_value"].apply(
        lambda x: np.array(safe_literal_eval(x.replace("\n", ""))) if isinstance(x, str) else x
    )
    return q_values


def normalize_values(values):
    """Normalize values to 0-1 range"""
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def create_radar_chart(ax, values, labels, title, color="blue", rotate=False):
    """Create a radar chart with improved visualization for similar values"""
    N = len(values)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = list(values) + [values[0]]

    # Ensure labels and values have the same length
    if len(labels) != len(values) - 1:  # -1 because values has duplicate first value
        # Truncate or pad labels to match values
        if len(labels) > len(values) - 1:
            labels = labels[: len(values) - 1]
        else:
            labels = labels + [f"Action_{i}" for i in range(len(labels), len(values) - 1)]

    if rotate:
        ax.set_theta_offset(0)
    else:
        ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, "o-", linewidth=3, color=color, markersize=8)
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])  # Exclude the duplicate point
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.08, fontsize=12, fontweight="bold")
    ax.grid(True)


def split_q_values(values, offball_action_idx, onball_action_idx):
    offball_values = values[offball_action_idx]
    onball_values = values[onball_action_idx]
    return offball_values, onball_values


def normalize_list(values):
    scaler = MinMaxScaler()
    values = np.array(values).reshape(-1, 1)
    return scaler.fit_transform(values).flatten()


def offside(attackers, defenders):
    attackers_x = [attacker["x"] for attacker in attackers]
    offside_attackers = [0] * len(attackers)
    defenders_min_x = min([defender["x"] for defender in defenders])
    offside_line = 52.5 if defenders_min_x < 52.5 else defenders_min_x

    for i, attacker_x in enumerate(attackers_x):
        if attacker_x > offside_line:
            offside_attackers[i] = 1

    return offside_attackers


def plot_q_values(df_sequence, q_values, name, team_name, match_id, sequence_id, output_path=None, test_mode=False):
    # Define action labels matching players_q_value_visualization.py structure
    offball_labels = ["left", "up_left", "up", "up_right", "right", "down_right", "down", "down_left"]
    onball_labels = ["shot", "dribble", "cross", "pass"]

    # Limit to 5 frames in test mode
    data_pairs = list(zip(df_sequence.iterrows(), q_values.iterrows()))
    if test_mode:
        data_pairs = data_pairs[:5]

    for idx, (data, q_value) in enumerate(data_pairs):
        # Create visualization layout like players_q_value_visualization.py
        fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(2, 3, hspace=0.4, width_ratios=[6, 4, 1])

        # Main field plot - spans both rows in left column
        ax_field = fig.add_subplot(gs[:, 0])
        mps.field(ax=ax_field, show=False, color="green")

        # Get ball position with coordinate transformation and scaling
        ball_x_raw = data[1]["ball_x"]
        ball_y_raw = data[1]["ball_y"]

        # Scale up coordinates to spread across field (multiply by field scale factor)
        # Since coordinates are clustered in ~0.5m range, scale by ~100 to spread across field
        scale_factor = 100
        ball_x = (ball_x_raw * scale_factor) + 52.5
        ball_y = (ball_y_raw * scale_factor) + 34

        ax_field.plot(ball_x, ball_y, "ko", markersize=10, label="Ball")

        # Plot attacking players with correct coordinates
        for position, player in zip(data[1]["attack_team_position"], data[1]["attack_team_player"]):
            # Coordinate transformation with scaling: field coordinates to visualization coordinates
            x_raw = position["x"]
            y_raw = position["y"]

            # Apply the same scaling factor to spread players across the field
            x = (x_raw * scale_factor) + 52.5
            y = (y_raw * scale_factor) + 34

            if player == name:  # Highlight target player
                ax_field.plot(x, y, "ro", markersize=16, markeredgecolor="yellow", markeredgewidth=3)
                ax_field.text(x, y + 2, player, ha="center", va="bottom", fontsize=12, fontweight="bold")
            else:
                ax_field.plot(x, y, "ro", markersize=12, alpha=0.5)

        # Plot defending players
        for position, player in zip(data[1]["defence_team_position"], data[1]["defence_team_player"]):
            x_raw = position["x"]
            y_raw = position["y"]

            # Apply the same scaling factor
            x = (x_raw * scale_factor) + 52.5
            y = (y_raw * scale_factor) + 34

            ax_field.plot(x, y, "bo", markersize=12, alpha=0.5)

            if player == name:  # Highlight if target player is on defense
                ax_field.plot(x, y, "bo", markersize=16, markeredgecolor="yellow", markeredgewidth=3)
                ax_field.text(x, y + 2, player, ha="center", va="bottom", fontsize=12, fontweight="bold")

        # Center field around ball position with ±25m range
        if ball_x - 25 < 0:
            field_x_min = 0
            field_x_max = 50
        elif ball_x + 25 > 105:
            field_x_min = 55
            field_x_max = 105
        else:
            field_x_min = ball_x - 25
            field_x_max = ball_x + 25

        ax_field.set_xlim(field_x_min, field_x_max)
        ax_field.set_ylim(0, 68)
        ax_field.set_title(f"{data[1]['team_name_attack']} vs {data[1]['team_name_defense']}", fontsize=16, fontweight="bold")
        ax_field.set_xticks([])
        ax_field.set_yticks([])

        # Get Q-values and split into onball/offball
        idle_q_value = q_value[1]["q_value_offball"][0]
        offball_q_values = q_value[1]["q_value_offball"][1:].copy()
        onball_q_values = q_value[1]["q_value_onball"].copy()

        # Handle NaN values
        offball_q_values = np.nan_to_num(offball_q_values, nan=0.0, posinf=0.0, neginf=0.0)
        onball_q_values = np.nan_to_num(onball_q_values, nan=0.0, posinf=0.0, neginf=0.0)
        idle_q_value = np.nan_to_num(idle_q_value, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure arrays have correct length for radar charts
        if len(offball_q_values) != len(offball_labels):
            # Pad or truncate to match labels
            if len(offball_q_values) < len(offball_labels):
                offball_q_values = np.pad(
                    offball_q_values, (0, len(offball_labels) - len(offball_q_values)), "constant", constant_values=0
                )
            else:
                offball_q_values = offball_q_values[: len(offball_labels)]

        if len(onball_q_values) != len(onball_labels):
            # Pad or truncate to match labels
            if len(onball_q_values) < len(onball_labels):
                onball_q_values = np.pad(
                    onball_q_values, (0, len(onball_labels) - len(onball_q_values)), "constant", constant_values=0
                )
            else:
                onball_q_values = onball_q_values[: len(onball_labels)]

        # Normalize for radar charts
        offball_normalized = normalize_values(offball_q_values)
        onball_normalized = normalize_values(onball_q_values)

        # Create onball radar chart - top middle
        ax_onball = fig.add_subplot(gs[0, 1], projection="polar")
        create_radar_chart(ax_onball, onball_normalized, onball_labels, f"Onball Q-values\n{name}", color="blue")

        # Create offball radar chart - bottom middle
        ax_offball = fig.add_subplot(gs[1, 1], projection="polar")
        create_radar_chart(ax_offball, offball_normalized, offball_labels, f"Offball Q-values\n{name}", color="red")

        # Save frame
        output_path = os.getcwd() + f"/test/data/figures/{match_id}/" if output_path is None else output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.4, wspace=0.3)
        plt.savefig(output_path + f"frame_{name}_{sequence_id}_{idx:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def movie_from_images(image_files, output_file):
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"MJPG"), 1, (width, height))

    for image in image_files:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

    for image in image_files:
        os.remove(image)


def create_movie(q_values_path, match_id, sequence_id, events_file_path, output_path=None, test_mode=False):
    target_file_path = q_values_path

    df = pd.DataFrame()
    data_list = []
    with open(events_file_path, "r") as file:
        for line in file:
            data_list.append(pd.read_json(StringIO(line)))

    df = pd.concat(data_list, axis=0)

    df_sequence = preprocess_tracking_data(df, sequence_id)

    q_values = pd.read_csv(target_file_path)

    q_values = preprocess_q_values(q_values)

    # create new columns for offball and onball q_values
    q_values[["q_value_offball", "q_value_onball"]] = pd.DataFrame(
        q_values["q_value"].apply(lambda x: split_q_values(x, offball_action_idx, onball_action_idx)).tolist(),
        index=q_values.index,
    )

    # standardize q_values
    q_values["q_value_offball"] = q_values["q_value_offball"].apply(normalize_list)
    q_values["q_value_onball"] = q_values["q_value_onball"].apply(normalize_list)

    player_name = q_values["player_name"].unique()

    # Limit to first player only in test mode
    if test_mode:
        player_name = player_name[:1]

    print(player_name)
    # Process only one player in test mode
    for i in range(1 if test_mode else len(player_name)):
        name = player_name[i]
        team_name = q_values["team_name"][q_values["player_name"] == name].values[0]

        print(f"player name: {name}")
        plot_q_values(
            df_sequence, q_values, name, team_name, match_id, sequence_id, output_path=output_path, test_mode=test_mode
        )
        image_file_path = os.getcwd() + f"/test/data/figures/{match_id}/" if output_path is None else output_path
        image_files = sorted(glob.glob(image_file_path + f"/frame_{name}_{sequence_id}*.png"))
        output_dir = os.getcwd() + "/test/data/movies/" if output_path is None else output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f"{match_id}_{sequence_id}_{name}.avi")
        movie_from_images(image_files, output_file)
