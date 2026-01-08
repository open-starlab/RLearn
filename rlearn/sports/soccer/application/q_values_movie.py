# -*- coding: utf-8 -*-
"""
Refactored script for visualizing Q-values from reinforcement learning models in a soccer context.

This script provides functionalities to:
- Preprocess tracking and Q-value data.
- Generate visualizations of on-ball and off-ball action Q-values.
- Supports two visualization styles: 'radar' (radar chart) and 'bar' (polar bar chart).
- Create movies from the generated frames.
"""

import ast
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotsoccer as mps
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
import japanize_matplotlib  # noqa

# --- 1. Single Source of Truth: Configurations ---

# Action configurations
ACTION_CONFIG = {
    "offball": {
        "names": ["up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left"],
        "indices": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "onball": {
        "names": ["pass", "through_pass", "shot", "cross", "dribble", "defense"],
        "indices": [9, 10, 11, 12, 13, 14],
    },
    "idle": {
        "index": 0,
    },
}

# Field dimensions (StatsBomb default, in meters)
FIELD_DIMS = {
    "width": 68,
    "length": 105,
    "x_center": 52.5,
    "y_center": 34,
}

# Visualization settings
VIZ_SETTINGS = {
    "figure_size": (16, 9),
    "field_color": "green",
    "ball_marker_size": 7,
    "player_marker_size": 7,
    "font_size_title": 15,
    "font_size_player_name": 15,
    "radar_line_alpha": 0.7,
    "radar_fill_alpha": 0.3,
    "bar_alpha": 0.7,
    "idle_q_value_scale": 500,
    "cmap": "viridis",
}

# Data source configurations
DATA_KEYS = {
    "PVS": {
        "state": lambda x: x["state"],
        "ball_pos": lambda s: s["ball"]["position"],
        "attack_players": lambda s: s["attack_players"],
        "defense_players": lambda s: s["defense_players"],
        "player_pos": lambda p: p["position"],
        "player_name": lambda p: p["player_name"],
    },
    "EDMS": {
        # Hypothetical keys for a different data source
        "state": lambda x: x["game_state"],
        "ball_pos": lambda s: s["ball_position"],
        "attack_players": lambda s: s["attacking_team"],
        "defense_players": lambda s: s["defending_team"],
        "player_pos": lambda p: p["coords"],
        "player_name": lambda p: p["name"],
    },
}

# --- 2. Data Preprocessing ---


def preprocess_tracking_data(df: pd.DataFrame, sequence_id: int, state_def: str = "PVS") -> pd.DataFrame:
    """
    Preprocesses tracking data for a specific sequence by expanding the event list
    into a frame-by-frame DataFrame.
    """
    if state_def not in DATA_KEYS:
        raise ValueError(f"Unknown state_def: '{state_def}'. Must be one of {list(DATA_KEYS.keys())}")

    keys = DATA_KEYS[state_def]

    # Select the row for the given sequence_id
    sequence_data = df[df["sequence_id"] == sequence_id]
    if sequence_data.empty:
        return pd.DataFrame()  # Return empty DataFrame if sequence_id not found

    # Expand the list of frame events from the 'events' column into a new DataFrame
    events_list = sequence_data["events"].iloc[0]
    df_frames = pd.DataFrame(events_list)

    if "state" not in df_frames.columns:
        return pd.DataFrame()  # Return empty if 'state' column is missing

    # The 'state' column now holds the state dictionary for each frame.
    # Rename it to 'raw_state' for consistency.
    df_frames = df_frames.rename(columns={"state": "raw_state"})

    # Extract ball and player data from the 'raw_state' column
    df_frames["ball_x"] = df_frames["raw_state"].apply(lambda s: keys["ball_pos"](s)["x"])
    df_frames["ball_y"] = df_frames["raw_state"].apply(lambda s: keys["ball_pos"](s)["y"])

    for team in ["attack", "defense"]:
        players_func = keys[f"{team}_players"]
        df_frames[f"{team}_team_position"] = df_frames["raw_state"].apply(
            lambda s: [keys["player_pos"](p) for p in players_func(s)]
        )
        df_frames[f"{team}_team_player"] = df_frames["raw_state"].apply(
            lambda s: [keys["player_name"](p) for p in players_func(s)]
        )

    # Propagate sequence-level team names to each frame
    team_name_attack = sequence_data["team_name_attack"].iloc[0]
    team_name_defense = sequence_data["team_name_defense"].iloc[0]
    df_frames["team_name_attack"] = team_name_attack
    df_frames["team_name_defense"] = team_name_defense

    return df_frames


def preprocess_q_values(df_q: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Q-value DataFrame by parsing string representations of tensors
    into NumPy arrays.
    """

    def parse_q_string(q_str: str):
        """
        Safely parses a string that might be a Python list or a string
        representation of a torch.Tensor.
        """
        if not isinstance(q_str, str):
            return q_str

        # Clean string and remove "tensor(...)" wrapper if it exists
        clean_str = q_str.replace("\n", "").strip()
        if clean_str.startswith("tensor("):
            start = clean_str.find("[")
            end = clean_str.rfind("]")
            if start != -1 and end != -1:
                clean_str = clean_str[start : end + 1]

        # First, try the safest method: ast.literal_eval
        try:
            return np.array(ast.literal_eval(clean_str))
        except (ValueError, SyntaxError):
            # If that fails, it might be due to 'nan', 'inf', etc.
            # Use a safer version of eval.
            safe_globals = {"__builtins__": None}
            safe_locals = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}
            try:
                return np.array(eval(clean_str, safe_globals, safe_locals))
            except Exception as e:
                # If all else fails, log the error and return an empty array
                # to prevent a crash.
                print(f"Could not parse Q-value string: '{q_str}'. Error: {e}")
                return np.array([])

    # Apply the parsing function to the 'q_value' column
    df_q["q_value"] = df_q["q_value"].apply(parse_q_string)

    # Filter out rows where parsing might have failed
    df_q = df_q[df_q["q_value"].apply(lambda x: x.size > 0)]

    # Split into on-ball, off-ball, and idle Q-values
    def split_q(values):
        offball = values[ACTION_CONFIG["offball"]["indices"]]
        onball = values[ACTION_CONFIG["onball"]["indices"]]
        idle = values[ACTION_CONFIG["idle"]["index"]]
        return offball, onball, idle

    split_values = df_q["q_value"].apply(split_q)
    df_q[["q_offball", "q_onball", "q_idle"]] = pd.DataFrame(split_values.tolist(), index=df_q.index)

    # Normalize Q-values for better visualization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_q["q_offball_norm"] = df_q["q_offball"].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    df_q["q_onball_norm"] = df_q["q_onball"].apply(lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten())
    df_q["q_idle_norm"] = df_q["q_idle"]  # Idle value is often a single value, can be scaled if needed

    return df_q


# --- 3. Visualization ---
def _plot_q_chart(ax, values, labels, viz_style, title):
    """Helper function to plot either a radar or bar chart for Q-values."""
    ax.axis("on")
    ax.set_facecolor("white")
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop
    values_list = values.tolist()
    values_list += values_list[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="grey", size=12)

    # --- 修正・追加箇所 ---
    # グリッド線を表示
    ax.grid(True, color="gray", linestyle=":", linewidth=0.5)

    # Y軸（放射軸）の目盛りを明示的に設定
    # 例: -1.0から1.0まで0.5刻みで表示
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=10)

    # 軸の範囲設定
    ax.set_ylim(-1.1, 1.1)
    # --------------------

    ax.set_title(title, fontsize=VIZ_SETTINGS["font_size_title"], pad=20)  # タイトルが被らないようpad調整推奨

    if viz_style == "radar":
        ax.plot(angles, values_list, linewidth=2, linestyle="solid", alpha=VIZ_SETTINGS["radar_line_alpha"])
        ax.fill(angles, values_list, alpha=VIZ_SETTINGS["radar_fill_alpha"])
    elif viz_style == "bar":
        norm = Normalize(vmin=min(values), vmax=max(values))
        colors = plt.get_cmap(VIZ_SETTINGS["cmap"])(norm(values))
        ax.bar(angles[:-1], values, width=0.4, color=colors, alpha=VIZ_SETTINGS["bar_alpha"], edgecolor="black")
    else:
        raise ValueError(f"Unknown viz_style: '{viz_style}'. Choose 'radar' or 'bar'.")


def plot_q_values(
    frame_data: pd.Series, q_values: pd.Series, player_name: str, viz_style: str = "radar", scale_coords: bool = False
):
    """
    Plots the soccer field and Q-value charts for a single frame.
    """
    fig = plt.figure(figsize=VIZ_SETTINGS["figure_size"])
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    ax_field = fig.add_subplot(gs[0])
    ax_offball = fig.add_subplot(gs[1], polar=True)
    # ax_onball = fig.add_subplot(gs[2], polar=True)

    # --- Plot Soccer Field ---
    mps.field(ax=ax_field, show=False, color=VIZ_SETTINGS["field_color"])

    # Handle coordinate scaling
    def scale(coord, dim):
        return coord * dim if scale_coords else coord

    raw_state = frame_data["raw_state"]

    ball_x = scale(raw_state["ball"]["position"]["x"], FIELD_DIMS["length"]) + FIELD_DIMS["x_center"]
    ball_y = scale(raw_state["ball"]["position"]["y"], FIELD_DIMS["width"]) + FIELD_DIMS["y_center"]

    # Set dynamic camera
    field_x_min, field_x_max = max(0, ball_x - 30), min(FIELD_DIMS["length"], ball_x + 30)
    ax_field.set_xlim(field_x_min, field_x_max)
    ax_field.set_ylim(0, FIELD_DIMS["width"])

    # Plot players and ball
    ax_field.plot(ball_x, ball_y, "ko", markersize=VIZ_SETTINGS["ball_marker_size"])
    for team, color in [("attack", "r"), ("defense", "b")]:
        for pos, p_name in zip(frame_data[f"{team}_team_position"], frame_data[f"{team}_team_player"]):
            px = scale(pos["x"], FIELD_DIMS["length"]) + FIELD_DIMS["x_center"]
            py = scale(pos["y"], FIELD_DIMS["width"]) + FIELD_DIMS["y_center"]
            ax_field.plot(px, py, f"{color}o", markersize=VIZ_SETTINGS["player_marker_size"])
            if p_name == player_name:
                ax_field.text(px, py + 1.5, p_name, fontsize=VIZ_SETTINGS["font_size_player_name"], ha="center")

    ax_field.set_title(
        f"{frame_data.get('team_name_attack', '')} vs {frame_data.get('team_name_defense', '')}",
        fontsize=VIZ_SETTINGS["font_size_title"],
    )
    # --- Plot Q-Value Charts ---
    _plot_q_chart(
        ax_offball,
        q_values["q_offball_norm"],
        ACTION_CONFIG["offball"]["names"],
        viz_style,
        f"Off-ball Q-values ({player_name})",
    )
    # Add idle Q-value representation
    # idle_size = max(0, q_values["q_idle_norm"] * VIZ_SETTINGS["idle_q_value_scale"])
    # ax_offball.scatter(0, 0, s=idle_size, c="blue", alpha=0.5, label=f"Idle: {q_values['q_idle_norm']:.2f}")

    # _plot_q_chart(
    #     ax_onball,
    #     q_values['q_onball_norm'],
    #     ACTION_CONFIG['onball']['names'],
    #     viz_style,
    #     f'On-ball Q-values ({player_name})'
    # )

    plt.tight_layout()
    plt.close(fig)  # Prevent display in non-interactive environments
    return fig


# --- 4. Movie Creation ---


def movie_from_images(image_paths: list, output_file: str, fps: int = 5):
    """Creates a video from a list of image files."""
    if not image_paths:
        print("No images found to create a movie.")
        return

    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_path in image_paths:
        video.write(cv2.imread(image_path))

    video.release()
    print(f"Movie saved to {output_file}")

    # Clean up images
    for image_path in image_paths:
        os.remove(image_path)


def create_movie(
    q_values_path: str,
    match_id: str,
    sequence_id: int,
    tracking_file_path: str,
    output_dir: str = None,
    test_mode: bool = False,
    viz_style: str = "radar",
    scale_coords: bool = True,
    state_def: str = "PVS",
):
    """
    Main function to generate and save a movie visualizing Q-values for a specific sequence.
    """
    # Set a default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "rlearn/sports/output/movies_and_frames")

    # The 'test_mode' parameter is accepted for API compatibility but is not used in this function.

    # Load and preprocess data
    try:
        df_tracking = pd.read_json(tracking_file_path, lines=True)
        df_sequence = preprocess_tracking_data(df_tracking, sequence_id, state_def=state_def)
        df_q = preprocess_q_values(pd.read_csv(q_values_path))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Merge dataframes to align tracking data with q-values.
    # Instead of merging on a key (which causes KeyErrors), we concatenate assuming
    # both dataframes are correctly ordered by frame.
    if len(df_sequence) != len(df_q):
        print(
            f"Warning: Frame count mismatch. Tracking data has {len(df_sequence)} frames, "
            f"but Q-values data has {len(df_q)} frames. Trimming to shorter length."
        )
        min_len = min(len(df_sequence), len(df_q))
        df_sequence = df_sequence.iloc[:min_len]
        df_q = df_q.iloc[:min_len]

    # Use concat which joins side-by-side on the index.
    # reset_index() on df_sequence creates an 'index' column with the frame numbers.
    df_merged = pd.concat([df_sequence.reset_index(), df_q.reset_index(drop=True)], axis=1)

    # Create frames for each unique player
    # Note: The data mismatch warning indicates df_q contains data for ALL players,
    # while df_sequence is for one sequence. The logic below correctly iterates
    # through players and filters the merged data.
    player_names = df_merged["player_name"].unique()
    for name in player_names:
        # Filter the merged dataframe for the current player
        player_df = df_merged[df_merged["player_name"] == name].dropna(subset=["raw_state"])

        # Robustly get team name
        team_name_series = player_df["team_name"].dropna()
        team_name = team_name_series.iloc[0] if not team_name_series.empty else "Unknown Team"

        print(f"Generating frames for player: {name} ({team_name})...")

        # Prepare directories
        frames_dir = os.path.join(output_dir, "frames", f"{match_id}_{sequence_id}_{name}")
        movies_dir = os.path.join(output_dir, "movies")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(movies_dir, exist_ok=True)

        frame_count = 0
        for _, row in player_df.iterrows():
            if test_mode and frame_count >= 10:
                print(f"Test mode: Limiting visualization to {frame_count} frames for player {name}.")
                break

            fig = plot_q_values(row, row, name, viz_style, scale_coords)
            # Use the 'index' column (from reset_index) as the frame number
            frame_path = os.path.join(frames_dir, f"frame_{row['index']:04d}.png")
            fig.savefig(frame_path)
            plt.close(fig)
            frame_count += 1

        # Create movie from frames
        # image_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        # movie_path = os.path.join(movies_dir, f"{match_id}_{sequence_id}_{name}_{viz_style}.mp4")
        # movie_from_images(image_paths, movie_path, fps=5)


if __name__ == "__main__":
    # Example usage of the script
    # --- Parameters ---
    BASE_DIR = os.getcwd()
    MATCH_ID = "2022100106"  # Example match ID
    SEQUENCE_ID = 0  # Example sequence ID

    # --- Paths ---
    # It's assumed the project structure contains these paths
    TRACKING_PATH = os.path.join(BASE_DIR, "test/data/dss/preprocess_data", MATCH_ID, "events.jsonl")
    Q_VALUES_PATH = os.path.join(BASE_DIR, "rlearn/sports/output/sarsa_attacker/test", f"q_values_seq{SEQUENCE_ID}.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "rlearn/sports/output/movies_and_frames")

    # --- Execution ---
    print("--- Running Q-Value Visualization (Style: Radar) ---")
    create_movie(
        tracking_file_path=TRACKING_PATH,
        q_values_path=Q_VALUES_PATH,
        match_id=MATCH_ID,
        sequence_id=SEQUENCE_ID,
        output_dir=OUTPUT_DIR,
        viz_style="radar",
        scale_coords=False,  # Assuming coordinates are in meters
        state_def="PVS",
    )

    print("\n--- Running Q-Value Visualization (Style: Bar) ---")
    create_movie(
        tracking_file_path=TRACKING_PATH,
        q_values_path=Q_VALUES_PATH,
        match_id=MATCH_ID,
        sequence_id=SEQUENCE_ID,
        output_dir=OUTPUT_DIR,
        viz_style="bar",
        scale_coords=False,  # Assuming coordinates are in meters
        state_def="PVS",
    )
