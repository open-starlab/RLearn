import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_q_values_to_csv(
    model, datamodule, state_def: str, output_dir, max_games: int = None, max_sequences_per_game: int = None
):
    """
    Save Q-values to CSV files for visualization purposes.
    All players' data for each sequence is saved in a single CSV file.

    Args:
        model: Trained Q-learning model
        datamodule: DataModule containing the datasets
        state_def: State definition ("EDMS" or "PVS")
        output_dir: Output directory for saving files
        max_games: Maximum number of games to process (None for all games)
        max_sequences_per_game: Maximum number of sequences per game (None for all sequences)
    """
    logger.info("Saving Q-values to CSV files...")

    # Count processed games and sequences per game
    processed_games = set()
    sequences_per_game = {}
    model.eval()
    model.to("cuda")

    # Group sequences by (game_id, sequence_id)
    sequence_groups = {}

    for data, batch in tqdm(
        zip(datamodule.train_dataset, datamodule.train_dataloader(batch_size=1, shuffle=False)),
        total=len(datamodule.train_dataset),
        desc="Processing sequences for Q-value CSV generation",
    ):
        game_id = data["game_id"]
        sequence_id = data["sequence_id"]

        # Check if we've reached the maximum number of games
        if max_games is not None and len(processed_games) >= max_games and game_id not in processed_games:
            continue

        # Initialize sequence counter for new games
        if game_id not in sequences_per_game:
            sequences_per_game[game_id] = 0

        # Check if we've reached the maximum sequences for this game
        if max_sequences_per_game is not None and sequences_per_game[game_id] >= max_sequences_per_game:
            continue

        # Process this sequence
        processed_games.add(game_id)
        sequences_per_game[game_id] += 1

        # Get Q-values from model
        q_values = (
            model(datamodule.transfer_batch_to_device(batch, "cuda", 0)).squeeze(0).detach().cpu()
        )  # (seq_len, num_actions)
        action_idx = batch["action"].squeeze(0)  # (seq_len, )

        # Gather q_values for the actions taken
        q_values_for_actions = q_values.gather(1, action_idx.unsqueeze(1)).squeeze(1).tolist()

        # Group by sequence
        seq_key = (game_id, sequence_id)
        if seq_key not in sequence_groups:
            sequence_groups[seq_key] = []

        sequence_groups[seq_key].append(
            {"data": data, "q_values": q_values, "action_idx": action_idx, "q_values_for_actions": q_values_for_actions}
        )

    # Create output directory
    figures_dir = output_dir / "players_q_state"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save combined CSV for each sequence
    for (game_id, sequence_id), players_data in sequence_groups.items():
        all_players_df = []

        for player_data in players_data:
            if state_def == "EDMS":
                player_df = _create_edms_dataframe(
                    player_data["data"],
                    player_data["q_values"],
                    player_data["action_idx"],
                    player_data["q_values_for_actions"],
                )
            elif state_def == "PVS":
                player_df = _create_pvs_dataframe(
                    player_data["data"],
                    player_data["q_values"],
                    player_data["action_idx"],
                    player_data["q_values_for_actions"],
                )
            else:
                logger.warning(f"Unsupported state definition: {state_def}")
                continue

            all_players_df.append(player_df)

        if all_players_df:
            # Combine all players' data for this sequence
            combined_df = pd.concat(all_players_df, ignore_index=True)

            # Save CSV file with combined data
            q_values_df_path = figures_dir / f"q_values_{game_id}_{sequence_id}.csv"
            combined_df.to_csv(q_values_df_path, index=False)
            logger.info(f"Saved Q-values CSV: {q_values_df_path}")

    logger.info("Q-values CSV generation completed.")


def _create_edms_dataframe(data, q_values, action_idx, q_values_for_actions):
    """Create DataFrame for EDMS state definition."""
    q_values_df = pd.DataFrame(
        index=range(len(data["sequence"])),
        columns=[
            "game_id",
            "sequence_id",
            "team_name",
            "player_name",
            "dist_ball_opponent",
            "dribble_score",
            "dist_goal",
            "angle_goal",
            "ball_speed",
            "shot_score",
            "long_ball_score",
            "fast_space",
            "dist_ball",
            "time_to_player",
            "time_to_passline",
            "variation_space",
            "pass_score",
            "dist_offside",
            "formation",
            "q_value",
            "action_idx",
            "q_values_for_actions",
        ],
    )

    for i, sequence in enumerate(data["sequence"]):
        q_values_df.loc[i, "game_id"] = data["game_id"]
        q_values_df.loc[i, "sequence_id"] = data["sequence_id"]
        q_values_df.loc[i, "team_name"] = sequence["observation"]["ego_player"]["team_name"]
        q_values_df.loc[i, "player_name"] = sequence["observation"]["ego_player"]["player_name"]

        absolute_state = sequence["observation"]["common_state"]["absolute_state"]
        onball_state = sequence["observation"]["common_state"]["onball_state"]
        offball_state = sequence["observation"]["common_state"]["offball_state"]

        # Fill state features
        q_values_df.loc[i, "dist_ball_opponent"] = onball_state.get("dist_ball_opponent")
        q_values_df.loc[i, "dribble_score"] = onball_state.get("dribble_score_vel")
        q_values_df.loc[i, "dist_goal"] = onball_state.get("dist_goal")
        q_values_df.loc[i, "angle_goal"] = onball_state.get("angle_goal")
        q_values_df.loc[i, "ball_speed"] = onball_state.get("ball_speed")
        q_values_df.loc[i, "shot_score"] = onball_state.get("shot_score")
        q_values_df.loc[i, "long_ball_score"] = onball_state.get("long_ball_score")
        q_values_df.loc[i, "fast_space"] = offball_state.get("fast_space")
        q_values_df.loc[i, "dist_ball"] = offball_state.get("dist_ball")
        q_values_df.loc[i, "time_to_player"] = offball_state.get("time_to_player")
        q_values_df.loc[i, "time_to_passline"] = offball_state.get("time_to_passline")
        q_values_df.loc[i, "variation_space"] = offball_state.get("variation_space")
        q_values_df.loc[i, "pass_score"] = offball_state.get("pass_score")
        q_values_df.loc[i, "dist_offside"] = absolute_state.get("dist_offside_line")
        q_values_df.loc[i, "formation"] = absolute_state.get("formation")
        q_values_df.loc[i, "q_value"] = q_values[i, :].tolist()
        q_values_df.loc[i, "action_idx"] = action_idx[i].item()
        q_values_df.loc[i, "q_values_for_actions"] = q_values_for_actions[i]

    return q_values_df


def _create_pvs_dataframe(data, q_values, action_idx, q_values_for_actions):
    """Create DataFrame for PVS state definition."""
    player = data["sequence"][0]["player"]

    q_values_df = pd.DataFrame(
        index=range(len(data["sequence"])),
        columns=["game_id", "sequence_id", "team_name", "player_name", "q_value", "action_idx", "q_values_for_actions"],
    )

    for i, sequence in enumerate(data["sequence"]):
        q_values_df.loc[i, "game_id"] = data["game_id"]
        q_values_df.loc[i, "sequence_id"] = data["sequence_id"]
        q_values_df.loc[i, "player_name"] = player["player_name"]
        q_values_df.loc[i, "q_value"] = q_values[i, :].tolist()
        q_values_df.loc[i, "action_idx"] = action_idx[i].item()
        q_values_df.loc[i, "q_values_for_actions"] = q_values_for_actions[i]

    return q_values_df
