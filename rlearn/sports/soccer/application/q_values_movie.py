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
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

# Action definitions
onball_action_names = ["pass", "through_pass", "shot", "cross", "dribble", "defense"]

offball_action_names = [
    "idle",
    "up",
    "up_right",
    "right",
    "down_right",
    "down",
    "down_left",
    "left",
    "up_left",
]

offball_action_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
onball_action_idx = [9, 10, 11, 12, 13, 14]


@dataclass
class CoordinationAnalysis:
    """Data class for coordination analysis results"""

    team_q_value: float
    individual_contributions: Dict[str, float]
    coordination_score: float
    peak_moments: List[int]
    synchronization_index: float


class SoccerAnalyzer:
    """Enhanced Soccer Analyzer with team coordination capabilities"""

    def __init__(self, config: Dict, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks if model path provided
        if model_path:
            self.agent_net = self._load_agent_network(model_path)
            self.mixer_net = self._load_mixer_network(model_path)
        else:
            self.agent_net = None
            self.mixer_net = None

        # Coordination analysis parameters
        self.coordination_threshold = config.get("coordination_threshold", 0.7)
        self.sync_window = config.get("sync_window", 5)

    def _load_agent_network(self, model_path: str):
        """Load agent network from checkpoint"""
        # Will edit this section once I run all the test for Qmix
        return None

    def _load_mixer_network(self, model_path: str):
        """Load mixer network from checkpoint"""
        # Will edit this section once I run all the test for Qmix
        return None

    def analyze_team_coordination(
        self, states: np.ndarray, observations: np.ndarray
    ) -> CoordinationAnalysis:
        """Analyze team coordination at current timestep"""

        # If models are loaded, use them for analysis
        if self.agent_net and self.mixer_net:
            team_q = self._compute_team_q_value(states, observations)
            individual_contribs = self._compute_individual_contributions(observations)
        else:
            # Fallback to heuristic analysis
            team_q = self._heuristic_team_coordination(states, observations)
            individual_contribs = self._heuristic_individual_contributions(observations)

        # Calculate coordination metrics
        coordination_score = self._calculate_coordination_score(individual_contribs)
        sync_index = self._calculate_synchronization_index(observations)

        return CoordinationAnalysis(
            team_q_value=team_q,
            individual_contributions=individual_contribs,
            coordination_score=coordination_score,
            peak_moments=[],  # Will be filled by time series analysis
            synchronization_index=sync_index,
        )

    def _compute_team_q_value(
        self, states: np.ndarray, observations: np.ndarray
    ) -> float:
        """Compute team Q-value using loaded models"""
        return 0.5  # Placeholder

    def _compute_individual_contributions(
        self, observations: np.ndarray
    ) -> Dict[str, float]:
        """Compute individual agent contributions to team performance"""
        return {f"player_{i}": np.random.random() for i in range(11)}  # Placeholder

    def _heuristic_team_coordination(
        self, states: np.ndarray, observations: np.ndarray
    ) -> float:
        """Heuristic-based team coordination calculation"""
        return np.random.random()  # Placeholder

    def _heuristic_individual_contributions(
        self, observations: np.ndarray
    ) -> Dict[str, float]:
        """Heuristic-based individual contribution calculation"""
        return {f"player_{i}": np.random.random() for i in range(11)}

    def _calculate_coordination_score(
        self, individual_contribs: Dict[str, float]
    ) -> float:
        """Calculate overall coordination score"""
        values = list(individual_contribs.values())
        # Coordination is higher when contributions are more balanced
        return 1.0 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0.0

    def _calculate_synchronization_index(self, observations: np.ndarray) -> float:
        """Calculate team synchronization index"""
        # Placeholder for synchronization calculation
        return np.random.random()


class IntegratedSoccerAnalysis:
    """Main class integrating Q-value visualization with coordination analysis"""

    def __init__(self, config: Dict):
        self.config = config
        self.analyzer = SoccerAnalyzer(config, config.get("model_path"))

    def preprocess_tracking_data(
        self, df: pd.DataFrame, sequence_id: int
    ) -> pd.DataFrame:
        """Preprocess tracking data for analysis"""
        df_sequence = df[df["sequence_id"] == sequence_id].copy()
        df_sequence.loc[:, "raw_state"] = df_sequence["events"].apply(
            lambda x: x["state"]
        )
        df_sequence = df_sequence.drop(columns=["events"])
        df_sequence.reset_index(drop=True, inplace=True)

        # Extract ball position
        df_sequence["ball_x"] = df_sequence["raw_state"].apply(
            lambda x: x["ball"]["position"]["x"]
        )
        df_sequence["ball_y"] = df_sequence["raw_state"].apply(
            lambda x: x["ball"]["position"]["y"]
        )

        # Extract team positions and actions
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

    def safe_literal_eval(self, val):
        """Safely evaluate literal expressions"""
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            val = val.replace("nan", "math.nan")
            val = eval(val)
            return val

    def preprocess_q_values(self, q_values: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Q-values data"""
        q_values["q_value"] = q_values["q_value"].apply(lambda x: x[7:-1])
        q_values["q_value"] = q_values["q_value"].apply(
            lambda x: np.array(self.safe_literal_eval(x.replace("\n", "")))
            if isinstance(x, str)
            else x
        )
        return q_values

    def normalize_list(self, values: List[float]) -> List[float]:
        """Normalize list of values"""
        scaler = MinMaxScaler()
        values = np.array(values).reshape(-1, 1)
        return scaler.fit_transform(values).flatten()

    def split_q_values(
        self, values: np.ndarray, offball_idx: List[int], onball_idx: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split Q-values into offball and onball actions"""
        offball_values = values[offball_idx]
        onball_values = values[onball_idx]
        return offball_values, onball_values

    def analyze_coordination_timeline(
        self, df_sequence: pd.DataFrame
    ) -> List[CoordinationAnalysis]:
        """Analyze coordination over the entire sequence"""
        coordination_timeline = []

        for idx, row in df_sequence.iterrows():
            # Extract state and observations (simplified)
            states = np.array([row["ball_x"], row["ball_y"]])  # Simplified state
            observations = np.array(
                [[pos["x"], pos["y"]] for pos in row["attack_team_position"]]
            )

            # Perform coordination analysis
            analysis = self.analyzer.analyze_team_coordination(states, observations)
            coordination_timeline.append(analysis)

        # Identify peak coordination moments
        team_q_values = [analysis.team_q_value for analysis in coordination_timeline]
        peaks = self._find_peaks(team_q_values)

        for i, peak_idx in enumerate(peaks):
            if peak_idx < len(coordination_timeline):
                coordination_timeline[peak_idx].peak_moments.append(peak_idx)

        return coordination_timeline

    def _find_peaks(self, values: List[float], threshold: float = 0.1) -> List[int]:
        """Find peaks in coordination values"""
        peaks = []
        for i in range(1, len(values) - 1):
            if (
                values[i] > values[i - 1]
                and values[i] > values[i + 1]
                and values[i] > threshold
            ):
                peaks.append(i)
        return peaks

    def plot_enhanced_q_values(
        self,
        df_sequence: pd.DataFrame,
        q_values: pd.DataFrame,
        coordination_timeline: List[CoordinationAnalysis],
        name: str,
        team_name: str,
        match_id: str,
        sequence_id: int,
    ):
        """Enhanced plotting with coordination analysis"""

        offball_directions = ["U", "U-L", "L", "B-L", "B", "B-R", "R", "U-R"]
        onball_actions = onball_action_names
        angles_offball = np.linspace(0, 2 * np.pi, len(offball_directions) + 1)
        angles_onball = np.linspace(0, 2 * np.pi, len(onball_actions) + 1)
        cmap = cm.viridis

        for idx, (data, q_value, coordination) in enumerate(
            zip(df_sequence.iterrows(), q_values.iterrows(), coordination_timeline)
        ):
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(
                3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1]
            )

            # Main field visualization
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_field_with_coordination(
                ax1, data[1], coordination, name, team_name, idx
            )

            # Individual Q-values (offball)
            ax2 = fig.add_subplot(gs[0, 2], polar=True)
            self._plot_offball_q_values(
                ax2, q_value[1], offball_directions, angles_offball, cmap, name
            )

            # Team coordination timeline
            ax3 = fig.add_subplot(gs[1, :])
            self._plot_coordination_timeline(ax3, coordination_timeline, idx)

            # Coordination metrics
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_coordination_metrics(ax4, coordination)

            # Individual contributions
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_individual_contributions(ax5, coordination, name)

            # Synchronization index
            ax6 = fig.add_subplot(gs[2, 2])
            self._plot_synchronization_index(ax6, coordination)

            # Save figure
            output_path = os.path.join(os.getcwd(), f"tests/data/figures/{match_id}/")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_path, f"enhanced_frame_{name}_{sequence_id}_{idx:04d}.png"
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            if idx >= 10:  # Limit frames for demo
                break

    def _plot_field_with_coordination(
        self,
        ax,
        data,
        coordination: CoordinationAnalysis,
        name: str,
        team_name: str,
        idx: int,
    ):
        """Plot field with coordination indicators"""
        mps.field(ax=ax, show=False, color="green")
        ball_x, ball_y = data["ball_x"] + 52.5, data["ball_y"] + 34

        # Set field bounds
        if ball_x - 25 < -2:
            field_x_min, field_x_max = -2, 55
        elif ball_x + 25 > 107:
            field_x_min, field_x_max = 45, 80
        else:
            field_x_min, field_x_max = ball_x - 25, ball_x + 25

        ax.set_xlim(field_x_min, field_x_max)
        ax.set_ylim(-2, 70)

        # Plot ball
        ax.plot(ball_x, ball_y, "ko", markersize=8)

        # Plot players with coordination-based colors
        coordination_norm = Normalize(vmin=0, vmax=1)
        coord_color = cm.RdYlGn(coordination_norm(coordination.team_q_value))

        for position, player in zip(
            data["attack_team_position"], data["attack_team_player"]
        ):
            player_x, player_y = position["x"] + 52.5, position["y"] + 34

            # Highlight coordinated players
            if player == name:
                ax.plot(
                    player_x,
                    player_y,
                    "ro",
                    markersize=10,
                    markeredgecolor="gold",
                    markeredgewidth=2,
                )
                ax.text(
                    player_x - 2,
                    player_y + 2,
                    player,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
            else:
                ax.plot(player_x, player_y, "ro", markersize=8)

        # Plot defense players
        for position in data["defence_team_position"]:
            ax.plot(position["x"] + 52.5, position["y"] + 34, "bo", markersize=8)

        # Add coordination indicator
        coord_text = f"Team Coordination: {coordination.team_q_value:.3f}"
        ax.text(
            0.02,
            0.98,
            coord_text,
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor=coord_color, alpha=0.8),
        )

        # Highlight peak coordination moments
        if idx in coordination.peak_moments:
            ax.add_patch(
                plt.Rectangle(
                    (field_x_min, -2),
                    field_x_max - field_x_min,
                    72,
                    fill=False,
                    edgecolor="gold",
                    linewidth=4,
                )
            )
            ax.text(
                0.5,
                0.02,
                "★ PEAK COORDINATION ★",
                transform=ax.transAxes,
                fontsize=16,
                ha="center",
                color="gold",
                weight="bold",
            )

        ax.set_title(
            f'{data["team_name_attack"]} vs {data["team_name_defense"]} - Frame {idx}',
            fontsize=16,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_offball_q_values(self, ax, q_value, directions, angles, cmap, name):
        """Plot offball Q-values on polar plot"""
        idle_q_values = q_value["q_value_offball"][0]
        offball_q_values = q_value["q_value_offball"][1:]

        offball_q_values = np.nan_to_num(
            offball_q_values, nan=0.0, posinf=0.0, neginf=0.0
        )
        idle_q_values = np.nan_to_num(idle_q_values, nan=0.0, posinf=0.0, neginf=0.0)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(directions, color="grey", fontsize=12)
        ax.set_ylim(-1, max(offball_q_values) + 0.3)
        ax.set_title(f"Q-Values: {name}", fontsize=14)

        # Plot bars
        norm = Normalize(vmin=min(offball_q_values), vmax=max(offball_q_values))
        colors = cmap(norm(offball_q_values))

        for i in range(len(offball_q_values)):
            ax.bar(
                angles[i],
                offball_q_values[i],
                width=(angles[i + 1] - 0.2 - angles[i]),
                color=colors[i],
                edgecolor="black",
                alpha=0.7,
            )
            ax.text(angles[i], max(offball_q_values), f"{offball_q_values[i]:.3f}")

        # Plot idle action
        scatter_size = max(abs(idle_q_values) * 1000, 50)
        ax.scatter(0, -1, s=scatter_size, c="blue", alpha=0.7)
        ax.text(
            0,
            -1,
            f"{idle_q_values:.3f}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    def _plot_coordination_timeline(self, ax, coordination_timeline, current_idx):
        """Plot team coordination over time"""
        team_q_values = [coord.team_q_value for coord in coordination_timeline]
        coordination_scores = [
            coord.coordination_score for coord in coordination_timeline
        ]

        x_values = range(len(team_q_values))

        ax.plot(x_values, team_q_values, "b-", label="Team Q-Value", linewidth=2)
        ax.plot(
            x_values,
            coordination_scores,
            "r--",
            label="Coordination Score",
            linewidth=2,
        )

        # Highlight current frame
        ax.axvline(x=current_idx, color="gold", linestyle="-", linewidth=3, alpha=0.7)

        # Mark peak moments
        for coord in coordination_timeline:
            for peak in coord.peak_moments:
                ax.axvline(x=peak, color="green", linestyle=":", alpha=0.8)

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title("Team Coordination Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_coordination_metrics(self, ax, coordination: CoordinationAnalysis):
        """Plot coordination metrics"""
        metrics = ["Team Q-Value", "Coordination Score", "Sync Index"]
        values = [
            coordination.team_q_value,
            coordination.coordination_score,
            coordination.synchronization_index,
        ]

        bars = ax.bar(metrics, values, color=["blue", "red", "green"], alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_title("Coordination Metrics")
        ax.set_ylabel("Value")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    def _plot_individual_contributions(
        self, ax, coordination: CoordinationAnalysis, current_player: str
    ):
        """Plot individual player contributions"""
        players = list(coordination.individual_contributions.keys())
        contributions = list(coordination.individual_contributions.values())

        colors = [
            "gold" if player == current_player else "skyblue" for player in players
        ]

        bars = ax.bar(range(len(players)), contributions, color=colors, alpha=0.7)
        ax.set_xticks(range(len(players)))
        ax.set_xticklabels([p.replace("player_", "P") for p in players], rotation=45)
        ax.set_title("Individual Contributions")
        ax.set_ylabel("Contribution")

    def _plot_synchronization_index(self, ax, coordination: CoordinationAnalysis):
        """Plot synchronization index as a gauge"""
        # Simple gauge visualization
        angles = np.linspace(0, np.pi, 100)
        sync_angle = coordination.synchronization_index * np.pi

        ax.plot(angles, np.ones_like(angles), "k-", linewidth=8, alpha=0.3)
        ax.plot([0, sync_angle], [0, 1], "r-", linewidth=4)
        ax.scatter([sync_angle], [1], color="red", s=100, zorder=5)

        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"Sync Index: {coordination.synchronization_index:.3f}")
        ax.set_xticks([0, np.pi / 2, np.pi])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([])

    def create_enhanced_analysis(
        self,
        q_values_path: str,
        match_id: str,
        sequence_id: int,
        tracking_data_path: Optional[str] = None,
    ):
        """Create enhanced analysis with coordination insights"""

        # Load tracking data
        if tracking_data_path is None:
            tracking_data_path = os.path.join(
                os.getcwd(), f"test/data/dss/preprocess_data/{match_id}/events.jsonl"
            )

        df = pd.DataFrame()
        data_list = []

        with open(tracking_data_path, "r") as file:
            for line in file:
                data_list.append(pd.read_json(StringIO(line)))

        df = pd.concat(data_list, axis=0)
        df_sequence = self.preprocess_tracking_data(df, sequence_id)

        # Load Q-values
        q_values = pd.read_csv(q_values_path)
        q_values = self.preprocess_q_values(q_values)

        # Split and normalize Q-values
        q_values[["q_value_offball", "q_value_onball"]] = pd.DataFrame(
            q_values["q_value"]
            .apply(
                lambda x: self.split_q_values(x, offball_action_idx, onball_action_idx)
            )
            .tolist(),
            index=q_values.index,
        )

        q_values["q_value_offball"] = q_values["q_value_offball"].apply(
            self.normalize_list
        )
        q_values["q_value_onball"] = q_values["q_value_onball"].apply(
            self.normalize_list
        )

        # Perform coordination analysis
        coordination_timeline = self.analyze_coordination_timeline(df_sequence)

        # Generate enhanced visualizations for each player
        player_names = q_values["player_name"].unique()

        for player_name in player_names:
            team_name = q_values[q_values["player_name"] == player_name][
                "team_name"
            ].values[0]
            player_q_values = q_values[q_values["player_name"] == player_name]

            print(f"Generating enhanced analysis for: {player_name}")

            self.plot_enhanced_q_values(
                df_sequence,
                player_q_values,
                coordination_timeline,
                player_name,
                team_name,
                match_id,
                sequence_id,
            )

            # Generate summary report
            self._generate_player_report(
                player_name, coordination_timeline, match_id, sequence_id
            )

            break  # For demo, process only first player (strictly)

    def _generate_player_report(
        self,
        player_name: str,
        coordination_timeline: List[CoordinationAnalysis],
        match_id: str,
        sequence_id: int,
    ):
        """Generate a summary report for player analysis"""

        # Calculate summary statistics
        team_q_values = [coord.team_q_value for coord in coordination_timeline]
        coordination_scores = [
            coord.coordination_score for coord in coordination_timeline
        ]
        sync_indices = [coord.synchronization_index for coord in coordination_timeline]

        report = {
            "player_name": player_name,
            "match_id": match_id,
            "sequence_id": sequence_id,
            "summary_stats": {
                "avg_team_q_value": np.mean(team_q_values),
                "max_team_q_value": np.max(team_q_values),
                "avg_coordination_score": np.mean(coordination_scores),
                "avg_synchronization": np.mean(sync_indices),
                "peak_coordination_moments": len(
                    [coord for coord in coordination_timeline if coord.peak_moments]
                ),
            },
        }

        # Save report
        output_path = os.path.join(os.getcwd(), f"tests/data/reports/{match_id}/")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(
            os.path.join(
                output_path, f"player_report_{player_name}_{sequence_id}.json"
            ),
            "w",
        ) as f:
            json.dump(report, f, indent=2)

        print(f"Report generated for {player_name}")
        print(
            f"Average team coordination: {report['summary_stats']['avg_team_q_value']:.3f}"
        )
        print(
            f"Peak coordination moments: {report['summary_stats']['peak_coordination_moments']}"
        )

    def create_coordination_movie(self, image_pattern: str, output_file: str):
        """Create movie from enhanced analysis frames"""
        image_files = sorted(glob.glob(image_pattern))

        if not image_files:
            print(f"No images found with pattern: {image_pattern}")
            return

        frame = cv2.imread(image_files[0])
        height, width, _ = frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video = cv2.VideoWriter(output_file, fourcc, 2.0, (width, height))

        for image_file in image_files:
            frame = cv2.imread(image_file)
            video.write(frame)

        video.release()
        cv2.destroyAllWindows()

        print(f"Enhanced coordination movie created: {output_file}")


def main():
    """Main function to run the integrated analysis"""

    # Configuration
    config = {
        "coordination_threshold": 0.7,
        "sync_window": 5,
        "model_path": None,  # Set to your model path if available
        "data_dir": "./test/data/",
    }

    # Initialize integrated analyzer
    analyzer = IntegratedSoccerAnalysis(config)

    # Example usage - replace with actual paths
    q_values_path = "path/to/your/q_values.csv"
    match_id = "example_match"
    sequence_id = 1

    try:
        # Create enhanced analysis
        analyzer.create_enhanced_analysis(q_values_path, match_id, sequence_id)

        # Create coordination movie
        image_pattern = (
            f"tests/data/figures/{match_id}/enhanced_frame_*_{sequence_id}_*.png"
        )
        output_movie = (
            f"tests/data/movies/enhanced_coordination_{match_id}_{sequence_id}.avi"
        )
        analyzer.create_coordination_movie(image_pattern, output_movie)

        print("Enhanced soccer analysis completed successfully!")

    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Please ensure all required data files are available.")


if __name__ == "__main__":
    main()
