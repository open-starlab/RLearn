from typing import List
import numpy as np
import math
import copy

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Position(BaseModel):
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(x=d["x"], y=d["y"])

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def angle_to(self, other: "Position") -> float:
        return math.atan2(other.y - self.y, other.x - self.x)


class Velocity(BaseModel):
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Velocity":
        return cls(x=d["x"], y=d["y"])


class Player(BaseModel):
    index: int
    team_name: str
    player_name: str
    player_id: int
    player_role: str
    position: Position
    velocity: Velocity
    action: str
    action_probs: List[float] | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "team_name": self.team_name,
            "player_name": self.player_name,
            "player_id": self.player_id,
            "player_role": self.player_role,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "action": self.action,
            "action_probs": self.action_probs or None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Player":
        return cls(
            index=d["index"],
            team_name=d["team_name"],
            player_name=d["player_name"],
            player_id=d["player_id"],
            player_role=d["player_role"],
            position=Position.from_dict(d["position"]),
            velocity=Velocity.from_dict(d["velocity"]),
            action=d["action"],
            action_probs=d["action_probs"] if "action_probs" in d else None,
        )


class Ball(BaseModel):
    position: Position
    velocity: Velocity

    def to_dict(self) -> dict:
        return {
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Ball":
        return cls(
            position=Position.from_dict(d["position"]),
            velocity=Velocity.from_dict(d["velocity"]),
        )


# Original State Features
class State(BaseModel):
    ball: Ball
    players: List[Player]
    attack_players: List[Player]
    defense_players: List[Player]

    @field_validator("attack_players", "defense_players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v

    def to_dict(self) -> dict:
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "attack_players": [player.to_dict() for player in self.attack_players],
            "defense_players": [player.to_dict() for player in self.defense_players],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class Observation(BaseModel):
    ball: Ball
    players: List[Player]  # without ego_player
    ego_player: Player

    @classmethod
    def from_state(cls, state: State, ego_player: Player) -> "Observation":
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @field_validator("players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v


class SimpleObservation(Observation):
    @classmethod
    def from_state(cls, state: State, ego_player: Player) -> "SimpleObservation":
        ego_position = ego_player.position
        ego_player_id = ego_player.player_id
        players = [
            Player(
                index=player.index,
                team_name=player.team_name,
                player_name=player.player_name,
                player_id=player.player_id,
                player_role=player.player_role,
                position=Position(
                    x=player.position.x - ego_position.x,
                    y=player.position.y - ego_position.y,
                ),
                velocity=player.velocity,
                action=player.action,
                action_probs=player.action_probs,
            )
            for player in state.players
            if player.player_id != ego_player_id
        ]
        ball = Ball(
            position=state.ball.position,
            velocity=state.ball.velocity,
        )
        return cls(ball=ball, players=players, ego_player=ego_player)

    def to_tensor(self) -> torch.Tensor:
        data = []
        for player in self.players:
            data.extend([player.position.x, player.position.y, player.velocity.x, player.velocity.y])
        data.extend([self.ball.position.x, self.ball.position.y, self.ball.velocity.x, self.ball.velocity.y])
        data.extend(
            [self.ego_player.position.x, self.ego_player.position.y, self.ego_player.velocity.x, self.ego_player.velocity.y]
        )
        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "ego_player": self.ego_player.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ball=Ball.from_dict(data["ball"]),
            players=[Player.from_dict(player) for player in data["players"]],
            ego_player=Player.from_dict(data["ego_player"]),
        )


class SimpleObservationAction(BaseModel):
    player: Player
    observation: SimpleObservation
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=SimpleObservation.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class SimpleObservationActionSequence(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[SimpleObservationAction]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[SimpleObservationAction.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class Event(BaseModel):
    state: State
    action: List[str] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event":
        if self.action is None:
            self.action = [player.action for player in self.state.players]
        for action in self.action:
            if not isinstance(action, str):
                raise TypeError("action must be a list of int")
        return self

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            state=State.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    events: List[Event]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event.from_dict(event) for event in d["events"]],
        )


# Position and Velocity Features
class State_PVS(BaseModel):
    ball: Ball
    players: List[Player]
    attack_players: List[Player]
    defense_players: List[Player]

    @field_validator("attack_players", "defense_players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v

    def to_dict(self) -> dict:
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "attack_players": [player.to_dict() for player in self.attack_players],
            "defense_players": [player.to_dict() for player in self.defense_players],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State_PVS":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class Observation_PVS(BaseModel):
    ball: Ball
    players: List[Player]  # without ego_player
    ego_player: Player

    @classmethod
    def from_state(cls, state: State_PVS, ego_player: Player) -> "Observation_PVS":
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @field_validator("players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v


class SimpleObservation_PVS(Observation_PVS):
    @classmethod
    def from_state(cls, state: State_PVS, ego_player: Player) -> "SimpleObservation_PVS":
        ego_position = ego_player.position
        ego_player_id = ego_player.player_id
        players = [
            Player(
                index=player.index,
                team_name=player.team_name,
                player_name=player.player_name,
                player_id=player.player_id,
                player_role=player.player_role,
                position=Position(
                    x=player.position.x - ego_position.x,
                    y=player.position.y - ego_position.y,
                ),
                velocity=player.velocity,
                action=player.action,
                action_probs=player.action_probs,
            )
            for player in state.players
            if player.player_id != ego_player_id
        ]
        ball = Ball(
            position=state.ball.position,
            velocity=state.ball.velocity,
        )
        return cls(ball=ball, players=players, ego_player=ego_player)

    def to_tensor(self) -> torch.Tensor:
        data = []
        for player in self.players:
            data.extend([player.position.x, player.position.y, player.velocity.x, player.velocity.y])
        data.extend([self.ball.position.x, self.ball.position.y, self.ball.velocity.x, self.ball.velocity.y])
        data.extend(
            [self.ego_player.position.x, self.ego_player.position.y, self.ego_player.velocity.x, self.ego_player.velocity.y]
        )
        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "ego_player": self.ego_player.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ball=Ball.from_dict(data["ball"]),
            players=[Player.from_dict(player) for player in data["players"]],
            ego_player=Player.from_dict(data["ego_player"]),
        )


class SimpleObservationAction_PVS(BaseModel):
    player: Player
    observation: SimpleObservation_PVS
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=SimpleObservation_PVS.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class SimpleObservationActionSequence_PVS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[SimpleObservationAction_PVS]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[SimpleObservationAction_PVS.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class Event_PVS(BaseModel):
    state: State_PVS
    action: List[str] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event_PVS":
        if self.action is None:
            self.action = [player.action for player in self.state.players]
        for action in self.action:
            if not isinstance(action, str):
                raise TypeError("action must be a list of int")
        return self

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event_PVS":
        return cls(
            state=State_PVS.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events_PVS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    sequence_start_frame: str
    sequence_end_frame: str
    team_name_attack: str
    team_name_defense: str
    events: List[Event_PVS]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event_PVS):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "sequence_start_frame": self.sequence_start_frame,
            "sequence_end_frame": self.sequence_end_frame,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events_PVS":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            sequence_start_frame=d["sequence_start_frame"],
            sequence_end_frame=d["sequence_end_frame"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event_PVS.from_dict(event) for event in d["events"]],
        )


# Extendable Dicision Making Features
class OnBall(BaseModel):
    dist_ball_opponent: List[float]
    dribble_score: List[float]
    dribble_score_vel: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    ball_speed: float
    transition: List[float]
    shot_score: float
    long_ball_score: List[float]

    def to_dict(self) -> dict:
        return {
            "dist_ball_opponent": self.dist_ball_opponent,
            "dribble_score": self.dribble_score,
            "dribble_score_vel": self.dribble_score_vel,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "ball_speed": self.ball_speed,
            "transition": self.transition,
            "shot_score": self.shot_score,
            "long_ball_score": self.long_ball_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OnBall":
        return cls(
            dist_ball_opponent=d["dist_ball_opponent"],
            dribble_score=d["dribble_score"],
            dribble_score_vel=d["dribble_score_vel"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            ball_speed=d["ball_speed"],
            transition=d["transition"],
            shot_score=d["shot_score"],
            long_ball_score=d["long_ball_score"],
        )


class OffBall(BaseModel):
    fast_space: List[float]
    fast_space_vel: List[float]
    dist_ball: List[float]
    angle_ball: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    time_to_player: List[float]
    time_to_passline: List[float]
    variation_space: List[List[float]]
    variation_space_vel: List[List[float]]
    defense_space: List[float]
    defense_space_vel: List[float]
    defense_dist_ball: List[float]

    def to_dict(self) -> dict:
        return {
            "fast_space": self.fast_space,
            "fast_space_vel": self.fast_space_vel,
            "dist_ball": self.dist_ball,
            "angle_ball": self.angle_ball,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "time_to_player": self.time_to_player,
            "time_to_passline": self.time_to_passline,
            "variation_space": self.variation_space,
            "variation_space_vel": self.variation_space_vel,
            "defense_space": self.defense_space,
            "defense_space_vel": self.defense_space_vel,
            "defense_dist_ball": self.defense_dist_ball,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OffBall":
        return cls(
            fast_space=d["fast_space"],
            fast_space_vel=d["fast_space_vel"],
            dist_ball=d["dist_ball"],
            angle_ball=d["angle_ball"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            time_to_player=d["time_to_player"],
            time_to_passline=d["time_to_passline"],
            variation_space=d["variation_space"],
            variation_space_vel=d["variation_space_vel"],
            defense_space=d["defense_space"],
            defense_space_vel=d["defense_space_vel"],
            defense_dist_ball=d["defense_dist_ball"],
        )


class RelativeState(BaseModel):
    onball: OnBall
    offball: OffBall

    def to_dict(self) -> dict:
        return {
            "onball": self.onball.to_dict(),
            "offball": self.offball.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelativeState":
        return cls(
            onball=OnBall.from_dict(d["onball"]),
            offball=OffBall.from_dict(d["offball"]),
        )


class AbsoluteState(BaseModel):
    dist_offside_line: List[float]
    formation: str
    attack_action: List[str]
    defense_action: List[str]

    def to_dict(self) -> dict:
        return {
            "dist_offside_line": self.dist_offside_line,
            "formation": self.formation,
            "attack_action": self.attack_action,
            "defense_action": self.defense_action,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AbsoluteState":
        return cls(
            dist_offside_line=d["dist_offside_line"],
            formation=d["formation"],
            attack_action=d["attack_action"],
            defense_action=d["defense_action"],
        )


class RawState(BaseModel):
    ball: Ball
    players: List[Player]
    attack_players: List[Player]
    defense_players: List[Player]

    @field_validator("attack_players", "defense_players")
    @classmethod
    def players_must_be_list_of_players(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("players must be a list")
        for player in v:
            if not isinstance(player, Player):
                raise TypeError("players must be a list of Player")
        return v

    def to_dict(self) -> dict:
        return {
            "ball": self.ball.to_dict(),
            "players": [player.to_dict() for player in self.players],
            "attack_players": [player.to_dict() for player in self.attack_players],
            "defense_players": [player.to_dict() for player in self.defense_players],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RawState":
        return cls(
            ball=Ball.from_dict(d["ball"]),
            players=[Player.from_dict(player) for player in d["players"]],
            attack_players=[Player.from_dict(player) for player in d["attack_players"]],
            defense_players=[Player.from_dict(player) for player in d["defense_players"]],
        )


class State_EDMS(BaseModel):
    relative_state: RelativeState
    absolute_state: AbsoluteState
    raw_state: RawState

    def __repr__(self):
        return f"State(relative_state={self.relative_state}, absolute_state={self.absolute_state}, raw_state={self.raw_state})"

    def to_dict(self) -> dict:
        return {
            "relative_state": self.relative_state.to_dict(),
            "absolute_state": self.absolute_state.to_dict(),
            "raw_state": self.raw_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "State_EDMS":
        return cls(
            relative_state=RelativeState(**d["relative_state"]),
            absolute_state=AbsoluteState(**d["absolute_state"]),
            raw_state=RawState(**d["raw_state"]),
        )


# Reinforcement Learning
class OnBall_RL(BaseModel):
    dist_ball_opponent: List[float]
    dribble_score: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    ball_speed: float
    transition: List[float]
    shot_score: float
    long_ball_score: List[float]

    def to_dict(self) -> dict:
        return {
            "dist_ball_opponent": self.dist_ball_opponent,
            "dribble_score_vel": self.dribble_score,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "ball_speed": self.ball_speed,
            "transition": self.transition,
            "shot_score": self.shot_score,
            "long_ball_score": self.long_ball_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OnBall_RL":
        return cls(
            dist_ball_opponent=d["dist_ball_opponent"],
            dribble_score=d["dribble_score_vel"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            ball_speed=d["ball_speed"],
            transition=d["transition"],
            shot_score=d["shot_score"],
            long_ball_score=d["long_ball_score"],
        )


class OffBall_RL(BaseModel):
    fast_space: List[float]
    dist_ball: List[float]
    angle_ball: List[float]
    dist_goal: List[float]
    angle_goal: List[float]
    time_to_player: List[float]
    time_to_passline: List[float]
    variation_space: List[List[float]]
    pass_score: List[float]

    @field_validator("variation_space")
    @classmethod
    def validate_variation_space(cls, v):  # type: ignore
        if not all(isinstance(i, list) for i in v):
            raise TypeError("variation_space must be a list of lists")
        return v

    def to_dict(self) -> dict:
        return {
            "fast_space": self.fast_space,
            "dist_ball": self.dist_ball,
            "angle_ball": self.angle_ball,
            "dist_goal": self.dist_goal,
            "angle_goal": self.angle_goal,
            "time_to_player": self.time_to_player,
            "time_to_passline": self.time_to_passline,
            "variation_space": self.variation_space,
            "pass_score": self.pass_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OffBall_RL":
        return cls(
            fast_space=d["fast_space"],
            dist_ball=d["dist_ball"],
            angle_ball=d["angle_ball"],
            dist_goal=d["dist_goal"],
            angle_goal=d["angle_goal"],
            time_to_player=d["time_to_player"],
            time_to_passline=d["time_to_passline"],
            variation_space=d["variation_space"],
            pass_score=d["pass_score"],
        )


class AbsoluteState_RL(BaseModel):
    dist_offside_line: List[float]
    formation: str

    def to_dict(self) -> dict:
        return {
            "dist_offside_line": self.dist_offside_line,
            "formation": self.formation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AbsoluteState_RL":
        return cls(
            dist_offside_line=d["dist_offside_line"],
            formation=d["formation"],
        )


class CommonState(BaseModel):
    onball_state: OnBall_RL
    offball_state: OffBall_RL
    absolute_state: AbsoluteState_RL

    @classmethod
    def from_dict(cls, data) -> "CommonState":
        return cls(
            onball_state=OnBall_RL.from_dict(data["onball_state"]),
            offball_state=OffBall_RL.from_dict(data["offball_state"]),
            absolute_state=AbsoluteState_RL.from_dict(data["absolute_state"]),
        )

    def to_dict(self) -> dict:
        return {
            "onball_state": self.onball_state.to_dict(),
            "offball_state": self.offball_state.to_dict(),
            "absolute_state": self.absolute_state.to_dict(),
        }


class Observation_EDMS(BaseModel):
    ego_player: Player
    common_state: CommonState

    @classmethod
    def from_state(cls, ego_player: Player, common_state: CommonState) -> "Observation_EDMS":
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError


class AgentCentricObservation(Observation_EDMS):
    @classmethod
    def onball_state(cls, onball_states: OnBall, state_def: str) -> OnBall_RL:
        # make input onball state the same size
        if not np.isnan(onball_states.shot_score):
            shot_score = onball_states.shot_score
            long_ball_score = [0, 0, 0]
        elif not np.all(np.isnan(onball_states.long_ball_score)):
            shot_score = 0
            long_ball_score = onball_states.long_ball_score
        else:
            shot_score = 0
            long_ball_score = [0, 0, 0]

        transition = [0] * 22
        ball_speed = 0
        dist_ball_opponent = [onball_states.dist_ball_opponent[0], 0]
        dist_goal = [onball_states.dist_goal[0], 0]
        angle_goal = [onball_states.angle_goal[0], 0]
        dribble_score = onball_states.dribble_score if state_def == "EDMS" else [max(onball_states.dribble_score)]

        onball_state = OnBall_RL(
            dist_ball_opponent=dist_ball_opponent,
            dribble_score=dribble_score,
            dist_goal=dist_goal,
            angle_goal=angle_goal,
            ball_speed=ball_speed,
            transition=transition,
            shot_score=shot_score,
            long_ball_score=long_ball_score,
        )

        return onball_state

    @staticmethod
    def closest_players_to_ball(target_state, source_players):
        target_x, target_y = target_state.position.x, target_state.position.y
        distances = []
        for idx, player in enumerate(source_players):
            player_x, player_y = player.position.x, player.position.y
            dist = ((target_x - player_x) ** 2 + (target_y - player_y) ** 2) ** 0.5
            distances.append((idx, dist))

        # Sort by distance and return indices in order of closest to farthest
        sorted_indices = [idx for idx, _ in sorted(distances, key=lambda x: x[1])]
        return sorted_indices

    @staticmethod
    def calc_pass_score(fast_space, dist_ball, time_to_player, time_to_passline):
        pass_score = []
        k_fast_space = 0.3
        k_dist_ball = 0.5
        k_time_to_player = 0.2
        k_time_to_passline = 0.2

        # calculate pass score for each player
        pass_score = [
            k_fast_space * fs + k_dist_ball * db + k_time_to_player * tp + k_time_to_passline * tpl
            for fs, db, tp, tpl in zip(fast_space, dist_ball, time_to_player, time_to_passline)
        ]

        return pass_score

    @staticmethod
    def pad_to_length(lst, length, padding=0, col=None):
        if col == "variation_space":
            lst.extend([[0] * 8] * (length - len(lst)))
        else:
            if len(lst) < length:
                lst.extend([padding] * (length - len(lst)))

        return lst

    @staticmethod
    def pad_player_indices(player_index: np.ndarray, required_length: int) -> np.ndarray:
        if len(player_index) >= required_length:
            return player_index[:required_length]

        if len(player_index) == 0:
            # 選手が一人もいない場合はゼロで埋める
            return np.zeros(required_length, dtype=int)

        # 循環パディング
        padded_indices = []
        for i in range(required_length):
            padded_indices.append(player_index[i % len(player_index)])

        return np.array(padded_indices)

    @classmethod
    def offball_observation(
        cls,
        offball_states: OffBall,
        raw_state: RawState,
        num_offball_players: int,
        number_of_players: int,
        ego_player_onball: int,
        ego_player_id: int,
        state_def: str,
        onball_player_idx: int = 0,
        gk_idx: int = None,
    ) -> OffBall_RL:
        if ego_player_onball:
            """
            If the ego player is on the ball, calculate pass score from offball states and select top 3 players offball state.
            """
            required_length = 10
            fast_space = cls.pad_to_length(offball_states.fast_space, required_length)
            dist_ball = cls.pad_to_length(offball_states.dist_ball, required_length)
            angle_ball = cls.pad_to_length(offball_states.angle_ball, required_length)
            dist_goal = cls.pad_to_length(offball_states.dist_goal, required_length)
            angle_goal = cls.pad_to_length(offball_states.angle_goal, required_length)
            time_to_player = cls.pad_to_length(offball_states.time_to_player, required_length)
            time_to_passline = cls.pad_to_length(offball_states.time_to_passline, required_length)
            variation_space = cls.pad_to_length(offball_states.variation_space, required_length, col="variation_space")

            pass_score = cls.calc_pass_score(
                fast_space=fast_space,
                dist_ball=dist_ball,
                time_to_player=time_to_player,
                time_to_passline=time_to_passline,
            )

            # get top num_offball_players index
            player_index = np.argsort(pass_score)[::-1][:num_offball_players]
            player_index = cls.pad_player_indices(player_index, num_offball_players)
            assert len(player_index) == num_offball_players, f"Expected {num_offball_players} players, got {len(player_index)}"

            variation_space_selected = []
            for idx in player_index:
                if idx < len(variation_space):
                    variation_space_selected.append(variation_space[idx])
                else:
                    variation_space_selected.append([0] * 8)

            while len(variation_space_selected) < num_offball_players:
                variation_space_selected.append([0] * 8)

            if state_def == "EDMS":
                if ego_player_onball:
                    variation_space = [list(v) if isinstance(v, list) else [v] for v in variation_space]
                else:
                    variation_space = [variation_space[number_of_players]] + [[0] * 8] * (num_offball_players - 1)
            elif state_def == "EDMS_max":
                if ego_player_onball:
                    variation_space = [[max(v)] if isinstance(v, list) else max([v]) for v in variation_space]
                else:
                    variation_space = [[max(variation_space[number_of_players])]] + [[0]] * (num_offball_players - 1)
            else:
                raise ValueError(f"Unknown state definition: {state_def}")

            offball_rl = OffBall_RL(
                fast_space=[fast_space[i] for i in player_index],
                dist_ball=[dist_ball[i] for i in player_index],
                angle_ball=[angle_ball[i] for i in player_index],
                dist_goal=[dist_goal[i] for i in player_index],
                angle_goal=[angle_goal[i] for i in player_index],
                time_to_player=[time_to_player[i] for i in player_index],
                time_to_passline=[time_to_passline[i] for i in player_index],
                variation_space=variation_space_selected,
                pass_score=[pass_score[i] for i in player_index],
            )

        else:
            """
            If the ego player is off the ball, use self's, onball player(if nobody is on the ball, closest to ball player), nearest teammate offball state.
            """
            self_idx = number_of_players

            # get index list of order of plaeyrs close to the ball
            sorted_indices = cls.closest_players_to_ball(raw_state.ball, raw_state.attack_players)
            ego_player_idx = next(
                (idx for idx, player in enumerate(raw_state.attack_players) if player.player_id == ego_player_id), 0
            )
            onball_idx = ego_player_idx if onball_player_idx > 0 else sorted_indices[0]

            # get index list of order of players close to the ego player
            nearest_player_idx = cls.closest_players_to_ball(raw_state.attack_players[self_idx], raw_state.attack_players)[1]

            # adjust indices considering the goalkeeper indices
            if gk_idx >= 0:
                onball_idx = onball_idx if onball_idx < gk_idx else onball_idx - 1

            nearest_player_idx = nearest_player_idx if nearest_player_idx < gk_idx else nearest_player_idx - 1

            try:
                offball_rl = OffBall_RL(
                    fast_space=[
                        offball_states.fast_space[self_idx],
                        offball_states.fast_space[onball_idx],
                        offball_states.fast_space[nearest_player_idx],
                    ],
                    dist_ball=[
                        offball_states.dist_ball[self_idx],
                        offball_states.dist_ball[onball_idx],
                        offball_states.dist_ball[nearest_player_idx],
                    ],
                    angle_ball=[
                        offball_states.angle_ball[self_idx],
                        offball_states.angle_ball[onball_idx],
                        offball_states.angle_ball[nearest_player_idx],
                    ],
                    dist_goal=[
                        offball_states.dist_goal[self_idx],
                        offball_states.dist_goal[onball_idx],
                        offball_states.dist_goal[nearest_player_idx],
                    ],
                    angle_goal=[
                        offball_states.angle_goal[self_idx],
                        offball_states.angle_goal[onball_idx],
                        offball_states.angle_goal[nearest_player_idx],
                    ],
                    time_to_player=[
                        offball_states.time_to_player[self_idx],
                        offball_states.time_to_player[onball_idx],
                        offball_states.time_to_player[nearest_player_idx],
                    ],
                    time_to_passline=[
                        offball_states.time_to_passline[self_idx],
                        offball_states.time_to_passline[onball_idx],
                        offball_states.time_to_passline[nearest_player_idx],
                    ],
                    variation_space=[
                        offball_states.variation_space[self_idx],
                        offball_states.variation_space[onball_idx],
                        offball_states.variation_space[nearest_player_idx],
                    ],
                    pass_score=[0, 0, 0],  # No pass score for offball player
                )
            except IndexError:
                breakpoint()
        return offball_rl

    @classmethod
    def from_state(
        cls,
        state: State_EDMS,
        ego_player: Player,
        ego_player_id: int,
        gk_idx: int,
        number_of_players: int,
        num_offball_players: int,
        onball_list: list,
        number_of_events: int,
        state_def: str,
    ) -> "SimpleObservation_EDMS":
        """
        Create ego player observation from state.
        """

        attack_action = ["pass", "dribble", "shot", "through_pass", "cross"]

        onball_states = copy.deepcopy(state.relative_state.onball)
        offball_states = copy.deepcopy(state.relative_state.offball)
        absolute_states = copy.deepcopy(state.absolute_state)
        raw_state = copy.deepcopy(state.raw_state)

        # remove GK state from offball states
        for attr_name, attr_value in state.relative_state.offball.__dict__.items():
            if isinstance(attr_value, list) and len(attr_value) > gk_idx and gk_idx >= 0:
                new_list = attr_value[:gk_idx] + attr_value[gk_idx + 1 :]
                setattr(offball_states, attr_name, new_list)

        # Check if the ego_player is onball player.
        ego_player_onball = 1 if ego_player.action in attack_action else 0
        if ego_player_onball:
            # ego player is onball player
            onball_state = cls.onball_state(onball_states, state_def)
            offball_state = cls.offball_observation(
                offball_states,
                raw_state,
                num_offball_players,
                number_of_players,
                ego_player_onball,
                ego_player_id,
                state_def,
            )
        else:
            # ego player is offball player
            onball_state = OnBall_RL(
                dist_ball_opponent=onball_states.dist_ball_opponent,
                dribble_score=[0] * 8 if state_def == "EDMS" else [0],
                dist_goal=onball_states.dist_goal,
                angle_goal=onball_states.angle_goal,
                ball_speed=onball_states.ball_speed,
                transition=onball_states.transition,
                shot_score=0,
                long_ball_score=[0, 0, 0],
            )

            # Check if there is an onball player in the current state
            onball_player_idx = 0
            if onball_list[number_of_events] > 0:
                onball_player_idx = onball_list[number_of_events]

            offball_state = cls.offball_observation(
                offball_states,
                raw_state,
                num_offball_players,
                number_of_players,
                ego_player_onball,
                ego_player_id,
                state_def,
                onball_player_idx,
                gk_idx,
            )

        absolute_state = AbsoluteState_RL(
            dist_offside_line=absolute_states.dist_offside_line, formation=absolute_states.formation
        )

        common_state = CommonState(onball_state=onball_state, offball_state=offball_state, absolute_state=absolute_state)

        return cls(ego_player=ego_player, common_state=common_state)

    def to_tensor(self, direction) -> torch.Tensor:
        data = []
        common_state = []
        if self.common_state.absolute_state.formation != "":
            self.common_state.absolute_state.formation = self.common_state.absolute_state.formation.replace(" ", "_")
        else:
            self.common_state.absolute_state.formation = "0"

        self.common_state.absolute_state.formation = (
            int(self.common_state.absolute_state.formation)
            if self.common_state.absolute_state.formation and not math.isnan(float(self.common_state.absolute_state.formation))
            else 442
        )

        self.common_state.absolute_state.formation = int(self.common_state.absolute_state.formation)

        ego_player = [
            self.ego_player.position.x,
            self.ego_player.position.y,
            self.ego_player.velocity.x,
            self.ego_player.velocity.y,
        ]
        ego_player = [0 if math.isnan(value) else value for value in ego_player]

        if direction == 1:
            common_state = [
                *self.common_state.onball_state.dist_ball_opponent[:2],
                self.common_state.onball_state.dribble_score[0],
                *self.common_state.onball_state.dist_goal[:2],
                *self.common_state.onball_state.angle_goal[:2],
                self.common_state.onball_state.ball_speed,
                self.common_state.onball_state.shot_score,
                *self.common_state.onball_state.long_ball_score[:3],
                *self.common_state.offball_state.fast_space[:3],
                *self.common_state.offball_state.dist_ball[:3],
                *self.common_state.offball_state.angle_ball[:3],
                *self.common_state.offball_state.dist_goal[:3],
                *self.common_state.offball_state.angle_goal[:3],
                *self.common_state.offball_state.time_to_player[:3],
                *self.common_state.offball_state.time_to_passline[:3],
                self.common_state.offball_state.variation_space[0][0],
                self.common_state.offball_state.variation_space[1][0],
                self.common_state.offball_state.variation_space[2][0],
                *self.common_state.offball_state.pass_score[:3],
                *self.common_state.absolute_state.dist_offside_line[:2],
                self.common_state.absolute_state.formation,
            ]
        else:
            for col, value in self.common_state.offball_state.__dict__.items():
                if col == "variation_space":
                    if len(value) != 3:
                        value.extend([[0] * 8] * (3 - len(value)))
                else:
                    if len(value) < 3:
                        value.extend([0] * (3 - len(value)))

            for col, value in self.common_state.onball_state.__dict__.items():
                if col == "dribble_score":
                    if len(value) != 8:
                        value.extend([0] * (8 - len(value)))

            common_state = [
                *self.common_state.onball_state.dist_ball_opponent[:2],
                *self.common_state.onball_state.dribble_score[:8],
                *self.common_state.onball_state.dist_goal[:2],
                *self.common_state.onball_state.angle_goal[:2],
                self.common_state.onball_state.ball_speed,
                self.common_state.onball_state.shot_score,
                *self.common_state.onball_state.long_ball_score[:3],
                *self.common_state.offball_state.fast_space[:3],
                *self.common_state.offball_state.dist_ball[:3],
                *self.common_state.offball_state.angle_ball[:3],
                *self.common_state.offball_state.dist_goal[:3],
                *self.common_state.offball_state.angle_goal[:3],
                *self.common_state.offball_state.time_to_player[:3],
                *self.common_state.offball_state.time_to_passline[:3],
                *self.common_state.offball_state.variation_space[0][:8],
                *self.common_state.offball_state.variation_space[1][:8],
                *self.common_state.offball_state.variation_space[2][:8],
                *self.common_state.offball_state.pass_score[:3],
                *self.common_state.absolute_state.dist_offside_line[:2],
                self.common_state.absolute_state.formation,
            ]

        common_state = [0 if math.isnan(value) else value for value in common_state]
        common_state = [0 if math.isinf(value) else value for value in common_state]

        data.extend(ego_player)
        data.extend(common_state)

        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):
        return {
            "ego_player": self.ego_player.to_dict(),
            "common_state": self.common_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ego_player=Player.from_dict(data["ego_player"]),
            common_state=CommonState.from_dict(data["common_state"]),
        )


class AgentCentricObservationAction(BaseModel):
    player: Player
    observation: AgentCentricObservation
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=AgentCentricObservation.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class AgentCentricObservationActionSequence(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[AgentCentricObservationAction]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[AgentCentricObservationAction.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class SimpleObservation_EDMS(Observation_EDMS):
    @classmethod
    def adjust_onball_state(cls, onball_states: OnBall, state_def: str) -> OnBall_RL:
        # make input onball state the same size
        if not np.isnan(onball_states.shot_score):
            shot_score = onball_states.shot_score
            long_ball_score = [0, 0, 0]
        elif not np.all(np.isnan(onball_states.long_ball_score)):
            shot_score = 0
            long_ball_score = onball_states.long_ball_score
        else:
            shot_score = 0
            long_ball_score = [0, 0, 0]

        transition = [0] * 22
        ball_speed = 0
        dist_ball_opponent = [onball_states.dist_ball_opponent[0], 0]
        dist_goal = [onball_states.dist_goal[0], 0]
        angle_goal = [onball_states.angle_goal[0], 0]
        dribble_score = onball_states.dribble_score if state_def == "EDMS" else [max(onball_states.dribble_score)]

        onball_state = OnBall_RL(
            dist_ball_opponent=dist_ball_opponent,
            dribble_score=dribble_score,
            dist_goal=dist_goal,
            angle_goal=angle_goal,
            ball_speed=ball_speed,
            transition=transition,
            shot_score=shot_score,
            long_ball_score=long_ball_score,
        )

        return onball_state

    @staticmethod
    def closest_player_to_ball(raw_state: RawState):
        ball_x, ball_y = raw_state.ball.position.x, raw_state.ball.position.y
        closest_player_idx = np.inf
        for idx, player in enumerate(raw_state.attack_players):
            player_x, player_y = player.position.x, player.position.y
            dist = ((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2) ** 0.5
            if dist < closest_player_idx:
                closest_player_idx = idx
        return closest_player_idx

    @classmethod
    def offball_observation(
        cls,
        offball_states: OffBall,
        num_offball_players: int,
        onball_flag: int,
        number_of_players: int,
        ego_player_onball: int,
        state_def: str,
    ) -> OffBall_RL:
        def pad_to_length(lst, length, padding=0, col=None):
            if col == "variation_space":
                lst.extend([[0] * 8] * (length - len(lst)))
            else:
                if len(lst) < length:
                    lst.extend([padding] * (length - len(lst)))

        fast_space = offball_states.fast_space_vel
        dist_ball = offball_states.dist_ball
        angle_ball = offball_states.angle_ball
        dist_goal = offball_states.dist_goal
        angle_goal = offball_states.angle_goal
        time_to_player = offball_states.time_to_player
        time_to_passline = offball_states.time_to_passline
        variation_space = offball_states.variation_space_vel

        pass_score = []
        required_length = 11

        pad_to_length(fast_space, required_length)
        pad_to_length(dist_ball, required_length)
        pad_to_length(angle_ball, required_length)
        pad_to_length(dist_goal, required_length)
        pad_to_length(angle_goal, required_length)
        pad_to_length(time_to_player, required_length)
        pad_to_length(time_to_passline, required_length)
        pad_to_length(variation_space, required_length, col="variation_space")

        k_fast_space = 0.3
        k_dist_ball = 0.5
        k_time_to_player = 0.2
        k_time_to_passline = 0.2

        # calculate pass score for each player
        pass_score = [
            k_fast_space * fs + k_dist_ball * db + k_time_to_player * tp + k_time_to_passline * tpl
            for fs, db, tp, tpl in zip(fast_space, dist_ball, time_to_player, time_to_passline)
        ]

        # get top num_offball_players index
        player_index = np.argsort(pass_score)[:num_offball_players]
        if len(player_index) < num_offball_players:
            player_index = np.pad(player_index, (0, num_offball_players - len(player_index)), "constant", constant_values=0)

        # if len(player_index) != num_offball_players:
        #     import pdb

        #     pdb.set_trace()
        assert len(player_index) == num_offball_players, "Number of offball players is not equal to num_offball_players"

        variation_space = [variation_space[i] for i in player_index]
        if len(variation_space) < required_length:
            variation_space.extend([[0] * 8] * (required_length - len(variation_space)))

        if state_def == "EDMS":
            if ego_player_onball:
                variation_space = [list(v) if isinstance(v, list) else [v] for v in variation_space]
            else:
                variation_space = [variation_space[number_of_players]] + [[0] * 8] * (num_offball_players - 1)
        elif state_def == "EDMS_max":
            if ego_player_onball:
                variation_space = [[max(v)] if isinstance(v, list) else max([v]) for v in variation_space]
            else:
                variation_space = [[max(variation_space[number_of_players])]] + [[0]] * (num_offball_players - 1)

        if onball_flag:
            if ego_player_onball:
                offball_rl = OffBall_RL(
                    fast_space=[fast_space[i] for i in player_index],
                    dist_ball=[dist_ball[i] for i in player_index],
                    angle_ball=[angle_ball[i] for i in player_index],
                    dist_goal=[dist_goal[i] for i in player_index],
                    angle_goal=[angle_goal[i] for i in player_index],
                    time_to_player=[time_to_player[i] for i in player_index],
                    time_to_passline=[time_to_passline[i] for i in player_index],
                    variation_space=variation_space,
                    pass_score=[pass_score[i] for i in player_index],
                )
            else:
                offball_rl = OffBall_RL(
                    fast_space=[fast_space[number_of_players]] + [0] * (num_offball_players - 1),
                    dist_ball=[dist_ball[number_of_players]] + [0] * (num_offball_players - 1),
                    angle_ball=[angle_ball[number_of_players]] + [0] * (num_offball_players - 1),
                    dist_goal=[dist_goal[number_of_players]] + [0] * (num_offball_players - 1),
                    angle_goal=[angle_goal[number_of_players]] + [0] * (num_offball_players - 1),
                    time_to_player=[time_to_player[number_of_players]] + [0] * (num_offball_players - 1),
                    time_to_passline=[time_to_passline[number_of_players]] + [0] * (num_offball_players - 1),
                    variation_space=variation_space,
                    pass_score=[0] * num_offball_players,
                )
        else:
            if ego_player_onball:
                offball_rl = OffBall_RL(
                    fast_space=[fast_space[i] for i in player_index],
                    dist_ball=[dist_ball[i] for i in player_index],
                    angle_ball=[angle_ball[i] for i in player_index],
                    dist_goal=[dist_goal[i] for i in player_index],
                    angle_goal=[angle_goal[i] for i in player_index],
                    time_to_player=[time_to_player[i] for i in player_index],
                    time_to_passline=[time_to_passline[i] for i in player_index],
                    variation_space=variation_space,
                    pass_score=[pass_score[i] for i in player_index],
                )
            else:
                offball_rl = OffBall_RL(
                    fast_space=[fast_space[number_of_players]] + [0] * (num_offball_players - 1),
                    dist_ball=[dist_ball[number_of_players]] + [0] * (num_offball_players - 1),
                    angle_ball=[angle_ball[number_of_players]] + [0] * (num_offball_players - 1),
                    dist_goal=[dist_goal[number_of_players]] + [0] * (num_offball_players - 1),
                    angle_goal=[angle_goal[number_of_players]] + [0] * (num_offball_players - 1),
                    time_to_player=[time_to_player[number_of_players]] + [0] * (num_offball_players - 1),
                    time_to_passline=[time_to_passline[number_of_players]] + [0] * (num_offball_players - 1),
                    variation_space=variation_space,
                    pass_score=[0] * num_offball_players,
                )

        return offball_rl

    @classmethod
    def from_state(
        cls,
        state: State_EDMS,
        ego_player: Player,
        number_of_players: int,
        num_offball_players: int,
        onball_list: list,
        number_of_events: int,
        onball_flag: int,
        state_def: str,
    ) -> "SimpleObservation_EDMS":
        attack_action = ["pass", "dribble", "shot", "through_pass", "cross"]

        onball_states = state.relative_state.onball
        offball_states = state.relative_state.offball
        absolute_states = state.absolute_state
        raw_state = state.raw_state
        # if ego_player.action in attack_action or (onball_list[number_of_events] == 1 and not any([player.action in attack_action for player in state.raw_state.players]) and onball_flag == 1):
        #     onball_flag = 1
        #     onball_state = cls.adjust_onball_state(onball_states)
        #     offball_state = cls.calc_pass_score(offball_states, num_offball_players, onball_flag)

        # elif not ego_player.action in attack_action and (onball_list[number_of_events] == 1 and any([player.action in attack_action for player in state.raw_state.players]) and onball_flag == 1):
        if onball_list[number_of_events] == 1 and any([player.action in attack_action for player in state.raw_state.players]):
            onball_flag = 1
            ego_player_onball = 1 if ego_player.action in attack_action else 0
            onball_state = cls.adjust_onball_state(onball_states, state_def)
            offball_state = cls.offball_observation(
                offball_states, num_offball_players, onball_flag, number_of_players, ego_player_onball, state_def
            )
        else:
            onball_flag = 0
            onball_state = OnBall_RL(
                dist_ball_opponent=onball_states.dist_ball_opponent,
                dribble_score=[0] * 8 if state_def == "EDMS" else [0],
                dist_goal=onball_states.dist_goal,
                angle_goal=onball_states.angle_goal,
                ball_speed=onball_states.ball_speed,
                transition=onball_states.transition,
                shot_score=0,
                long_ball_score=[0, 0, 0],
            )

            closest_player_idx = cls.closest_player_to_ball(raw_state=raw_state)
            ego_player_onball = 1 if ego_player.player_id == raw_state.attack_players[closest_player_idx].player_id else 0

            # remove the closest player state from the offball state
            if closest_player_idx in offball_states.fast_space:
                offball_states.fast_space.pop(closest_player_idx)

            offball_states.dist_ball.pop(closest_player_idx)
            offball_states.time_to_player.pop(closest_player_idx)
            offball_states.time_to_passline.pop(closest_player_idx)

            # assuming that the most closest player to the ball is the onball player, then calculate the offball state
            offball_state = cls.offball_observation(
                offball_states, num_offball_players, onball_flag, number_of_players, ego_player_onball, state_def
            )

        absolute_state = AbsoluteState_RL(
            dist_offside_line=absolute_states.dist_offside_line, formation=absolute_states.formation
        )

        common_state = CommonState(onball_state=onball_state, offball_state=offball_state, absolute_state=absolute_state)

        onball_flag = 0 if any([player.action in attack_action for player in state.raw_state.players]) else 1

        return cls(ego_player=ego_player, common_state=common_state), onball_flag

    def to_tensor(self, direction) -> torch.Tensor:
        data = []
        common_state = []
        if self.common_state.absolute_state.formation != "":
            self.common_state.absolute_state.formation = self.common_state.absolute_state.formation.replace(" ", "_")
        else:
            self.common_state.absolute_state.formation = "0"

        self.common_state.absolute_state.formation = (
            int(self.common_state.absolute_state.formation)
            if self.common_state.absolute_state.formation and not math.isnan(float(self.common_state.absolute_state.formation))
            else 442
        )

        self.common_state.absolute_state.formation = int(self.common_state.absolute_state.formation)

        ego_player = [
            self.ego_player.position.x,
            self.ego_player.position.y,
            self.ego_player.velocity.x,
            self.ego_player.velocity.y,
        ]
        ego_player = [0 if math.isnan(value) else value for value in ego_player]

        if direction == 1:
            common_state = [
                *self.common_state.onball_state.dist_ball_opponent[:2],
                self.common_state.onball_state.dribble_score[0],
                *self.common_state.onball_state.dist_goal[:2],
                *self.common_state.onball_state.angle_goal[:2],
                self.common_state.onball_state.ball_speed,
                self.common_state.onball_state.shot_score,
                *self.common_state.onball_state.long_ball_score[:3],
                *self.common_state.offball_state.fast_space[:3],
                *self.common_state.offball_state.dist_ball[:3],
                *self.common_state.offball_state.angle_ball[:3],
                *self.common_state.offball_state.dist_goal[:3],
                *self.common_state.offball_state.angle_goal[:3],
                *self.common_state.offball_state.time_to_player[:3],
                *self.common_state.offball_state.time_to_passline[:3],
                self.common_state.offball_state.variation_space[0][0],
                self.common_state.offball_state.variation_space[1][0],
                self.common_state.offball_state.variation_space[2][0],
                *self.common_state.offball_state.pass_score[:3],
                *self.common_state.absolute_state.dist_offside_line[:2],
                self.common_state.absolute_state.formation,
            ]
        else:
            for col, value in self.common_state.offball_state.__dict__.items():
                if col == "variation_space":
                    if len(value) != 3:
                        value.extend([[0] * 8] * (3 - len(value)))
                else:
                    if len(value) < 3:
                        value.extend([0] * (3 - len(value)))

            for col, value in self.common_state.onball_state.__dict__.items():
                if col == "dribble_score":
                    if len(value) != 8:
                        value.extend([0] * (8 - len(value)))

            common_state = [
                *self.common_state.onball_state.dist_ball_opponent[:2],
                *self.common_state.onball_state.dribble_score[:8],
                *self.common_state.onball_state.dist_goal[:2],
                *self.common_state.onball_state.angle_goal[:2],
                self.common_state.onball_state.ball_speed,
                self.common_state.onball_state.shot_score,
                *self.common_state.onball_state.long_ball_score[:3],
                *self.common_state.offball_state.fast_space[:3],
                *self.common_state.offball_state.dist_ball[:3],
                *self.common_state.offball_state.angle_ball[:3],
                *self.common_state.offball_state.dist_goal[:3],
                *self.common_state.offball_state.angle_goal[:3],
                *self.common_state.offball_state.time_to_player[:3],
                *self.common_state.offball_state.time_to_passline[:3],
                *self.common_state.offball_state.variation_space[0][:8],
                *self.common_state.offball_state.variation_space[1][:8],
                *self.common_state.offball_state.variation_space[2][:8],
                *self.common_state.offball_state.pass_score[:3],
                *self.common_state.absolute_state.dist_offside_line[:2],
                self.common_state.absolute_state.formation,
            ]

        common_state = [0 if math.isnan(value) else value for value in common_state]
        common_state = [0 if math.isinf(value) else value for value in common_state]

        data.extend(ego_player)
        data.extend(common_state)

        return torch.tensor(data)

    @property
    def dimension(self) -> int:
        return len(self.to_tensor())

    def to_dict(self):
        return {
            "ego_player": self.ego_player.to_dict(),
            "common_state": self.common_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            ego_player=Player.from_dict(data["ego_player"]),
            common_state=CommonState.from_dict(data["common_state"]),
        )


class SimpleObservationAction_EDMS(BaseModel):
    player: Player
    observation: SimpleObservation_EDMS
    action: str
    reward: float

    def to_dict(self):
        return {
            "player": self.player.to_dict(),
            "observation": self.observation.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            player=Player.from_dict(data["player"]),
            observation=SimpleObservation_EDMS.from_dict(data["observation"]),
            action=data["action"],
            reward=data["reward"],
        )


class SimpleObservationActionSequence_EDMS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    team_name_attack: str
    team_name_defense: str
    sequence: List[SimpleObservationAction_EDMS]

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "sequence": [obs_action.to_dict() for obs_action in self.sequence],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            game_id=data["game_id"],
            half=data["half"],
            sequence_id=data["sequence_id"],
            team_name_attack=data["team_name_attack"],
            team_name_defense=data["team_name_defense"],
            sequence=[SimpleObservationAction_EDMS.from_dict(obs_action) for obs_action in data["sequence"]],
        )


class Event_EDMS(BaseModel):
    state: State_EDMS
    action: List[List[str]] | None = None
    reward: float

    @model_validator(mode="after")
    def set_and_validate_action(self) -> "Event_EDMS":
        if self.action is None:
            self.action = [self.state.absolute_state.attack_action, self.state.absolute_state.defense_action]
        # for action in self.action:
        #     if not isinstance(action, str):
        #         raise TypeError("action must be a list of int")
        return self

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event_EDMS":
        return cls(
            state=State_EDMS.from_dict(d["state"]),
            action=d["action"],
            reward=d["reward"],
        )


class Events_EDMS(BaseModel):
    game_id: str
    half: str
    sequence_id: int
    sequence_start_frame: str
    sequence_end_frame: str
    team_name_attack: str
    team_name_defense: str
    events: List[Event_EDMS]

    @field_validator("events")
    @classmethod
    def events_must_be_list_of_events(cls, v):  # type: ignore
        if not isinstance(v, list):
            raise TypeError("events must be a list")
        for event in v:
            if not isinstance(event, Event_EDMS):
                raise TypeError("events must be a list of Event")
        return v

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "half": self.half,
            "sequence_id": self.sequence_id,
            "sequence_start_frame": self.sequence_start_frame,
            "sequence_end_frame": self.sequence_end_frame,
            "team_name_attack": self.team_name_attack,
            "team_name_defense": self.team_name_defense,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Events_EDMS":
        return cls(
            game_id=d["game_id"],
            half=d["half"],
            sequence_id=d["sequence_id"],
            sequence_start_frame=d["sequence_start_frame"],
            sequence_end_frame=d["sequence_end_frame"],
            team_name_attack=d["team_name_attack"],
            team_name_defense=d["team_name_defense"],
            events=[Event_EDMS.from_dict(event) for event in d["events"]],
        )


class ObservationactionInstance(BaseModel):
    observation: torch.Tensor  # (events_len, n_agents, obs_dim)
    action: torch.Tensor  # (events_len, n_agents)
    reward: torch.Tensor  # (events_len, n_agents)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservationactionBatch(BaseModel):
    observation: torch.Tensor  # (batch_size, max_events_len, n_agents, obs_dim)
    action: torch.Tensor  # (batch_size, max_events_len, n_agents)
    reward: torch.Tensor  # (batch_size, max_events_len, n_agents)
    mask: torch.Tensor  # (batch_size, max_events_len, n_agents)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Prediction(BaseModel):
    q_values: torch.Tensor  # (batch_size, max_events_len, n_agents, action_dim)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ObservationactionForLMInstance(BaseModel):
    sequence: List[int]
    action_mask: List[int]  # 1 if action, else 0


class ObservationActionForLMBatch(BaseModel):
    sequence: torch.Tensor  # (batch_size, max_seq_len )
    mask: torch.Tensor  # (batch_size, max_seq_len)
    action_mask: torch.Tensor  # (batch_size, max_seq_len)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ActionPrediction(BaseModel):
    q_values: torch.Tensor  # (batch_size, max_events_len, action_dim)
    model_config = ConfigDict(arbitrary_types_allowed=True)
