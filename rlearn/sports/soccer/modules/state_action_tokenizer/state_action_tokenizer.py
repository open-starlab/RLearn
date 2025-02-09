from typing import Dict, List

from tango.common import Registrable

from rlearn.sports.soccer.constant import FIELD_LENGTH, FIELD_WIDTH
from rlearn.sports.soccer.dataclass import Position, Velocity
from rlearn.sports.soccer.modules.state_action_tokenizer.preprocess_frames import discretize_direction


class StateActionTokenizerBase(Registrable):
    def __init__(
        self, action2id: Dict[str, int], position_granularity: float = 0.5, origin_pos: str = "center"
    ) -> None:
        self.action2id = action2id
        self.position_granularity = position_granularity  # unit: m
        self.origin_pos = origin_pos
        self.num_actions = len(action2id)

    def encode(self, token: str | Position | Velocity) -> int:
        raise NotImplementedError

    def decode(self, token_id: int) -> str | Position:
        raise NotImplementedError

    def _action2id(self, action: str) -> int:
        return self.action2id[action]

    def _velocity2action(self, velocity: Velocity) -> str:
        raise NotImplementedError

    def _position2id(self, position: Position) -> int:
        raise NotImplementedError

    @property
    def encode_position_separately(self) -> bool:
        raise NotImplementedError


@StateActionTokenizerBase.register("simple")
class SimpleStateActionTokenizer(StateActionTokenizerBase):
    def __init__(
        self,
        action2id: Dict[str, int],
        position_granularity: float = 0.5,
        origin_pos: str = "center",
        special_tokens: List[str] = ["[SEP]", "[PAD]", "[ACTION_SEP]"],
    ) -> None:
        self.action2id = action2id
        self.id2action = {id_: action for action, id_ in action2id.items()}
        self.position_granularity = position_granularity  # unit: m
        self.origin_pos = origin_pos
        self.num_actions = len(action2id)
        self.num_positions = int(2 * FIELD_LENGTH / position_granularity) * int(2 * FIELD_WIDTH / position_granularity)
        self.num_normal_tokens = self.num_actions + self.num_positions
        self.special_tokens = special_tokens
        self.num_tokens = self.num_normal_tokens + len(self.special_tokens)

    def encode_position_separately(self) -> bool:
        return False

    def encode(self, token: str | Position | Velocity) -> int:
        # if str, then token is action -> return action_id
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.num_normal_tokens + self.special_tokens.index(token)
            else:
                return self._action2id(token)
        elif isinstance(token, Position):
            return self._position2id(token)
        elif isinstance(token, Velocity):
            return self._action2id(self._velocity2action(token))
        else:
            raise TypeError(f"token must be str or Position, but got {type(token)}")

    def decode(self, token_id: int) -> str | Position:
        if token_id < self.num_actions:
            return self._id2action(token_id)
        elif token_id < self.num_normal_tokens:
            return self._id2position(token_id)
        elif token_id < self.num_normal_tokens + len(self.special_tokens):
            return self.special_tokens[token_id - self.num_normal_tokens]
        else:
            raise ValueError(f"token_id must be less than {self.num_normal_tokens + len(self.special_tokens)}")

    def _action2id(self, action: str) -> int:
        return self.action2id[action]

    def _position2id(self, position: Position) -> int:
        # position.x is [-FIELD_LENGTH, FIELD_LENGTH] -> discretize to 0.5m (note that this is relative coordinate)
        # position.y is [-FIELD_WIDTH, FIELD_WIDTH] -> discretize to 0.5m (note that this is relative coordinate)
        x = int(
            (position.x + FIELD_LENGTH) / self.position_granularity
        )  # [0, 2 * FIELD_LENGTH / position_granularity]
        y = int((position.y + FIELD_WIDTH) / self.position_granularity)  #  [0, 2 * FIELD_WIDTH / position_granularity]
        assert 0 <= x <= int(2 * FIELD_LENGTH / self.position_granularity), f"position.x: {position.x}, x: {x}"
        assert 0 <= y <= int(2 * FIELD_WIDTH / self.position_granularity), f"position.y: {position.y}, y: {y}"
        return (
            x * int(2 * FIELD_WIDTH / self.position_granularity) + y + self.num_actions
        )  # [0, 2 * FIELD_LENGTH / position_granularity * 2 * FIELD_WIDTH / position_granularity + num_actions]

    def _velocity2action(self, velocity: Velocity) -> str:
        return discretize_direction(velocity.x, velocity.y)


@StateActionTokenizerBase.register("xy_separate")
class XYSeparateTokenizer(StateActionTokenizerBase):
    def __init__(
        self,
        action2id: Dict[str, int],
        position_granularity: float = 0.5,
        origin_pos: str = "center",
        special_tokens: List[str] = ["[SEP]", "[PAD]", "[ACTION_SEP]"],
    ) -> None:
        self.action2id = action2id
        self.id2action = {id_: action for action, id_ in action2id.items()}
        self.position_granularity = position_granularity  # unit: m
        self.origin_pos = origin_pos
        self.num_actions = len(action2id)
        self.num_positions = int(2 * FIELD_LENGTH / position_granularity)
        self.num_normal_tokens = self.num_actions + self.num_positions
        self.special_tokens = special_tokens
        self.num_tokens = self.num_normal_tokens + len(self.special_tokens)

    def encode_position_separately(self) -> bool:
        return True

    def encode(self, token: str | float | Velocity) -> int:
        # if str, then token is action -> return action_id
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.num_normal_tokens + self.special_tokens.index(token)
            else:
                return self._action2id(token)
        elif isinstance(token, float):
            return self._coordinate2id(token)
        elif isinstance(token, Velocity):
            return self._action2id(self._velocity2action(token))
        else:
            raise TypeError(f"token must be str, float or Velocity, but got {type(token)}")

    def decode(self, token_id: int) -> str | Position:
        if token_id < self.num_actions:
            return self._id2action(token_id)
        elif token_id < self.num_normal_tokens:
            return self._id2position(token_id)
        elif token_id < self.num_normal_tokens + len(self.special_tokens):
            return self.special_tokens[token_id - self.num_normal_tokens]
        else:
            raise ValueError(f"token_id must be less than {self.num_normal_tokens + len(self.special_tokens)}")

    def _action2id(self, action: str) -> int:
        return self.action2id[action]

    def _id2action(self, id_: int) -> str:
        return self.id2action[id_]

    def _coordinate2id(self, coordinate: float) -> int:
        # range is [-FIELD_LENGTH, FIELD_LENGTH]
        id_ = int(
            (int(coordinate) + FIELD_LENGTH) / self.position_granularity
        )  # [0, 2 * FIELD_LENGTH / position_granularity]
        assert 0 <= id_ <= self.num_positions, f"coordinate: {coordinate}, id: {id_}"
        return id_ + self.num_actions

    def _velocity2action(self, velocity: Velocity) -> str:
        return discretize_direction(velocity.x, velocity.y)


@StateActionTokenizerBase.register("action_only")
class ActionOnlyTokenizer(StateActionTokenizerBase):
    def __init__(
        self,
        action2id: Dict[str, int],
        origin_pos: str = "center",
        special_tokens: List[str] = ["[SEP]", "[PAD]", "[ACTION_SEP]"],
    ) -> None:
        super().__init__(action2id, position_granularity=0.5, origin_pos=origin_pos)
        self.action2id = action2id
        self.id2action = {id_: action for action, id_ in action2id.items()}
        self.origin_pos = origin_pos
        self.num_actions = len(action2id)
        self.num_normal_tokens = self.num_actions
        self.special_tokens = special_tokens
        self.num_tokens = self.num_normal_tokens + len(self.special_tokens)

    def encode_position_separately(self) -> bool:
        return False

    def encode(self, token: str) -> int:
        # if str, then token is action -> return action_id
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.num_normal_tokens + self.special_tokens.index(token)
            else:
                return self._action2id(token)
        else:
            raise TypeError(f"token must be str, but got {type(token)}")

    def decode(self, token_id: int) -> str | Position:
        if token_id < self.num_actions:
            return self._id2action(token_id)
        elif token_id < self.num_normal_tokens + len(self.special_tokens):
            return self.special_tokens[token_id - self.num_normal_tokens]
        else:
            raise ValueError(f"token_id must be less than {self.num_normal_tokens + len(self.special_tokens)}")

    def _action2id(self, action: str) -> int:
        return self.action2id[action]
