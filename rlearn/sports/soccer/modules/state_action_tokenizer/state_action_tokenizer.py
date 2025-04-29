#Updated state_action_tokenizer.py for the position encoding granularity can be dynamically adjusted based on different environments (e.g., for non-rectangular fields).
from typing import Dict, List
import logging
from tango.common import Registrable
from rlearn.sports.soccer.constant import FIELD_LENGTH, FIELD_WIDTH
from rlearn.sports.soccer.dataclass import Position, Velocity
from rlearn.sports.soccer.modules.state_action_tokenizer.preprocess_frames import discretize_direction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateActionTokenizerBase(Registrable):
    def __init__(
        self, action2id: Dict[str, int], position_granularity: float = 0.5, origin_pos: str = "center"
    ) -> None:
        self.action2id = action2id
        self.position_granularity = position_granularity
        self.origin_pos = origin_pos
        self.num_actions = len(action2id)
    
    def encode(self, token: str | Position | Velocity) -> int:
        raise NotImplementedError

    def decode(self, token_id: int) -> str | Position:
        raise NotImplementedError

    def _action2id(self, action: str) -> int:
        return self.action2id.get(action, None)

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
        super().__init__(action2id, position_granularity, origin_pos)
        self.id2action = {id_: action for action, id_ in action2id.items()}
        self.position_granularity = position_granularity
        self.origin_pos = origin_pos
        self.num_positions = int(2 * FIELD_LENGTH / position_granularity) * int(2 * FIELD_WIDTH / position_granularity)
        self.special_tokens = special_tokens
        self.num_normal_tokens = self.num_actions + self.num_positions
        self.num_tokens = self.num_normal_tokens + len(self.special_tokens)

    def encode_position_separately(self) -> bool:
        return False

    def encode(self, token: str | Position | Velocity) -> int:
        # If str, token is action -> return action_id
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.num_normal_tokens + self.special_tokens.index(token)
            elif token in self.action2id:
                return self._action2id(token)
            else:
                logger.error(f"Action '{token}' not found in action2id mapping.")
                raise ValueError(f"Action '{token}' not found in action2id mapping.")
        elif isinstance(token, Position):
            return self._position2id(token)
        elif isinstance(token, Velocity):
            return self._action2id(self._velocity2action(token))
        else:
            logger.error(f"Invalid token type: {type(token)}. Expected str, Position, or Velocity.")
            raise TypeError(f"token must be str, Position, or Velocity, but got {type(token)}")

    def decode(self, token_id: int) -> str | Position:
        if token_id < self.num_actions:
            return self._id2action(token_id)
        elif token_id < self.num_normal_tokens:
            return self._id2position(token_id)
        elif token_id < self.num_normal_tokens + len(self.special_tokens):
            return self.special_tokens[token_id - self.num_normal_tokens]
        else:
            logger.error(f"Invalid token ID: {token_id}. Must be less than {self.num_normal_tokens + len(self.special_tokens)}.")
            raise ValueError(f"Invalid token ID: {token_id}. Must be less than {self.num_normal_tokens + len(self.special_tokens)}.")

    def _action2id(self, action: str) -> int:
        action_id = self.action2id.get(action)
        if action_id is None:
            logger.error(f"Action '{action}' is not present in action2id mapping.")
            raise ValueError(f"Action '{action}' is not found in action2id mapping.")
        return action_id

    def _position2id(self, position: Position) -> int:
        # Validate the position ranges
        x = int((position.x + FIELD_LENGTH) / self.position_granularity)
        y = int((position.y + FIELD_WIDTH) / self.position_granularity)
        assert 0 <= x <= int(2 * FIELD_LENGTH / self.position_granularity), f"Invalid x: {position.x} for position: {position}"
        assert 0 <= y <= int(2 * FIELD_WIDTH / self.position_granularity), f"Invalid y: {position.y} for position: {position}"
        return (x * int(2 * FIELD_WIDTH / self.position_granularity) + y + self.num_actions)

    def _velocity2action(self, velocity: Velocity) -> str:
        return discretize_direction(velocity.x, velocity.y)


@StateActionTokenizerBase.register("action_only")
class ActionOnlyTokenizer(StateActionTokenizerBase):
    def __init__(
        self,
        action2id: Dict[str, int],
        special_tokens: List[str] = ["[SEP]", "[PAD]", "[ACTION_SEP]"],
    ) -> None:
        super().__init__(action2id, position_granularity=0.5)
        self.id2action = {id_: action for action, id_ in action2id.items()}
        self.special_tokens = special_tokens
        self.num_actions = len(action2id)
        self.num_tokens = self.num_actions + len(self.special_tokens)

    def encode_position_separately(self) -> bool:
        return False

    def encode(self, token: str) -> int:
        if isinstance(token, str):
            if token in self.special_tokens:
                return self.num_actions + self.special_tokens.index(token)
            return self._action2id(token)
        logger.error(f"Expected a string token, but got {type(token)}.")
        raise TypeError(f"Expected a string token, but got {type(token)}.")

    def decode(self, token_id: int) -> str:
        if token_id < self.num_actions:
            return self._id2action(token_id)
        elif token_id < self.num_actions + len(self.special_tokens):
            return self.special_tokens[token_id - self.num_actions]
        logger.error(f"Invalid token ID: {token_id}. It should be less than {self.num_actions + len(self.special_tokens)}.")
        raise ValueError(f"Invalid token ID: {token_id}. It should be less than {self.num_actions + len(self.special_tokens)}.")

    def _action2id(self, action: str) -> int:
        return self.action2id.get(action)

    def _id2action(self, id_: int) -> str:
        return self.id2action.get(id_)

# Example usage of the tokenizer
if __name__ == "__main__":
    action2id = {
        "MOVE_UP": 0,
        "MOVE_DOWN": 1,
        "MOVE_LEFT": 2,
        "MOVE_RIGHT": 3,
        "SHOOT": 4,
    }

    tokenizer = SimpleStateActionTokenizer(action2id=action2id)

    # Example for encoding action
    token_id = tokenizer.encode("MOVE_UP")
    print(f"Token ID for 'MOVE_UP': {token_id}")

    # Example for decoding token ID
    decoded_action = tokenizer.decode(token_id)
    print(f"Decoded action: {decoded_action}")
