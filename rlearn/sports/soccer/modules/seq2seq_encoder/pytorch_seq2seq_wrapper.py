from typing import cast

import torch
from torch import nn

from rlearn.sports.soccer.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder


class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        try:
            if not module.batch_first:
                raise ValueError("PytorchSeq2SeqWrapper only supports batch_first=True")
        except AttributeError:
            pass
        super().__init__()
        self._module = module
        try:
            self._is_bidirectional = cast(bool, self._module.bidirectional)
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return cast(int, self._module.input_size)

    def get_output_dim(self) -> int:
        return cast(int, self._module.hidden_size) * self._num_directions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self._module(inputs)
        return cast(torch.Tensor, output)


@Seq2SeqEncoder.register("gru")
class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "gru".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.GRU(  # type: ignore
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@Seq2SeqEncoder.register("lstm")
class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@Seq2SeqEncoder.register("rnn")
class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "rnn".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


class PytorchSeq2SeqWrapper_MLP(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        try:
            if not module.batch_first:
                raise ValueError("PytorchSeq2SeqWrapper only supports batch_first=True")
        except AttributeError:
            pass
        super().__init__()
        self._module = module
        try:
            self._is_bidirectional = cast(bool, self._module.bidirectional)
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return cast(int, self._module.in_features)

    def get_output_dim(self) -> int:
        return cast(int, self._module.out_features) * self._num_directions

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self._module(inputs)
        return cast(torch.Tensor, output)


@Seq2SeqEncoder.register("mlp")
class MlpSeq2SeqEncoder(PytorchSeq2SeqWrapper_MLP):
    """
    Registered as a `Seq2SeqEncoder` with name "mlp".
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        module = nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias)
        super().__init__(module=module)
