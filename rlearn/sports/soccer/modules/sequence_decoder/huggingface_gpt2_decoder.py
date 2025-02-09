import logging

import torch
from transformers import GPT2Config, GPT2Model

from rlearn.sports.soccer.modules.sequence_decoder.sequence_decoder import SequenceDecoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@SequenceDecoder.register("huggingface_gpt2_decoder")
class HuggingfaceGPT2Decoder(SequenceDecoder):
    """
    Huggingface GPT2 Decoder

    Args:
        input_dim: input dimension
        num_layers: number of layers
        feedforward_hidden_dim: feedforward hidden dimension
        num_attention_head: number of attention heads
        activation: activation function
        n_positions: number of positions
        pretrained_model_name_or_path: pretrained model name or path
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int,
        num_attention_head: int,
        activation: str = "gelu_new",
        n_positions: int = 8192,
        pretrained_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        if pretrained_model_name_or_path is not None:
            logger.info(f"Loading pretrained model from {pretrained_model_name_or_path}")
            self.model = GPT2Model.from_pretrained(pretrained_model_name_or_path)
            self.config = self.model.config
            self.input_dim = self.config.n_embd
            self.num_layers = self.config.n_layer
            self.feedforward_hidden_dim = self.config.n_inner
            self.num_attention_head = self.config.n_head
            self.activation = self.config.activation_function
            self.n_positions = self.config.n_positions
        else:
            logger.info("Building model from scratch")
            self.config = GPT2Config(
                n_positions=n_positions,
                n_embd=input_dim,
                n_layer=num_layers,
                n_head=num_attention_head,
                n_inner=feedforward_hidden_dim,
                activation_function=activation,
                output_hidden_states=True,
            )
            self.model = GPT2Model(self.config)
            self.input_dim = input_dim
            self.num_layers = num_layers
            self.feedforward_hidden_dim = feedforward_hidden_dim
            self.num_attention_head = num_attention_head
            self.activation = activation
            self.n_positions = n_positions

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.input_dim

    def is_bidirectional(self) -> bool:
        return False

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, input_dim)
        """
        output = self.model(inputs_embeds=inputs, attention_mask=mask)
        return output.last_hidden_state
