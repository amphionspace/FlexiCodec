import torch

from flexicodec.ar_tts.utils.transformer.activation import Swish
from flexicodec.ar_tts.utils.transformer.subsampling import (
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    IdentityNoSubsampling,
)
from flexicodec.ar_tts.utils.transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
)
from flexicodec.ar_tts.utils.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from flexicodec.ar_tts.utils.transformer.embedding import EspnetRelPositionalEncoding
from flexicodec.ar_tts.utils.transformer.subsampling import LegacyLinearNoSubsampling


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "silu": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    "identity": IdentityNoSubsampling,
    "paraformer_dummy": torch.nn.Identity,
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
