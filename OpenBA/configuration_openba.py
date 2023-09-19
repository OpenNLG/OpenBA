from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


class OpenBAConfig(PretrainedConfig):
    model_type = "openba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers"
    }

    def __init__(
        self,
        vocab_size=32128,
        hidden_size=512,
        kv_channels=64,
        ffn_hidden_size=2048,
        num_layers=12,
        num_decoder_layers=None,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        num_heads=8,
        is_encoder_decoder=True,
        use_cache=True,
        initializer_factor=1.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        add_qkv_bias=False,
        add_ffn_bias=False,
        add_lm_head_bias=False,
        max_seq_length=1024,
        decoder_max_seq_length=256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.kv_channels = kv_channels
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_factor = initializer_factor
        self.num_heads = num_heads
        self.add_qkv_bias = add_qkv_bias
        self.add_ffn_bias = add_ffn_bias
        self.add_lm_head_bias = add_lm_head_bias
        self.max_seq_length = max_seq_length
        self.decoder_max_seq_length = decoder_max_seq_length
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )