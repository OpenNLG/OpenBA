from typing import Optional, Tuple, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from transformers.utils import logging, is_torch_fx_proxy

from .configuration_openbt5 import OpenBT5Config


logger = logging.get_logger(__name__)

# Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


def rotate_half(x) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.cat((sin, sin), dim=-1).to(tensor.device)[:, :, None, :]
    cos = torch.cat((cos, cos), dim=-1).to(tensor.device)[:, :, None, :]
    return (tensor * cos) + (rotate_half(tensor) * sin)


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        hidden_size = config.hidden_size
        ffn_hidden_size = int(2 * config.ffn_hidden_size / 3) 
        ffn_hidden_size = multiple_of * ((ffn_hidden_size + multiple_of - 1) // multiple_of)
        self.ffn_hidden_size = ffn_hidden_size

        self.fc_in = nn.Linear(hidden_size, 2 * ffn_hidden_size, bias=config.add_ffn_bias)
        self.fc_out = nn.Linear(ffn_hidden_size, hidden_size, bias=config.add_ffn_bias)
        
        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        self.act_func = swiglu

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act_func(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states


class OpenBT5Attention(nn.Module):
    def __init__(self, config, attn_type='self'):
        super().__init__()
        self.attn_type = attn_type
        self.is_decoder = config.is_decoder
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.kv_channels = config.kv_channels
        self.proj_size = self.kv_channels * self.num_heads
        self.dropout = config.attention_dropout
        self.scale_attn = torch.sqrt(torch.tensor(self.kv_channels, dtype=torch.float32))

        if self.attn_type == 'self':
            self.qkv = nn.Linear(self.hidden_size, 3 * self.proj_size, bias=config.add_qkv_bias)
        else:
            assert self.attn_type == 'cross'
            self.q = nn.Linear(self.hidden_size, self.proj_size, bias=config.add_qkv_bias)
            self.kv = nn.Linear(self.hidden_size, 2 * self.proj_size, bias=config.add_qkv_bias)

        self.rotary_embedding = create_sinusoidal_positions(
            num_pos=config.max_seq_length,
            dim=self.kv_channels,
        )

        self.o = nn.Linear(self.proj_size, self.hidden_size, bias=config.add_qkv_bias)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        key_value_states: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[Tuple[torch.Tensor]] = None,
        position_ids:Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # input is (batch_size, seq_length, hidden_size)
        batch_size, seq_length = hidden_states.shape[:2]
        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )

        if self.rotary_embedding.device != position_ids.device:
            self.rotary_embedding = self.rotary_embedding.to(position_ids.device)

        if self.attn_type == 'self':
            mixed_qkv_states = self.qkv(hidden_states)
            new_tensor_shape = mixed_qkv_states.size()[:-1] + (self.num_heads, 3 * self.kv_channels)
            mixed_qkv_states = mixed_qkv_states.view(*new_tensor_shape)
            query_states, key_states, value_states = torch.chunk(mixed_qkv_states, 3, dim=-1)
            # rotary position embedding
            sincos = self.rotary_embedding[position_ids]
            sin, cos = torch.chunk(sincos, 2, dim=-1)
            query_states = apply_rotary_pos_emb(query_states, sin, cos)
            key_states = apply_rotary_pos_emb(key_states, sin, cos)
            # reshape to (batch_size, num_head, seq_length, kv_channels)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            if past_key_value is not None:
                past_key_states, past_value_states = past_key_value
                key_states = torch.cat([past_key_states, key_states], dim=-2)
                value_states = torch.cat([past_value_states, value_states], dim=-2)
        else:
            assert self.attn_type == 'cross'
            query_states = self.q(hidden_states)
            new_tensor_shape = query_states.size()[:-1] + (self.num_heads, self.kv_channels)
            query_states = query_states.view(*new_tensor_shape)
            # reshape to (batch_size, num_head, seq_length, kv_channels)
            query_states = query_states.transpose(1, 2)
            if past_key_value is None:
                mixed_kv_states = self.kv(key_value_states)
                new_tensor_shape = mixed_kv_states.size()[:-1] + (self.num_heads, 2 * self.kv_channels)
                mixed_kv_states = mixed_kv_states.view(*new_tensor_shape)
                key_states, value_states = torch.chunk(mixed_kv_states, 2, dim=-1)
                # reshape to (batch_size, num_head, seq_length, kv_channels)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
            else:
                key_states, value_states = past_key_value

        # compute attention score
        query_states = query_states.to(torch.float32)
        key_states = key_states.to(torch.float32)
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / self.scale_attn
        attn_scores = attn_scores.masked_fill_(attention_mask, -10000.0)
        attn_weights = F.softmax(attn_scores, dim=-1).type_as(attn_scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_weights = attn_weights.to(value_states.dtype)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.proj_size)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output, present_key_value_state)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class OpenBT5Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.is_decoder = config.is_decoder
        self.dropout = config.hidden_dropout
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = OpenBT5Attention(config, attn_type='self')
        self.post_attn_layernorm = nn.LayerNorm(config.hidden_size)
        if self.is_decoder:
            self.inter_attn = OpenBT5Attention(config, attn_type='cross')
            self.post_inter_attn_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                raise ValueError("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attn_outputs = self.self_attn(
            layernorm_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output, present_key_value_state = attn_outputs[:2]
        attn_weights = attn_outputs[2:]
        residual = hidden_states
        # Layer norm post the self attention.
        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        layernorm_input = residual + attn_output
        layernorm_output = self.post_attn_layernorm(layernorm_input)

        if self.is_decoder:
            assert encoder_hidden_states is not None
            attn_outputs = self.inter_attn(
                layernorm_output,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                position_ids=position_ids,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
            attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
            # residual connection
            residual = layernorm_input
            layernorm_input = residual + attn_output
            layernorm_output = self.post_inter_attn_layernorm(layernorm_input)
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state += attn_outputs[1]
            attn_weights += attn_outputs[2:]

        # MLP.
        mlp_output = self.mlp(layernorm_output)
        mlp_output = nn.functional.dropout(mlp_output, p=self.dropout, training=self.training)
        # Second residual connection.
        residual = layernorm_input
        output = residual + mlp_output
        outputs = (output,)

        if use_cache:
            outputs += (present_key_value_state,) + attn_weights
        else:
            outputs += attn_weights
        return outputs


class OpenBT5PreTrainedModel(PreTrainedModel):
    config_class = OpenBT5Config
    base_model_prefix = "transformer"
    _no_split_modules = ["OpenBT5Block"]

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OpenBT5Attention, OpenBT5Stack)):
            module.gradient_checkpointing = value

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, OpenBT5ForConditionalGeneration):
            module.shared_embedding.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, SwiGLUMLP):
            module.fc_in.weight.data.normal_(mean=0.0, std=factor * ((self.config.hidden_size) ** -0.5))
            if hasattr(module.fc_in, "bias") and module.fc_in.bias is not None:
                module.fc_in.bias.data.zero_()
            module.fc_out.weight.data.normal_(mean=0.0, std=factor * ((module.ffn_hidden_size) ** -0.5))
            if hasattr(module.fc_out, "bias") and module.fc_out.bias is not None:
                module.fc_out.bias.data.zero_()
        elif isinstance(module, OpenBT5Attention):
            hidden_size = self.config.hidden_size
            kv_channels = self.config.kv_channels
            n_heads = self.config.num_heads
            if module.attn_type == 'self':
                module.qkv.weight.data[:n_heads * kv_channels].normal_(mean=0.0, std=factor * ((hidden_size * kv_channels) ** -0.5))
                module.qkv.weight.data[n_heads * kv_channels:].normal_(mean=0.0, std=factor * (hidden_size ** -0.5))
            else:
                module.q.weight.data.normal_(mean=0.0, std=factor * ((hidden_size * kv_channels) ** -0.5))
                module.kv.weight.data.normal_(mean=0.0, std=factor * (hidden_size ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * kv_channels) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

class OpenBT5Stack(OpenBT5PreTrainedModel):
    def __init__(self, config, embed_tokens):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList(
            [OpenBT5Block(config) for _ in range(config.num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get batch size and seq_length
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # required mask seq length can be calculated via length of past
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.block)
        else:
            past_length = past_key_values[0][0].size(-2)
        cur_length = past_length + seq_length

        # position ids
        position_ids = torch.arange(past_length, cur_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device)
        # get extended self-attention mask
        if self.is_decoder:
            if len(attention_mask.shape) == 2:
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            elif len(attention_mask.shape) == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask < 0.5
        # get extended self-attention mask
        # here we replace encoder_decoder_attention_mask with encoder_attention_mask
        if self.is_decoder and encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_seq_length = encoder_hidden_states.shape[1]
                encoder_attention_mask = torch.ones(
                    batch_size, encoder_seq_length, device=device, dtype=torch.long
                )
            extended_encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            extended_encoder_attention_mask = extended_encoder_attention_mask < 0.5
        else:
            extended_encoder_attention_mask = None
        

        # input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            if use_cache:
                present_key_value_states += (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[3],)

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class OpenBT5ForConditionalGeneration(OpenBT5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]
    def __init__(self, config):
        super().__init__(config)
        self.shared_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hidden_size = config.hidden_size

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = OpenBT5Stack(encoder_config, self.shared_embedding)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.max_seq_length = config.decoder_max_seq_length
        self.decoder = OpenBT5Stack(decoder_config, self.shared_embedding)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.add_lm_head_bias)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared_embedding

    def set_input_embeddings(self, new_embeddings):
        self.shared_embedding = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,\
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        # share embedding and softmax embedding
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.hidden_size ** -0.5)

        lm_logits = self.lm_head(sequence_output).to(torch.float32)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
