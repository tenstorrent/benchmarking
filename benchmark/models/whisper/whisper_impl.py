# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional, Tuple, Union

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LogitsProcessorList,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)


class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores
    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.
        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, "str"] = "cpu",
    ) -> torch.Tensor:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        key_value_length: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        expanded_4d_mask = expanded_attn_mask if causal_4d_mask is None else expanded_attn_mask + causal_4d_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`
    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


class Whisper_encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model.model.encoder(input_features=input_features)


class Whisper_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds, *past_key_values
    ):
        presents = []
        pkv = []

        input_embeds = self.model.model.decoder.embed_tokens(decoder_input_ids)
        hidden_states = input_embeds + position_embeds

        attention_mask = _prepare_4d_causal_attention_mask(
            decoder_attention_mask, decoder_input_ids.size(), input_embeds, past_key_values[0].shape[2]
        )

        presents = []
        for i, decoder_layer in enumerate(self.model.model.decoder.layers):
            pkv = tuple([past_key_values[(i * 4) + j] for j in range(4)])

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_last_hidden_state,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=pkv,
                output_attentions=False,
                use_cache=True,
            )
            hidden_states = layer_outputs[0]
            presents.append(layer_outputs[1])

        hidden_states = self.model.model.decoder.layer_norm(hidden_states)
        lm_logits = self.model.proj_out(hidden_states)

        return lm_logits, *presents


# check the name later # enc-dec
def generate_model_whisper_enc_dec(variant):
    import pybuda
    from pybuda._C.backend_api import BackendDevice
    from pybuda.config import _get_global_compiler_config
    from pybuda.pybudaglobal import TILE_DIM

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.amp_level = 1
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER_THRESHOLD_TILES"] = "1536"

    if variant == "openai/whisper-base":
        os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "None"
        compiler_cfg.enable_auto_fusing = False

    # pybuda.set_configuration_options(performance_trace=pybuda.PerfTraceLevel.VERBOSE)
    processor = AutoProcessor.from_pretrained(variant)
    config = WhisperConfig.from_pretrained(variant)
    config.return_dict = False

    pad_model = True
    padded_len = 1536
    if pad_model:
        config.max_source_positions = padded_len
    else:
        os.environ["PYBUDA_PAD_PARAMETER"] = (
            f"model.model.encoder.embed_positions.weight, {padded_len}, {config.d_model}"
        )

    max_length = config.max_length
    model = WhisperForConditionalGeneration.from_pretrained(
        variant,
        ignore_mismatched_sizes=True,
        config=config,
    )
    if pad_model:
        unpadded_model = WhisperForConditionalGeneration.from_pretrained(variant)
        padded_param = torch.nn.functional.pad(unpadded_model.model.encoder.embed_positions.weight.data, (0, 0, 0, 36))
        model.model.encoder.embed_positions.weight.data = padded_param

    feature_extractor = WhisperFeatureExtractor.from_pretrained(variant)
    tokenizer = WhisperTokenizer.from_pretrained(variant)
    encoder_module = pybuda.PyTorchModule("Whisper_encoder", Whisper_encoder(model))
    decoder_module_cross_attention = pybuda.PyTorchModule("Whisper_decoder_with_ca", Whisper_decoder(model))
    decoder_module_no_cross_attention = pybuda.PyTorchModule("Whisper_decoder_no_ca", Whisper_decoder(model))

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    smaller_dataset = []
    if True:
        for i in range(10):
            smaller_dataset.append(ds[i])
        ds = smaller_dataset

    sample_rate = 16000
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt", sampling_rate=sample_rate)

    if pad_model:
        input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
    else:
        input_features = inputs.input_features

    encoder_last_hidden_state_shape = (1, padded_len, config.d_model)
    encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)

    logits_processor = model._get_logits_processor(
        model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList()
    )
    decoder_attention_mask = torch.zeros((1, max_length))
    decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
    first_current_index = max_length - TILE_DIM
    position_embeds = torch.zeros((TILE_DIM, config.d_model))
    enc_past_cache_self_shape = (
        1,
        config.decoder_attention_heads,
        max_length - TILE_DIM,
        config.d_model // config.decoder_attention_heads,
    )
    enc_past_cache_cross_shape = (1, 1, 1, 1)

    decoder_with_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_with_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]

    enc_past_cache_cross_shape = (
        1,
        config.decoder_attention_heads,
        padded_len,
        config.d_model // config.decoder_attention_heads,
    )
    decoder_no_ca_inputs = [decoder_input_ids, decoder_attention_mask, encoder_last_hidden_state, position_embeds]
    for _ in range(config.decoder_layers):
        decoder_no_ca_inputs += [
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_self_shape),
            torch.zeros(enc_past_cache_cross_shape),
            torch.zeros(enc_past_cache_cross_shape),
        ]

    # outputs
    compile_inputs = (
        (input_features,),
        (decoder_with_ca_inputs),
        (decoder_no_ca_inputs),
    )
    logits_processor = model._get_logits_processor(
        model.generation_config, TILE_DIM, input_features, None, LogitsProcessorList()
    )
    modules = (
        encoder_module,
        decoder_module_cross_attention,
        decoder_module_no_cross_attention,
    )
    return (
        modules,
        {
            "compile_inputs": compile_inputs,
            "logits_processor": logits_processor,
            "max_length": max_length,
            "first_current_index": first_current_index,
            "processor": processor,
            "embed_positions_weight": model.model.decoder.embed_positions.weight,
        },
    )
