# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

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

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


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
    compiler_cfg.enable_t_streaming = True
    compiler_cfg.enable_tvm_cpu_fallback = False  # Run full model on silicon
    compiler_cfg.input_queues_on_host = True
    compiler_cfg.compile_subgraphs = True
    compiler_cfg.enable_link_past_cache_ios = True
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.amp_level = 0
            compiler_cfg.backend_opt_level = 3
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "28672"  # 28 * 1024
            os.environ["PYBUDA_DISABLE_STREAM_OUTPUT"] = "1"  # Disable streaming for LM head to output queue (perf)
            os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
            os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
            os.environ["TT_BACKEND_PROFILER"] = "1"
            os.environ["PYBUDA_NOP_ON_DIRECT_SHORT_PATH"] = "1"

    os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
    os.environ["PYBUDA_PAD_OUTPUT_BUFFER_THRESHOLD_TILES"] = "1536"

    os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"

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
        os.environ[
            "PYBUDA_PAD_PARAMETER"
        ] = f"model.model.encoder.embed_positions.weight, {padded_len}, {config.d_model}"

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
