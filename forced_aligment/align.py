# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from utils.data_prep import (
    add_t_start_end_to_utt_obj,
    get_batch_starts_ends,
    get_batch_variables,
    get_manifest_lines_batch,
    is_entry_in_all_lines,
    is_entry_in_any_lines,
)
from utils.make_ass_files import make_ass_files
from utils.make_ctm_files import make_ctm_files
from utils.make_output_manifest import write_manifest_out_line
from utils.viterbi_decoding import viterbi_decoding

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class CTMFileConfig:
    remove_blank_tokens: bool = False
    # minimum duration (in seconds) for timestamps in the CTM.If any line in the CTM has a
    # duration lower than this, it will be enlarged from the middle outwards until it
    # meets the minimum_timestamp_duration, or reaches the beginning or end of the audio file.
    # Note that this may cause timestamps to overlap.
    minimum_timestamp_duration: float = 0


@dataclass
class ASSFileConfig:
    fontsize: int = 20
    vertical_alignment: str = "center"
    # if resegment_text_to_fill_space is True, the ASS files will use new segments
    # such that each segment will not take up more than (approximately) max_lines_per_segment
    # when the ASS file is applied to a video
    resegment_text_to_fill_space: bool = False
    max_lines_per_segment: int = 2
    text_already_spoken_rgb: List[int] = field(default_factory=lambda: [49, 46, 61])  # dark gray
    text_being_spoken_rgb: List[int] = field(default_factory=lambda: [57, 171, 9])  # dark green
    text_not_yet_spoken_rgb: List[int] = field(default_factory=lambda: [194, 193, 199])  # light gray


@dataclass
class AlignmentConfig:
    # Required configs
    pretrained_name: Optional[str] = None
    model_path: Optional[str] = None
    manifest_filepath: Optional[str] = None
    output_dir: Optional[str] = None

    # General configs
    align_using_pred_text: bool = False
    transcribe_device: Optional[str] = None
    viterbi_device: Optional[str] = None
    batch_size: int = 1
    use_local_attention: bool = True
    additional_segment_grouping_separator: Optional[str] = None
    audio_filepath_parts_in_utt_id: int = 1

    # Buffered chunked streaming configs
    use_buffered_chunked_streaming: bool = False
    chunk_len_in_secs: float = 1.6
    total_buffer_in_secs: float = 4.0
    chunk_batch_size: int = 32

    # Cache aware streaming configs
    simulate_cache_aware_streaming: Optional[bool] = False

    # Output file configs
    save_output_file_formats: List[str] = field(default_factory=lambda: ["ctm", "ass"])
    ctm_file_config: CTMFileConfig = field(default_factory=lambda: CTMFileConfig())
    ass_file_config: ASSFileConfig = field(default_factory=lambda: ASSFileConfig())


@hydra_runner(config_name="AlignmentConfig", schema=AlignmentConfig)
def main(cfg: AlignmentConfig):

    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    # Validate config
    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None")

    if cfg.model_path is not None and cfg.pretrained_name is not None:
        raise ValueError("One of cfg.model_path and cfg.pretrained_name must be None")

    if cfg.manifest_filepath is None:
        raise ValueError("cfg.manifest_filepath must be specified")

    if cfg.output_dir is None:
        raise ValueError("cfg.output_dir must be specified")

    if cfg.batch_size < 1:
        raise ValueError("cfg.batch_size cannot be zero or a negative number")

    if cfg.additional_segment_grouping_separator == "" or cfg.additional_segment_grouping_separator == " ":
        raise ValueError("cfg.additional_grouping_separator cannot be empty string or space character")

    if cfg.ctm_file_config.minimum_timestamp_duration < 0:
        raise ValueError("cfg.minimum_timestamp_duration cannot be a negative number")

    if cfg.ass_file_config.vertical_alignment not in ["top", "center", "bottom"]:
        raise ValueError("cfg.ass_file_config.vertical_alignment must be one of 'top', 'center' or 'bottom'")

    for rgb_list in [
        cfg.ass_file_config.text_already_spoken_rgb,
        cfg.ass_file_config.text_already_spoken_rgb,
        cfg.ass_file_config.text_already_spoken_rgb,
    ]:
        if len(rgb_list) != 3:
            raise ValueError(
                "cfg.ass_file_config.text_already_spoken_rgb,"
                " cfg.ass_file_config.text_being_spoken_rgb,"
                " and cfg.ass_file_config.text_already_spoken_rgb all need to contain"
                " exactly 3 elements."
            )

    # Validate manifest contents
    if not is_entry_in_all_lines(cfg.manifest_filepath, "audio_filepath"):
        raise RuntimeError(
            "At least one line in cfg.manifest_filepath does not contain an 'audio_filepath' entry. "
            "All lines must contain an 'audio_filepath' entry."
        )

    if cfg.align_using_pred_text:
        if is_entry_in_any_lines(cfg.manifest_filepath, "pred_text"):
            raise RuntimeError(
                "Cannot specify cfg.align_using_pred_text=True when the manifest at cfg.manifest_filepath "
                "contains 'pred_text' entries. This is because the audio will be transcribed and may produce "
                "a different 'pred_text'. This may cause confusion."
            )
    else:
        if not is_entry_in_all_lines(cfg.manifest_filepath, "text"):
            raise RuntimeError(
                "At least one line in cfg.manifest_filepath does not contain a 'text' entry. "
                "NFA requires all lines to contain a 'text' entry when cfg.align_using_pred_text=False."
            )

    # init devices
    if cfg.transcribe_device is None:
        transcribe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        transcribe_device = torch.device(cfg.transcribe_device)
    logging.info(f"Device to be used for transcription step (`transcribe_device`) is {transcribe_device}")

    if cfg.viterbi_device is None:
        viterbi_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        viterbi_device = torch.device(cfg.viterbi_device)
    logging.info(f"Device to be used for viterbi step (`viterbi_device`) is {viterbi_device}")

    if transcribe_device.type == 'cuda' or viterbi_device.type == 'cuda':
        logging.warning(
            'One or both of transcribe_device and viterbi_device are GPUs. If you run into OOM errors '
            'it may help to change both devices to be the CPU.'
        )

    # load model
    model, _ = setup_model(cfg, transcribe_device)
    model.eval()

    if isinstance(model, EncDecHybridRNNTCTCModel):
        model.change_decoding_strategy(decoder_type="ctc")

    if cfg.use_local_attention:
        logging.info(
            "Flag use_local_attention is set to True => will try to use local attention for model if it allows it"
        )
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])

    if not (isinstance(model, EncDecCTCModel) or isinstance(model, EncDecHybridRNNTCTCModel)):
        raise NotImplementedError(
            f"Model is not an instance of NeMo EncDecCTCModel or ENCDecHybridRNNTCTCModel."
            " Currently only instances of these models are supported"
        )

    if cfg.ctm_file_config.minimum_timestamp_duration > 0:
        logging.warning(
            f"cfg.ctm_file_config.minimum_timestamp_duration has been set to {cfg.ctm_file_config.minimum_timestamp_duration} seconds. "
            "This may cause the alignments for some tokens/words/additional segments to be overlapping."
        )

    buffered_chunk_params = {}
    if cfg.use_buffered_chunked_streaming:
        model_cfg = copy.deepcopy(model._cfg)

        OmegaConf.set_struct(model_cfg.preprocessor, False)
        # some changes for streaming scenario
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0

        if model_cfg.preprocessor.normalize != "per_feature":
            logging.error(
                "Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently"
            )
        # Disable config overwriting
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        feature_stride = model_cfg.preprocessor['window_stride']
        model_stride_in_secs = feature_stride * cfg.model_downsample_factor
        total_buffer = cfg.total_buffer_in_secs
        chunk_len = float(cfg.chunk_len_in_secs)
        tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
        mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
        logging.info(f"tokens_per_chunk is {tokens_per_chunk}, mid_delay is {mid_delay}")

        model = FrameBatchASR(
            asr_model=model,
            frame_len=chunk_len,
            total_buffer=cfg.total_buffer_in_secs,
            batch_size=cfg.chunk_batch_size,
        )
        buffered_chunk_params = {
            "delay": mid_delay,
            "model_stride_in_secs": model_stride_in_secs,
            "tokens_per_chunk": tokens_per_chunk,
        }
    # get start and end line IDs of batches
    starts, ends = get_batch_starts_ends(cfg.manifest_filepath, cfg.batch_size)

    # init output_timestep_duration = None and we will calculate and update it during the first batch
    output_timestep_duration = None

    # init f_manifest_out
    os.makedirs(cfg.output_dir, exist_ok=True)
    tgt_manifest_name = str(Path(cfg.manifest_filepath).stem) + "_with_output_file_paths.json"
    tgt_manifest_filepath = str(Path(cfg.output_dir) / tgt_manifest_name)
    f_manifest_out = open(tgt_manifest_filepath, 'w')

    # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        manifest_lines_batch = get_manifest_lines_batch(cfg.manifest_filepath, start, end)

        (log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration,) = get_batch_variables(
            manifest_lines_batch,
            model,
            cfg.additional_segment_grouping_separator,
            cfg.align_using_pred_text,
            cfg.audio_filepath_parts_in_utt_id,
            output_timestep_duration,
            cfg.simulate_cache_aware_streaming,
            cfg.use_buffered_chunked_streaming,
            buffered_chunk_params,
        )

        alignments_batch = viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device)

        for utt_obj, alignment_utt in zip(utt_obj_batch, alignments_batch):

            utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)

            if "ctm" in cfg.save_output_file_formats:
                utt_obj = make_ctm_files(utt_obj, cfg.output_dir, cfg.ctm_file_config,)

            if "ass" in cfg.save_output_file_formats:
                utt_obj = make_ass_files(utt_obj, cfg.output_dir, cfg.ass_file_config)

            write_manifest_out_line(
                f_manifest_out, utt_obj,
            )

    f_manifest_out.close()

    return None


if __name__ == "__main__":
    main()
