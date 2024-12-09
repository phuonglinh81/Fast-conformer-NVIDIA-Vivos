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

"""
Align the utterances in manifest_filepath. 
Results are saved in ctm files in output_dir.

Arguments:
    pretrained_name: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded
        from NGC and used for generating the log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
    model_path: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the
        log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
        Note: if a model_path is provided, it will override the pretrained_name.
    manifest_filepath: filepath to the manifest of the data you want to align,
        containing 'audio_filepath' and 'text' fields.
    output_dir: the folder where output CTM files and new JSON manifest will be saved.
    align_using_pred_text: if True, will transcribe the audio using the specified model and then use that transcription 
        as the reference text for the forced alignment. 
    transcribe_device: None, or a string specifying the device that will be used for generating log-probs (i.e. "transcribing").
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    viterbi_device: None, or string specifying the device that will be used for doing Viterbi decoding. 
        The string needs to be in a format recognized by torch.device(). If None, NFA will set it to 'cuda' if it is available 
        (otherwise will set it to 'cpu').
    batch_size: int specifying batch size that will be used for generating log-probs and doing Viterbi decoding.
    use_local_attention: boolean flag specifying whether to try to use local attention for the ASR Model (will only
        work if the ASR Model is a Conformer model). If local attention is used, we will set the local attention context 
        size to [64,64].
    additional_segment_grouping_separator: an optional string used to separate the text into smaller segments. 
        If this is not specified, then the whole text will be treated as a single segment. 
    remove_blank_tokens_from_ctm:  a boolean denoting whether to remove <blank> tokens from token-level output CTMs. 
    audio_filepath_parts_in_utt_id: int specifying how many of the 'parts' of the audio_filepath
        we will use (starting from the final part of the audio_filepath) to determine the 
        utt_id that will be used in the CTM files. Note also that any spaces that are present in the audio_filepath 
        will be replaced with dashes, so as not to change the number of space-separated elements in the 
        CTM files.
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 1 => utt_id will be "e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 2 => utt_id will be "d_e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and audio_filepath_parts_in_utt_id is 3 => utt_id will be "c_d_e1"
    use_buffered_infer: False, if set True, using streaming to do get the logits for alignment
                        This flag is useful when aligning large audio file.
                        However, currently the chunk streaming inference does not support batch inference,
                        which means even you set batch_size > 1, it will only infer one by one instead of doing
                        the whole batch inference together.
    chunk_len_in_secs: float chunk length in seconds
    total_buffer_in_secs: float  Length of buffer (chunk + left and right padding) in seconds
    chunk_batch_size: int batch size for buffered chunk inference,
                      which will cut one audio into segments and do inference on chunk_batch_size segments at a time

    simulate_cache_aware_streaming: False, if set True, using cache aware streaming to do get the logits for alignment

    save_output_file_formats: List of strings specifying what type of output files to save (default: ["ctm", "ass"])
    ctm_file_config: CTMFileConfig to specify the configuration of the output CTM files
    ass_file_config: ASSFileConfig to specify the configuration of the output ASS files
"""
import json
import os
from termcolor import colored
from typing import List, Optional, Tuple, Union

from IPython.display import Audio, HTML, Image, display
import numpy as np
import texterrors

def get_detailed_wer_labels(ref: List[str], hyp: List[str], return_eps_padded_hyp: bool = False):
    """Get detailed WER labels, aligning reference with hypothesis.

    Possible WER labels:
        - 'C' for Correct,
        - 'I' for Insertion,
        - 'D' for Deletion,
        - 'S' for Substitution.

    Returns:
        WER labels list.
        [Optional] Epsilin-padded hypothesis if return_eps_padded_hyp set to True.
    """

    # Align reference and hypothesis using "<eps>"
    aligned_ref, aligned_hyp = texterrors.align_texts(ref, hyp, False)[:-1]

    # Determine labels
    labels = []
    for r, h in zip(aligned_ref, aligned_hyp):
        if r == h:
            labels.append("C")
        elif r == "<eps>":
            labels.append("I")
        elif h == "<eps>":
            labels.append("D")
        else:
            labels.append("S")

    return labels if not return_eps_padded_hyp else labels, aligned_hyp


def fill_confidence_deletions(confidence_scores: List[float], labels: List[str], fill_value: float = 0.0):
    """Fill confidence scores list with the provided value for deletions.
    Assumes that we have no natural confidence scores for deletions.

    Returns:
        Confidence scores list with deletion scores.
    """

    assert len(confidence_scores) <= len(labels)

    # If the lengths of confidence_scores and labels are equal, then we assume that there are no deletions
    if len(confidence_scores) == len(labels):
        return confidence_scores

    # Insert fill_value into confidence_scores where label == "D"
    new_confidence_scores = []
    score_index = 0
    for label in labels:
        if label == "D":
            new_confidence_scores.append(fill_value)
        else:
            new_confidence_scores.append(confidence_scores[score_index])
            score_index += 1
    return new_confidence_scores


def pretty_pad_word_labels(labels: List[str], words: List[str]):
    """Pad word labels with dash for pretty printing.
    Expects labels and words to have the same length.

    Returns:
        Padded labels list.
    """

    # Check that words and labels without 'D' have the same length
    assert len(words) == len(labels)

    # Pad the labels with dashes to align them with the words
    padded_labels = []
    for word, label in zip(words, labels):
        label_len = len(word)
        left_padding = (label_len - 1) // 2
        right_padding = label_len - left_padding - 1
        padded_label = "-" * left_padding + label + "-" * right_padding
        padded_labels.append(padded_label)

    return padded_labels


def _html_paint_word_grey(word: str, shade: str):
    if shade == "black":
        color = "0,0,0"
    elif shade == "grey":
        color = "150,150,150"
    elif shade == "light_grey":
        color = "200,200,200"
    else:
        raise ValueError(
            f"`shade` has to be one of the following: `black`, `grey`, `light_grey`. Provided: `{shade}`"
        )
    return f'<mark style="color:rgb({color});background-color:rgb(255,255,255);">{word}</font></mark>'


def pretty_print_transcript_with_confidence(
    transcript: str,
    confidence_scores: List[float],
    threshold: float,
    reference: Optional[str] = None,
    terminal_width: int = 120,
    html: bool = False,
):
    if html:
        shade_if_low_confidence = lambda x, y: _html_paint_word_grey(x, 'light_grey' if y < threshold else 'black')
        new_line_mark = "<br>"
        pretty_print = lambda x: display(HTML("<code>" + new_line_mark.join(x) + "</code>"))
    else:
        shade_if_low_confidence = lambda x, y: colored(x, 'light_grey') if y < threshold else x
        new_line_mark = "\n"
        pretty_print = lambda x: print(new_line_mark.join(x))
    with_labels = reference is not None
    transcript_list = transcript.split()
    output_lines = []
    if with_labels:
        reference_list = reference.split()
        labels, eps_padded_hyp = get_detailed_wer_labels(reference_list, transcript_list, True)
        padded_labels = pretty_pad_word_labels(labels, eps_padded_hyp)
        current_line_len = 0
        current_word_line = ""
        current_label_line = ""
        for word, label, padded_label, score in zip(
            eps_padded_hyp, labels, padded_labels, fill_confidence_deletions(confidence_scores, labels)
        ):
            word_len = len(word)
            # shield angle brackets for <eps>
            if html and word == "<eps>":
                word = "&lt;eps&gt;"
            if current_line_len + word_len + 1 <= terminal_width:
                if current_line_len > 0:
                    current_line_len += 1
                    current_word_line += " "
                    current_label_line += "-"
                current_line_len += word_len
                current_word_line += shade_if_low_confidence(word, score)
                current_label_line += padded_label
            else:
                output_lines.append(current_word_line + new_line_mark + current_label_line)
                current_line_len = word_len
                current_word_line = shade_if_low_confidence(word, score)
                current_label_line = padded_label
        if current_word_line:
            output_lines.append(current_word_line + new_line_mark + current_label_line)
    else:
        current_line_len = 0
        current_word_line = ""
        for word, score in zip(transcript_list, confidence_scores):
            word_len = len(word)
            # shield angle brackets for <eps>
            if html and word == "<eps>":
                word = "&lt;eps&gt;"
            if current_line_len + word_len + 1 <= terminal_width:
                if current_line_len > 0:
                    current_line_len += 1
                    current_word_line += " "
                current_line_len += word_len
                current_word_line += shade_if_low_confidence(word, score)
            else:
                output_lines.append(current_word_line)
                current_line_len = word_len
                current_word_line = shade_if_low_confidence(word, score)
        if current_word_line:
            output_lines.append(current_word_line)

    pretty_print(output_lines)

from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecCTCModel
from dataclasses import dataclass
import json
from typing import List

# Hàm tải mô hình từ tệp .nemo
def load_model_from_nemo(model_path: str):
    """Load a pre-trained model from a .nemo file.

    Args:
        model_path: Path to the .nemo file.

    Returns:
        A model loaded into GPU with .eval() mode set.
    """
    model = EncDecCTCModel.restore_from(model_path)  # Tải mô hình từ tệp .nemo
    model.eval()  # Đặt mô hình vào chế độ evaluation (không huấn luyện)
    return model

@dataclass
class TestSet:
    filepaths: List[str]
    reference_texts: List[str]
    durations: List[float]

def load_data(manifest_path: str):
    filepaths = []
    reference_texts = []
    durations = []
    with open(manifest_path, "r") as f:
        for line in f:
            item = json.loads(line)
            audio_file = item["audio_filepath"]
            filepaths.append(str(audio_file))
            text = item["text"]
            reference_texts.append(text)
            durations.append(float(item["duration"]))
    return TestSet(filepaths, reference_texts, durations)

# Đường dẫn đến tệp manifest và mô hình .nemo
TEST_MANIFESTS = {
    "test_other": "/content/Fast-conformer-NVIDIA-Vivos/data_train/vivos_test_manifest_v2.json",
}

MODEL_PATH = '/content/drive/MyDrive/dataset_vivos/FastConformer-CTC-BPE-wer10.nemo'

# Tải dữ liệu từ manifest
test_sets = {manifest: load_data(path) for manifest, path in TEST_MANIFESTS.items()}

# Tải mô hình từ tệp .nemo
asr_model = load_model_from_nemo(MODEL_PATH)


from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)

# Tải mô hình từ tệp .nemo
def load_model_from_nemo(model_path: str):
    from nemo.collections.asr.models import EncDecCTCModel
    return EncDecCTCModel.restore_from(model_path)

# Đường dẫn đến mô hình của bạn
MODEL_PATH = '/content/drive/MyDrive/dataset_vivos/FastConformer-CTC-BPE-wer10.nemo'

# Tải mô hình
asr_model = load_model_from_nemo(MODEL_PATH)
asr_model.eval()

# Tạo cấu hình ConfidenceConfig
confidence_cfg = ConfidenceConfig(
    preserve_frame_confidence=True,
    preserve_token_confidence=True,
    preserve_word_confidence=True,
    aggregation="prod",
    exclude_blank=False,
    method_cfg=ConfidenceMethodConfig(
        name="max_prob",
        entropy_type="gibbs",
        alpha=0.5,
        entropy_norm="lin"
    )
)

# Thay đổi chiến lược decoding dựa trên loại mô hình (CTC hoặc RNNT)
is_rnnt = False  # Đặt true nếu bạn muốn sử dụng RNNT, false nếu sử dụng CTC

if is_rnnt:
    # Đối với RNNT
    asr_model.change_decoding_strategy(
        RNNTDecodingConfig(fused_batch_size=-1, strategy="greedy_batch", confidence_cfg=confidence_cfg)
    )
else:
    # Đối với CTC
    asr_model.change_decoding_strategy(
        CTCDecodingConfig(confidence_cfg=confidence_cfg)
    )

# Sau khi thay đổi decoding strategy, bạn có thể sử dụng mô hình để thực hiện dự đoán.

# Lấy test set
current_test_set = test_sets["test_other"]

# Kiểm tra nếu filepaths có dữ liệu
if not current_test_set.filepaths:
    raise ValueError("File paths in the test set are empty!")

# Chạy transcribe với batch_size và num_workers
transcriptions = asr_model.transcribe(
    audio=current_test_set.filepaths,  # Sử dụng 'file_paths' thay vì 'audio'
    batch_size=16,
    return_hypotheses=True,
    num_workers=4
)

# Kiểm tra nếu mô hình sử dụng RNNT
if is_rnnt:
    # Đối với RNNT, lấy kết quả đầu tiên của tuple
    transcriptions = transcriptions[0]

def round_confidence(confidence_number, ndigits=3):
    if isinstance(confidence_number, float):
        return round(confidence_number, ndigits)
    elif len(confidence_number.size()) == 0:  # torch.tensor with one element
        return round(confidence_number.item(), ndigits)
    elif len(confidence_number.size()) == 1:  # torch.tensor with a list if elements
        return [round(c.item(), ndigits) for c in confidence_number]
    else:
        raise RuntimeError(f"Unexpected confidence_number: `{confidence_number}`")


tran = transcriptions[0]
print(
    f"""    Recognized text: `{tran.text}`\n
    Word confidence: {[round_confidence(c) for c in tran.word_confidence]}\n
    Token confidence: {[round_confidence(c) for c in tran.token_confidence]}\n
    Frame confidence: {
        [([round_confidence(cc) for cc in c] if is_rnnt else round_confidence(c)) for c in tran.frame_confidence]
    }"""
)




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
