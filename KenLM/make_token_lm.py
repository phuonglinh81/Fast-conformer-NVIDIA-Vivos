# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import os
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Thread

from nemo.collections.common import tokenizers
from nemo.utils import logging


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Create token LM for input manifest and tokenizer.""",
    )
    parser.add_argument(
        "--manifest", required=True, type=str, help="Comma separated list of manifest files",
    )
    parser.add_argument(
        "--tokenizer_dir",
        required=True,
        type=str,
        help="The directory path to the tokenizer vocabulary + additional metadata",
    )
    parser.add_argument(
        "--tokenizer_type",
        required=True,
        type=str,
        choices=["bpe", "wpe"],
        help="The type of the tokenizer. Currently supports `bpe` and `wpe`",
    )
    parser.add_argument(
        "--lm_builder",
        default="/content/Fast-conformer-NVIDIA-Vivos/KenLM/scripts/asr_language_modeling/ngram_lm/make_phone_lm.py",
        type=str,
        help=(
            "The path or name of an LM builder. Supported builders: chain-est-phone-lm "
            "and scripts/asr_language_modeling/ngram_lm/make_phone_lm.py"
        ),
    )
    parser.add_argument(
        "--ngram_order", type=int, default=2, choices=[2, 3, 4, 5], help="Order of n-gram to use",
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="The path to store the token LM",
    )
    parser.add_argument(
        "--do_lowercase", action="store_true", help="Whether to apply lower case conversion on the text",
    )
    args = parser.parse_args()


    """ TOKENIZER SETUP """
    logging.info(f"Loading {args.tokenizer_type} tokenizer from '{args.tokenizer_dir}' ...")
    if args.tokenizer_type == "bpe":
        # This is a BPE Tokenizer
        model_path = os.path.join(args.tokenizer_dir, "tokenizer.model")

        # Update special tokens
        tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)
    else:
        # This is a WPE Tokenizer
        vocab_path = os.path.join(args.tokenizer_dir, "vocab.txt")
        tokenizer = tokenizers.AutoTokenizer(pretrained_model_name="bert-base-cased", vocab_file=vocab_path)

    logging.info(f"Tokenizer {tokenizer.__class__.__name__} loaded with {tokenizer.vocab_size} tokens")

    """ DATA PROCESSING """
    if "," in args.manifest:
        manifests = args.manifest.split(",")
    else:
        manifests = [args.manifest]

    offset = 1  # tokens in token LM cannot be 0
    tok_text_list = []
    num_lines = 0
    for manifest in manifests:
      logging.info(f"Processing manifest : {manifest} ...")
      with open(manifest, "r") as in_reader:
          for line in in_reader:
              item = json.loads(line)
              text = item["text"]
              if args.do_lowercase:
                  text = text.lower()
              tok_text = " ".join([str(i + offset) for i in tokenizer.text_to_ids(text)])
              tok_text_list.append(tok_text)
              num_lines += 1

    tok_texts = "\n".join(tok_text_list)
    del tok_text_list
    logging.info("Finished processing all manifests ! Number of sentences : {}".format(num_lines))

    """ LM BUILDING """
    logging.info(f"Calling {args.lm_builder} ...")
    pipe_args = [
      args.lm_builder,
      f"--ngram-order={args.ngram_order}",
      f"--no-backoff-ngram-order={args.ngram_order}",
      "--phone-disambig-symbol=-11",
    ]  
    # Gọi make_phone_lm.py
    p1 = Popen(pipe_args, stdout=PIPE, stdin=PIPE, text=True)

    # Truyền dữ liệu tokenized vào make_phone_lm.py
    out, err = p1.communicate(tok_texts)

    # Kiểm tra lỗi trong quá trình chạy
    if p1.returncode != 0:
        logging.error(f"Error while building LM: {err}")
        raise RuntimeError("Failed to build the language model.")

    # Ghi đầu ra vào tệp
    logging.info(f"LM is built, writing to {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(out)

    logging.info(f"Done writing to '{args.output_file}'.")

if __name__ == "__main__":
    main()