python /content/Fast-conformer-NVIDIA-Vivos/process_asr_text_tokenizer.py \
  --manifest="/content/Fast-conformer-NVIDIA-Vivos/data_train/scripts_manifest.json" \
  --data_root="/content/Fast-conformer-NVIDIA-Vivos/vocab_tokenizers/" \
  --vocab_size=10000 \
  --tokenizer="spe" \
  --no_lower_case \
  --spe_type="bpe" \
  --spe_character_coverage=1.0 \
  --log
