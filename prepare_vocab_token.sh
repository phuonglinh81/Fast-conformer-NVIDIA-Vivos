!python /content/process_asr_text_tokenizer.py \
  --manifest="/content/drive/MyDrive/dataset_vivos/vivos/vivos_train_manifest.json" \
  --data_root="/content/tokenizers/dict/" \
  --vocab_size=10000 \
  --tokenizer="spe" \
  --no_lower_case \
  --spe_type="bpe" \
  --spe_character_coverage=1.0 \
  --log
