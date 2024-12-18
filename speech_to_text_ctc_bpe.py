# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    trainer.devices=-1 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

# Pretrained Models

For documentation on existing pretrained models, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/results.html

"""

import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
# from confidence_callback import ConfidenceScoreCallback
import wandb
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch
from pytorch_lightning.callbacks import Callback

# class ConfidenceScoreCallback(pl.Callback):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
    
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # Tính confidence score cho mỗi batch trong tập validation
#         total_confidence_score = 0
#         num_batches = 0

#         val_dataloaders = trainer.val_dataloaders
#         # Kiểm tra xem val_dataloaders có là danh sách không
#         if isinstance(val_dataloaders, list):
#             val_dataloader = val_dataloaders[0]
#         else:
#             val_dataloader = val_dataloaders

#         for batch in val_dataloader:
#             inputs, labels = batch
#             logits = self.model(inputs)  # Dự đoán từ mô hình
            
#             # Tính toán confidence cho mỗi từ hoặc câu từ logits
#             confidence_score = self.calculate_confidence(logits)
#             total_confidence_score += confidence_score
#             num_batches += 1
        
#         # Tính điểm confidence trung bình cho epoch
#         average_confidence_score = total_confidence_score / num_batches
        
#         # Log điểm confidence vào W&B
#         if trainer.logger:
#             trainer.logger.experiment.log({"confidence_score": average_confidence_score})
    
#     def calculate_confidence(self, logits):
#         # Ví dụ tính confidence score từ logits
#         probabilities = torch.softmax(logits, dim=-1)
#         confidence_score = probabilities.max(dim=-1).values.mean().item()
#         return confidence_score

# class ConfidenceScoreCallback(Callback):
#     def __init__(self):
#         super().__init__()

#     def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
#         # Assume outputs are the logits or probabilities for the predictions
#         logits = outputs['logits']  # you may need to check the exact output format

#         # Compute probabilities
#         probabilities = torch.nn.functional.softmax(logits, dim=-1)

#         # Compute confidence as the max probability for each time step in each sample
#         max_probs, _ = probabilities.max(dim=-1)
#         confidence_scores = max_probs.mean(dim=-1)  # Average confidence over all time steps

#         # Log the average confidence score for the batch
#         avg_confidence_score = confidence_scores.mean().item()
#         trainer.logger.log_metrics({'val_confidence': avg_confidence_score}, step=trainer.global_step)

@hydra_runner(config_path="../conf/citrinet/", config_name="fast-conformer_ctc_bpe")
def main(cfg):
    experiment_name = f"fast-conformer_lr{cfg.model.optim.lr}_epochs{cfg.trainer.max_epochs}"
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)

    # Thiết lập exp_manager để tự động ghi log với W&B
    cfg.exp_manager.create_wandb_logger = True
    cfg.exp_manager.wandb_logger_kwargs = {
        "project": "fast-conformer-vivos",
        "name": experiment_name
    }


    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # # Khởi tạo callback ConfidenceScoreCallback
    # confidence_callback = ConfidenceScoreCallback(asr_model)

    # # Thêm callback vào trainer
    # trainer.callbacks.extend([confidence_callback])

    # Initialize the weights of the model from another model, if provided via config
    asr_model.maybe_init_from_pretrained_checkpoint(cfg)
    # asr_model.encoder.freeze()
    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)

    # Kết thúc phiên W&B sau khi huấn luyện hoàn tất
    wandb.finish()

    # Hủy bỏ ProcessGroup để tránh cảnh báo
    # from torch.distributed import destroy_process_group
    # destroy_process_group()

if __name__ == '__main__':
    main()