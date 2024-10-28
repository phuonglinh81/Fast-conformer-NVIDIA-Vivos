from pytorch_lightning.callbacks import Callback
import torch

class ConfidenceScoreCallback(Callback):
    def __init__(self, asr_model):
        super().__init__()
        self.asr_model = asr_model

    def on_validation_epoch_end(self, trainer, pl_module):
        # Lấy tập validation và tính toán các confidence scores
        val_dataloader = trainer.val_dataloaders[0]
        confidence_scores = []

        for batch in val_dataloader:
            # Chuyển batch sang thiết bị của mô hình
            audio_signal, audio_lengths, transcripts = batch
            audio_signal, audio_lengths = audio_signal.to(pl_module.device), audio_lengths.to(pl_module.device)

            # Lấy dự đoán từ mô hình và tính confidence score
            log_probs, encoded_len, predictions = pl_module.forward(audio_signal, audio_lengths)
            probs = torch.exp(log_probs)  # Chuyển log-probabilities sang probabilities

            # Tính confidence score trung bình cho mỗi mẫu trong batch
            for prob in probs:
                confidence_score = prob.max(dim=-1).values.mean().item()  # Lấy giá trị trung bình của xác suất cao nhất
                confidence_scores.append(confidence_score)

        # In ra confidence scores sau mỗi epoch
        avg_confidence_score = sum(confidence_scores) / len(confidence_scores)
        print(f'Average Confidence Score after epoch {trainer.current_epoch}: {avg_confidence_score}')
