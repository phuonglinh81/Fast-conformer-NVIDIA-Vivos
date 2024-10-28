from pytorch_lightning.callbacks import Callback
import torch

class CTCLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.val_losses = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Tính toán CTC loss của batch
        loss = outputs['loss'].detach().cpu().item()
        self.val_losses.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Tính CTC loss trung bình và in ra
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        print(f'Average CTC Loss after epoch {trainer.current_epoch}: {avg_loss}')
        self.val_losses.clear()
