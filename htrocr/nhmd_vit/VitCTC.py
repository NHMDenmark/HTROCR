import torch
import pytorch_lightning as pl
from htrocr.nhmd_vit.utils.scheduler import CosineWarmupScheduler
from torch import optim
import torch
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate
from itertools import groupby
import htrocr.nhmd_vit.utils.device as devutils
from htrocr.nhmd_vit.NHMDVit import create_vit
from htrocr.nhmd_vit.nhmdtokenizer import NHMDTokenizer

class ViTCTC(pl.LightningModule):
    def __init__(self, model, max_len, num_labels, lr, warmup, tokenizer, max_iters=2000):
        super().__init__()
        self.save_hyperparameters()
        self.nhmd_vit = create_vit(num_labels, model)
        self.tokenizer = tokenizer
        self.blank_label = 0
        self.criterion = CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        self.cer = CharErrorRate()
        self.seq_len = max_len

    def forward(self, x, seqlen=50):
        return self.nhmd_vit(x, seqlen)

    def accuracy(self, outputs, labels):
        batch_size = outputs.shape[1] # batch size param after permutation
        _, max_index = torch.max(outputs, dim=2)
        cer_scores = []
        for i in range(batch_size):
            raw_prediction_idxs = max_index[:, i]
            prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction_idxs) if c != self.blank_label])
            prediction_idxs = devutils.to_device(prediction_idxs, devutils.default_device())
            try:
                idx = prediction_idxs.index(1)
            except:
                decoded_text = prediction_idxs
            else:
                decoded_text = prediction_idxs[: idx + 1]
            pred_str = self.tokenizer.decode(decoded_text, skip_special_tokens=True)
            label_str = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            cer_score = self.cer(pred_str, label_str)
            cer_scores.append(cer_score)
        cer_scores = torch.tensor(cer_scores)
        cer_mean = torch.mean(cer_scores)
        return cer_mean

    def _get_lengths(self, attention_mask):
        lens = [
            row.argmin() if row.argmin() > 0 else len(row) for row in attention_mask
        ]
        return torch.tensor(lens)

    def _calculate_loss(self, batch, mode="train"):
        images, labels, attn_masks = batch # batch.shape == torch.Size([32, 1, 40, some_width])
        batch_size = images.shape[0] 
        pred = self.forward(images, self.seq_len) # pred.shape == (batch_size, input_length, num_labels)
        pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
        logits = pred.log_softmax(2)
        target_lengths = self._get_lengths(attn_masks)
        input_lengths = torch.full(
            size=(batch_size,), fill_value=logits.shape[0]
        ).type_as(images)
        loss = self.criterion(logits, labels, input_lengths.long(), target_lengths.long())
        cer = self.accuracy(logits, labels)
        return loss, cer

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        loss, cer = self._calculate_loss(batch, mode="train")
        # Logging
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cer = self._calculate_loss(batch, mode="val")
        # Logging
        self.log(f"val_loss", loss, logger=True, sync_dist=True)
        self.log(f"val_cer", cer, logger=True, sync_dist=True)  

    def test_step(self, batch, batch_idx):
        loss, cer = self._calculate_loss(batch, mode="test")
        # Logging
        self.log(f"test_loss", loss, logger=True, sync_dist=True)
        self.log(f"test_cer", cer, logger=True, sync_dist=True)


# img = torch.randn(1, 1, 384, 384)
# img = img.float()
# tokenizer = NHMDTokenizer()
# model = ViTCTC('nhmddeit_small_patch16_384',
#                 tokenizer=tokenizer,
#                 num_labels = 154,
#                 lr = 5e-4,
#                 warmup = 100)
# result_shape = model(img, 50).shape
# print(result_shape)
