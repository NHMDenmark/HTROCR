import torch
from torch import nn
import pytorch_lightning as pl
from utils.scheduler import CosineWarmupScheduler
from torch import optim
import torch
from torch import nn
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate
from itertools import groupby
import utils.device as devutils
from encoder import PositionalEncoding, TransformerEncoder
from backbone import BlockCNN

class HybridSeq2SeqModel(pl.LightningModule):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        # Remove logit classifier head and map to decoder input
        self.encoder.dropout = nn.Dropout(0)
        self.encoder.classifier = nn.Linear(512, 768)
        self.decoder = decoder

    def forward(self, batch):
        images, labels, attn_images, attn_masks  = batch # batch.shape == torch.Size([32, 1, 40, some_width])
        batch_size = images.shape[0] 
        x = self.encoder(images, mask=attn_images)
        decoder_outputs = self.decoder(
            input_ids=labels,
            attention_mask=attn_masks,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            use_cache=None,
            past_key_values=None,
            return_dict=None
        )
        # print('backbone x shape', x.shape)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.encoder(x, mask=mask) # out => [Batch, SeqLen, Model_dim]
        sequence_output = self.dropout(x)
        logits = self.classifier(sequence_output)
        return logits
    
    def accuracy(self, outputs, labels, attn_images, mode):
        batch_size = outputs.shape[1] # batch size param after permutation
        _, max_index = torch.max(outputs, dim=2)
        cer_scores = []
        for i in range(batch_size):
            # raw_prediction_idxs = list(max_index[:, i].detach().cpu().numpy())
            # print(list(max_index[:, i].detach().cpu().numpy()))
            # print(attn_images)
            raw_prediction_idxs = max_index[:, i][attn_images[i] > 0]
            prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction_idxs) if c != self.blank_label])
            prediction_idxs = devutils.to_device(prediction_idxs, devutils.default_device())
            try:
                # try to stop at eos_token_id for a specific line in batch
                idx = prediction_idxs.index(2)
            except:
                decoded_text = prediction_idxs
            else:
                decoded_text = prediction_idxs[: idx + 1]
            pred_str = self.tokenizer.decode(decoded_text, skip_special_tokens=True)
            label_str = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            if mode == 'val':
                self.log(f"Pred", pred_str, logger=True, sync_dist=True)
                self.log(f"Actual", label_str, logger=True, sync_dist=True)
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
        images, labels, attn_images, attn_masks  = batch # batch.shape == torch.Size([32, 1, 40, some_width])
        batch_size = images.shape[0] 
        pred = self.forward(images, mask=attn_images) # pred.shape == (batch_size, input_length, num_labels)
        pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
        logits = pred.log_softmax(2)
        target_lengths = self._get_lengths(attn_masks)

        if attn_images is not None:
            input_lengths = self._get_lengths(attn_images).type_as(images)
        else:
            input_lengths = torch.full(
                size=(batch_size,), fill_value=logits.shape[0]
            ).type_as(images)
        # print('logits', logits.shape)
        # print('labels', labels.shape)
        # print('input_l', input_lengths.shape)
        # print('target_l', target_lengths.shape)
        loss = self.criterion(logits, labels, input_lengths.long(), target_lengths.long())
        cer = self.accuracy(logits, labels, attn_images)

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