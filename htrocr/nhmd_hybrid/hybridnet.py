import torch
from torch import nn
import pytorch_lightning as pl
from htrocr.nhmd_hybrid.utils.scheduler import CosineWarmupScheduler
from torch import optim
import torch
from torch import nn
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate
from itertools import groupby
import htrocr.nhmd_hybrid.utils.device as devutils
from htrocr.nhmd_hybrid.encoder import PositionalEncoding, TransformerEncoder
from htrocr.nhmd_hybrid.backbone import BlockCNN
from torchvision.models import resnet50
from htrocr.nhmd_hybrid.nhmdtokenizer import NHMDTokenizer

class CNNTransformerHybrid(pl.LightningModule):

    def __init__(self, model_dim, num_heads, num_layers, num_labels, lr, warmup, tokenizer, max_iters=2000, dropout=0.1):
        '''
        Parameters
        ----------
        backbone_input_dim : integer
            Input image channel dimensionality.
        backbone_fib_dim : integer
            Output dimensionality after fib layers.
        backbone_expansion_factor : integer
            Fused Inverted bottleneck layer expansion factor.
        model_dim : integer
            Hidden dimensionality to use inside the Transformer.
        num_heads : integer
            Number of heads to use in the Multi-Head Attention blocks.
        num_layers : integer
            Number of encoder blocks to use.
        num_labels : integer
            -
        lr : integer
            Learning rate in the optimizer.
        warmup : integer
            Number of warmup steps. Usually between 50 and 500
        tokenizer : integer
            Tokenizer for label decoding
        max_iters : integer
            Number of maximum iterations the model is trained for.
        dropout : float
            Dropout in MLP and in-between layers ratio
        '''
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self.tokenizer = tokenizer
        self.blank_label = 0
        self.criterion = CTCLoss(blank=self.blank_label, reduction='mean', zero_infinity=True)
        self.cer = CharErrorRate()

    def _create_model(self):
        # Input dim -> Model dim
        self.backbone = BlockCNN()
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)

        self.encoder = TransformerEncoder(num_layers=self.hparams.num_layers,
                                            input_dim=self.hparams.model_dim,
                                            feedforward_dim=2*self.hparams.model_dim,
                                            num_heads=self.hparams.num_heads,
                                            dropout=self.hparams.dropout)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(self.hparams.model_dim, self.hparams.num_labels)


    def forward(self, x, mask=None, add_positional_encoding=True):
        '''
        Parameters
        ----------
        x : integer
            Batch of input image chunks.
        img_widths : integer
            Original image widths for reconstruction.
        '''
        x = self.backbone(x)
        # x = x.permute(0,3,2,1)
        # x = x.reshape(x.shape[0], -1, 512)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.encoder(x, mask=mask) # out => [Batch, SeqLen, Model_dim]
        sequence_output = self.dropout(x)
        logits = self.classifier(sequence_output) 
        return logits
    
    def accuracy(self, outputs, labels, attn_images):
        batch_size = outputs.shape[1] # batch size param after permutation
        _, max_index = torch.max(outputs, dim=2)
        cer_scores = []
        for i in range(batch_size):
            raw_prediction_idxs = max_index[:, i][attn_images[i] > 0]
            prediction_idxs = torch.IntTensor([c for c, _ in groupby(raw_prediction_idxs) if c != self.blank_label])
            prediction_idxs = devutils.to_device(prediction_idxs, devutils.default_device())
            try:
                # try to stop at eos_token_id for a specific line in batch
                idx = prediction_idxs.index(1)
                decoded_text = prediction_idxs[: idx + 1]
            except ValueError:
                decoded_text = prediction_idxs
            pred_str = self.tokenizer.decode(decoded_text, skip_special_tokens=True)
            label_str = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            cer_score = self.cer(pred_str, label_str)
            cer_scores.append(cer_score)
        cer_scores = torch.tensor(cer_scores)
        cer_mean = torch.mean(cer_scores)
        return cer_mean

    def _get_lengths(self, attention_mask):
        return torch.tensor([row.tolist().index(0) if 0 in row.tolist() else len(row) for row in attention_mask])

    def _calculate_loss(self, batch):
        images, labels, attn_images, attn_masks  = batch # batch.shape == torch.Size([32, 1, 40, some_width])
        batch_size = images.shape[0] 
        pred = self.forward(images, mask=attn_images) # pred.shape == (batch_size, input_length, num_labels)
        pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
        logits = pred.log_softmax(2)
        target_lengths = self._get_lengths(attn_masks)
        if attn_images is not None:
            input_lengths = self._get_lengths(attn_images).type_as(images)
        else:
            input_lengths = torch.full(size=(batch_size,), fill_value=logits.shape[0]).type_as(images)

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
        loss, cer = self._calculate_loss(batch,)
        # Logging
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cer = self._calculate_loss(batch)
        # Logging
        self.log(f"val_loss", loss, logger=True, sync_dist=True)
        self.log(f"val_cer", cer, logger=True, sync_dist=True)  

    def test_step(self, batch, batch_idx):
        loss, cer = self._calculate_loss(batch)
        # Logging
        self.log(f"test_loss", loss, logger=True, sync_dist=True)
        self.log(f"test_cer", cer, logger=True, sync_dist=True)

# img = torch.randn(1, 3, 32, 360)
# img = img.float()
# tokenizer = NHMDTokenizer()
# model = CNNTransformerHybrid(num_labels=154,
#                              tokenizer=tokenizer,
#                               model_dim = 512,
#                               num_heads = 4,
#                               num_layers = 16,
#                               lr = 5e-4,
#                               warmup = 100,
#                               dropout=0.1)
# result_shape = model(img).shape
# print(result_shape)