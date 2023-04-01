import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import groupby
from torchmetrics import CharErrorRate
import utils.device as device

blank_label = 10
num_classes = 11
cnn_output_height = 4
gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 32


def accuracy(outputs, labels):
    cer = CharErrorRate()
    batch_size = outputs.shape[1] # batch size param after permutation
    _, max_index = torch.max(outputs, dim=2)
    cer_scores = []
    val_correct = 0
    val_total = 0
    for i in range(batch_size):
        raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
        prediction = device.to_device(prediction, device.default_device())
        if len(prediction) == len(labels[i]) and torch.all(prediction.eq(labels[i])):
            val_correct += 1
        val_total += 1
        cer_score = 0 #cer(prediction, labels[i])
        cer_scores.append(cer_score)
    acc = torch.tensor(val_correct/val_total)
    cer_scores = torch.tensor(cer_scores)
    cer_mean = 0.0#torch.mean(cer_scores)
    return acc, cer_mean

class DCNNEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
    self.norm1 = nn.InstanceNorm2d(32)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
    self.norm2 = nn.InstanceNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
    self.norm3 = nn.InstanceNorm2d(64)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
    self.norm4 = nn.InstanceNorm2d(64)
    self.gru_input_size = cnn_output_height * 64
    self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(gru_hidden_size * 2, num_classes)
        
  def forward(self, xb):
    batch_size = xb.shape[0]
    out = self.conv1(xb)
    out = self.norm1(out)
    out = F.leaky_relu(out)
    out = self.conv2(out)
    out = self.norm2(out)
    out = F.leaky_relu(out)
    out = self.conv3(out)
    out = self.norm3(out)
    out = F.leaky_relu(out)
    out = self.conv4(out)
    out = self.norm4(out)
    out = F.leaky_relu(out)
    out = out.permute(0, 3, 2, 1)
    out = out.reshape(batch_size, -1, self.gru_input_size)
    out, _ = self.gru(out)
    out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
    return out

  def training_step(self, batch, criterion):
    data, labels = batch
    batch_size = data.shape[0]  # data.shape == torch.Size([64, 28, 140])
    x_train = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
    input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
    target_lengths = torch.IntTensor([len(t) for t in labels])
    pred = self(x_train) # pred.shape == torch.Size([64, 32, 11]) => (batch_size, input_length, no_of_classes)
    pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
    loss = criterion(pred, labels, input_lengths, target_lengths)
    return loss

  def validation_step(self, batch, criterion):
    data, labels = batch
    batch_size = data.shape[0]  # data.shape == torch.Size([64, 28, 140])
    x_val = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
    input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
    target_lengths = torch.IntTensor([len(t) for t in labels])
    pred = self(x_val) # pred.shape == torch.Size([64, 32, 11]) => (batch_size, input_length, no_of_classes)
    pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
    loss = criterion(pred, labels, input_lengths, target_lengths)
    acc, _ = accuracy(pred, labels)
    return {'val_loss': loss, 'val_acc': acc}
  
  def validation_epoch_end(self, outputs):
    batch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    batch_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
    # batch_cer = torch.stack([x['val_cer'] for x in outputs]).mean()
    return {'val_loss': batch_loss.item(), 'val_acc': batch_acc.item()}

  def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
  

