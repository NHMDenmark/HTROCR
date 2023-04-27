import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch import nn
import pytorch_lightning as pl
from utils.scheduler import CosineWarmupScheduler
from torchvision.models import resnet50, resnet101
import math
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import optim
import collections
from itertools import repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CTCLoss

class FusedInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor):
        '''
        Parameters
        ----------
        in_channels : integer
            Input channels to the FIB block.
        out_channels : integer
            Out channels after fused inverted bottleneck layers.
        expansion_factor : integer
            Expansion factor for depthwise separable convolution in fused inverted bottleneck layers.
        '''
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    
class ReduceBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, f3, f4):
        '''
        Parameters
        ----------
        in_channels : integer
            Input channels to the reduce block.
        f1 : integer
            Height reduction convolution with 'valid' padding kernel size
        f2 : integer
            Width feature extraction convolution with 'same' padding kernel size
        f3 : integer
            Height feature extraction convolution with 'same' padding kernel size
        '''
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=(1,1), padding='same')
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(f1,1), stride=(1,1), padding='valid')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,f2), stride=(1,1), padding='same')
        self.conv3 = nn.Conv2d(in_channels, in_channels*2, kernel_size=(f3,1), stride=(1,1), padding='same')
        self.residual = nn.Conv2d(in_channels, in_channels*2, kernel_size=(f4,1), stride=(1,1), padding='valid')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels * 2)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.relu(out)
        residual = self.residual(x)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        # residual = torch.mean(residual, dim=2, keepdim=True)
        out = torch.cat((out,residual), dim=1)
        return out

class Backbone(nn.Module):
    '''
    CNN backbone with Fused Inverted Bottleneck layers
    adjusted according to:
    'Rethinking Text Line Recognition Models' by Diaz et al. 2021
    '''
    def __init__(self, in_channels, out_channels, expansion_factor):
        '''
        Parameters
        ----------
        in_channels : integer
            Image color channels
        out_channels : integer
            Out channels after fused inverted bottleneck layers.
        expansion_factor : integer
            Expansion factor for depthwise separable convolution in fused inverted bottleneck layers.
        '''
        super().__init__()
        self.space_to_depth = nn.PixelUnshuffle(4)
        self.depthwise_conv = nn.Conv2d(16* in_channels, 16 * in_channels * expansion_factor, kernel_size=3, padding=1,  groups=16*in_channels)
        self.bn1 = nn.BatchNorm2d(16 * in_channels * expansion_factor)
        self.pointwise_conv = nn.Conv2d(16 * expansion_factor, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fib_layers = nn.ModuleList([FusedInvertedBottleneck(out_channels, out_channels, expansion_factor) for _ in range(10)])
        self.reduce_block = ReduceBlock(out_channels, 10, 3, 6, 10)

    def forward(self, x):
        x = x.unsqueeze(1) # add channel dimension
        out = self.space_to_depth(x)
        out = self.depthwise_conv(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        for layer in self.fib_layers:
            out = layer(out)
        out = self.reduce_block(out)
        return out

print('Running Backbone test.')
img = torch.randn(1, 40, 768)
img = img.float()
model = Backbone(1, 64, 8)
result_shape = model(img).shape
assert result_shape == torch.Size([1, 256, 1, 192]), f'Shapes do not match! Received {result_shape}'
print('Backbone test passed.')

'''
The following Transformer encoder is based on the content from
'UvA Deep Learning Tutorials' by Phillip Lippe 2022
University of Amsterdam
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        '''
        Parameters
        ----------
        d_model : integer
            Input hidden dimensionality.
        max_len : integer
            Sequence maximum length.
        '''
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

def scaled_dot_product(q, k, v, mask=None):
    '''
    Computes Attention(Q,K,V).
    '''
    # hidden dimmensionality of key/query
    d_k = q.size()[-1]
    # Since we want to support higher dimmensions (for batches),
    # transpose is along the second-to-last and last dimensions.
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    # Optional masking for sequences with different lengths
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        '''
        Parameters
        ----------
        input_dim : integer
            Input sequence channel (model) dimensionality.
        embed_dim : integer
            Embedding vector dimmensionality.
        num_heads : integer
            Number of independent attention heads.
        '''
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        # Weight matrix after head concat
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, feedforward_dim, dropout=0.0):
        '''
        Parameters
        ----------
        input_dim : integer
            Input sequence channel (model) dimensionality.
        num_heads : integer
            Number of independent attention heads to use for Multi-Head Attention.
        feedforward_dim : integer
            Multilayer perceptron hidden layer dimensionality.
        dropout : float
            Dropout in MLP and in-between layers ratio
        '''
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps



class CNNTransformerHybrid(pl.LightningModule):

    def __init__(self, backbone_input_dim, 
                       backbone_fib_dim,
                       backbone_expansion_factor,
                       model_dim,
                       num_heads,
                       num_layers,
                       vocab_size,
                       lr,
                       warmup,
                       max_iters=2000,
                       dropout=0.0):
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
        vocab_size : integer
            Vocabulary size
        lr : integer
            Learning rate in the optimizer.
        warmup : integer
            Number of warmup steps. Usually between 50 and 500
        max_iters : integer
            Number of maximum iterations the model is trained for.
        dropout : float
            Dropout in MLP and in-between layers ratio
        '''
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self.criterion = CTCLoss(zero_infinity=True)

    def _create_model(self):
        # Input dim -> Model dim
        self.backbone = Backbone(self.hparams.backbone_input_dim, 
                                 self.hparams.backbone_fib_dim, 
                                 self.hparams.backbone_expansion_factor)

        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)

        self.encoder = TransformerEncoder(num_layers=self.hparams.num_layers,
                                            input_dim=self.hparams.model_dim,
                                            feedforward_dim=2*self.hparams.model_dim,
                                            num_heads=self.hparams.num_heads,
                                            dropout=self.hparams.dropout)
        self.linear = nn.Linear(self.hparams.model_dim, self.hparams.vocab_size)


    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        data, labels, origw = batch
        batch_size = data.shape[0]  # data.shape == torch.Size([8, 40, 200])
        x_train = data.view(data.shape[0], 1, data.shape[1], data.shape[2])
        input_lengths = torch.IntTensor(batch_size).fill_(50)
        target_lengths = torch.IntTensor([len(t) for t in labels])
        pred = self.forward(x_train, origw) # pred.shape == torch.Size([8, 50, 91]) => (batch_size, input_length, no_of_classes)
        pred = pred.permute(1, 0, 2) # we need (input_length, batch_size, no_of_classes)
        loss = self.criterion(pred, labels, input_lengths, target_lengths)
        # Logging
        self.log(f"{mode}_loss", loss)
        return loss

    def concat_chunks(self, batches, width_stack):
        '''
        Concat feature chunks after encoder

        Parameters
        ----------
        batches : Arraylike
            Constructed feature block
        width_stack : integer
            Stack of original image widths in reverse order.
        '''
        cat_features = []
        padding = 10
        w = width_stack.pop()
        cat_feature = torch.zeros((int(w/4), batches.size(2)))
        target_size = w/4
        processed = 0
        step = 60
        for element in batches:
            if w/4 >= element.shape[0]:
                if len(width_stack) == 0:
                    cat_features.append(element)
                    break
                w = width_stack.pop()
                continue
            if processed < target_size and processed + 60 < target_size:
                # Process regular chunk
                step = 60
            elif processed < target_size and processed + 60 > target_size:
                # Final chunk
                step = target_size - processed
            cat_feature[0:step,:] = element[padding:padding+step,:]
            processed += step
            if processed == target_size:
                # Concatenation finished
                cat_features.append(cat_feature)
                if len(width_stack) == 0:
                    break
                processed = 0
                cat_feature = torch.zeros((int(w/4), batches.size(2)))
                w = width_stack.pop()
        return torch.stack(cat_features, dim=0)

    def forward(self, x, img_widths, mask=None, add_positional_encoding=True):
        '''
        Parameters
        ----------
        x : integer
            Batch of input image chunks.
        img_widths : integer
            Original image widths for reconstruction.
        '''
        x = self.backbone(x)
        # Input features of shape [Batch, SeqLen, input_dim]
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.encoder(x, mask=mask)
        x = self.concat_chunks(x, img_widths)
        x = self.linear(x)
        output = F.log_softmax(x, dim=2)
        return output

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.backbone(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")


print('Transformer test.')
# img = torch.randn(1, 40, 320)
img = torch.randn(1, 40, 200)
img = img.float()
model = CNNTransformerHybrid(backbone_input_dim = 1,
                             backbone_fib_dim = 64,
                             backbone_expansion_factor = 8,
                             model_dim = 256,
                             num_heads = 4,
                             num_layers = 16,
                             vocab_size = 91,
                             lr = 5e-4,
                             warmup = 100,
                             dropout=0.1)
result_shape = model(img, [200]).shape
assert result_shape == torch.Size([1, 50, 91]), f'Transformer shape test check failed. Resulting shape {result_shape}.'
print('Test passed.')