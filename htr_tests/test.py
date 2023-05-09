import torch
from torch import masked_fill

pad = 0

src = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]])
src_mask = (src != pad).type(torch.int16).unsqueeze(1)
print(src_mask)
print(src_mask.shape)
seq_len = src.shape[-1]
bs = src.shape[0]

# Multiheads
heads = 8
attn_shape = (bs, heads, seq_len, seq_len)

attn_scores = torch.rand(attn_shape)
print(attn_scores.shape)
# src_mask_bool = (src_mask == 1)
# print(src_mask_bool.shape)
attn_scores_masked = attn_scores.masked_fill(src_mask == 0, value=-1e9)
print(attn_scores_masked)