import torch.nn as nn
from fairseq import utils
import torch
from fairseq.models import FairseqEncoder

class BEiTEncoder(FairseqEncoder):

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        if hasattr(args, 'only_keep_pretrained_encoder_structure') and args.only_keep_pretrained_encoder_structure:
            pretrained = False
        else:
            pretrained = True
        
        if 'custom_size' in args.deit_arch:
            self.beit = create_model(args.deit_arch, pretrained=pretrained, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.beit = create_model(args.deit_arch, pretrained=pretrained, ape=args.ape, mask_ratio=args.mask_ratio)
        
        self.fp16 = args.fp16

    def forward(self, imgs):
        if self.fp16:
            imgs = imgs.half()

        x, encoder_embedding = self.deit.forward_features(imgs)  # bs, n + 2, dim
        x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        _encoder_out = encoder_out['encoder_out'][0]
        _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
        _encoder_embedding = encoder_out['encoder_embedding'][0]
        return {
            "encoder_out": [_encoder_out.index_select(1, new_order)],
            "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
            "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
            "encoder_states": [], 
            "src_tokens": [],
            "src_lengths": [],
        }