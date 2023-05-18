import torch 
import torch.nn as nn
import logging
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model

_logger = logging.getLogger(__name__)

__all__ = [
    'nhmddeit_small_patch16_384', 
    'nhmdbeit_base_patch16_384',
    'nhmdbeit_large_patch16_384',
]

class ViTSTR(VisionTransformer):
    '''
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x, seqlen: int = 25):
        x = self.forward_features(x)
        x = x[:, :seqlen]

        # batch, seqlen, embsize
        b, s, e = x.size()
        x = x.reshape(b*s, e)
        x = self.head(x).view(b, s, self.num_classes)
        return x


def load_pretrained(model, cfg=None, trasform_patch_size=False, is_trocr_state_dict=False, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')

    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = torch.hub.load_state_dict_from_url(
            url=cfg['url'],
            map_location="cpu", check_hash=True
        )

    if "model" in state_dict.keys():
        state_dict = state_dict["model"]
    
    new_dict = {}
    if is_trocr_state_dict:
        for k,v in state_dict.items():
            if k.startswith('decoder'):
                continue
            if k.startswith('encoder.deit.'):
                newK = k.split('encoder.deit.')[1]
                new_dict[newK] = v
            else:
                new_dict[k] = v
        state_dict = new_dict

    if trasform_patch_size:
        # adapt 224 model to 384
        model_seq_len = model.state_dict()['pos_embed'].shape[1]
        ckpt_seq_len = state_dict['pos_embed'].shape[1]
        if model_seq_len <= ckpt_seq_len:
            state_dict['pos_embed'] = state_dict['pos_embed'][:, :model_seq_len, :]
        else:
            t = model.state_dict()['pos_embed']
            t[:, :ckpt_seq_len, :] = state_dict['pos_embed']
            state_dict['pos_embed'] = t

    # Convert to grayscale
    conv1_name = cfg['first_conv']
    conv1_weight = state_dict[conv1_name + '.weight']
    # Some weights are in torch.half, ensure it's float for sum on CPU
    conv1_type = conv1_weight.dtype
    conv1_weight = conv1_weight.float()
    O, I, J, K = conv1_weight.shape
    if I > 3:
        assert conv1_weight.shape[1] % 3 == 0
        # For models with space2depth stems
        conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
        conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
    else:
        conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
    conv1_weight = conv1_weight.to(conv1_type)
    state_dict[conv1_name + '.weight'] = conv1_weight

    # completely discard fully connected for all other differences between pretrained and created model
    classifier_name = cfg['classifier']
    del state_dict[classifier_name + '.weight']
    del state_dict[classifier_name + '.bias']
    if not is_trocr_state_dict:
        del state_dict[classifier_name + '_dist.weight']
        del state_dict[classifier_name + '_dist.bias']
    strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)


@register_model
def nhmddeit_small_patch16_384(pretrained=False, **kwargs):
    model = ViTSTR(distilled=True,
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    )
    if pretrained:
        load_pretrained(
            model, trasform_patch_size=True)
    return model

@register_model
def nhmdbeit_base_patch16_384(pretrained=False, **kwargs):
    model = ViTSTR(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg(
        url="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt"
    )
    if pretrained:
        load_pretrained(
            model, trasform_patch_size=False, is_trocr_state_dict=True)
    return model

@register_model
def nhmdbeit_large_patch16_384(pretrained=False, **kwargs):
    model = ViTSTR(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, in_chans=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg(
        url="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-handwritten.pt"
    )
    if pretrained:
        load_pretrained(
            model, trasform_patch_size=False, is_trocr_state_dict=True)
    return model

def create_vit(num_tokens, model=None, checkpoint_path=''):
    nhmd_vit = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path)
    nhmd_vit.reset_classifier(num_classes=num_tokens)
    return nhmd_vit

# if __name__ == '__main__':
#     from transformers import VisionEncoderDecoderModel

#     # Instantiate the VisionEncoderDecoderModel
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

#     # Get the number of parameters in the encoder
#     encoder_parameters = model.encoder.num_parameters()
#     print("Number of parameters in the encoder:", encoder_parameters)

#     # Get the number of parameters in the decoder
#     decoder_parameters = model.decoder.num_parameters()
#     print("Number of parameters in the decoder:", decoder_parameters)
    # create_vit(154, 'nhmdbeit_large_patch16_384')
    # state_dict = torch.hub.load_state_dict_from_url(
    #         url='https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt',
    #         map_location="cpu", check_hash=True
    #     )
    # print(state_dict)
