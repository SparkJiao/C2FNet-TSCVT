from mmseg.models.decode_heads import UPerHead, FCNHead
from mmcv.utils import build_from_cfg
from mmseg.models.segmentors import EncoderDecoder
from lib.beit import BEiT
import torch
from torch import nn
from lib.mae import MAE


class Segmentor(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 decode_head: nn.Module,
                 neck=None,
                 auxiliary_head=None,
                 pretrained=None):
        super().__init__()

        self.backbone = backbone
        if neck is not None:
            self.neck = neck

        # initialize decode head
        self.decode_head = decode_head
        self.align_cornets = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        # initialize auxiliary head
        if auxiliary_head is not None:
            self.auxiliary_head = auxiliary_head

        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "neck"):
            x = self.neck(x)

        x0 = self.decode_head(x)
        if hasattr(self, "auxiliary_head"):
            x1 = self.auxiliary_head(x)
        else:
            x1 = None
        return x0, x1


def mae_vit_segmentor(**kwargs):
    backbone = MAE(img_size=512,
                   patch_size=16,
                   embed_dim=768,
                   depth=12,
                   num_heads=12,
                   mlp_ratio=4,
                   qkv_bias=True,
                   use_abs_pos_emb=True,  # here different
                   use_rel_pos_bias=True,
                   init_values=1.,
                   drop_path_rate=0.1,
                   out_indices=[3, 5, 7, 11])
    decode_head = UPerHead(in_channels=[768, 768, 768, 768],
                           in_index=[0, 1, 2, 3],
                           pool_scales=(1, 2, 3, 6),
                           channels=768,
                           dropout_ratio=0.1,
                           num_classes=1,
                           norm_cfg=dict(type='SyncBN', requires_grad=True),
                           align_corners=False,
                           loss_decode=dict(
                               type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

