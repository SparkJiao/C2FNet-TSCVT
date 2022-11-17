from mmseg.models.decode_heads import UPerHead, FCNHead
from mmcv.utils import build_from_cfg
from mmseg.models.segmentors import EncoderDecoder
from lib.beit import BEiT
import torch
from torch import nn
import torch.nn.functional as F
from lib.mae import MAE
from mmseg.ops import resize


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
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        # initialize auxiliary head
        if auxiliary_head is not None:
            self.auxiliary_head = auxiliary_head

        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, x):
        orig_shape = x.shape
        x = self.backbone(x)
        if hasattr(self, "neck"):
            x = self.neck(x)

        # for xx in x:
        #     print(xx.size())
        x0 = self.decode_head(x)
        # print(x0.size())
        # x0 = F.interpolate(x0, scale_factor=4, mode='bilinear', align_corners=False)
        x0 = resize(
            input=x0,
            size=orig_shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if hasattr(self, "auxiliary_head"):
            x1 = self.auxiliary_head(x)
            # print(x1.size())
            # x1 = F.interpolate(x1, scale_factor=16, mode='bilinear', align_corners=False)
            x1 = resize(x1, size=orig_shape[2:], mode='bilinear', align_corners=self.align_corners)
        else:
            x1 = None
        return x0, x1


def mae_vit_segmentor(pretrained, depth=12, **kwargs):
    if depth == 12:
        out_indices = [3, 5, 7, 11]
    elif depth == 4:
        out_indices = [0, 1, 2, 3]
    elif depth == 6:
        out_indices = [2, 3, 4, 5]
    elif depth == 8:
        out_indices = [3, 5, 6, 7]
    elif depth == 10:
        out_indices = [3, 5, 7, 9]
    else:
        raise NotImplementedError()

    backbone = MAE(img_size=352,
                   patch_size=16,
                   embed_dim=768,
                   depth=depth,
                   num_heads=12,
                   mlp_ratio=4,
                   qkv_bias=True,
                   use_abs_pos_emb=True,  # here different
                   use_rel_pos_bias=True,
                   init_values=1.,
                   drop_path_rate=0.1,
                   out_indices=out_indices,
                   fpn1_norm='batch_norm')  # Single process training.
    decode_head = UPerHead(in_channels=[768, 768, 768, 768],
                           in_index=[0, 1, 2, 3],
                           pool_scales=(1, 2, 3, 6),
                           channels=768,
                           dropout_ratio=0.1,
                           num_classes=1,
                           norm_cfg=dict(type='BN2d', requires_grad=True),
                           align_corners=False,
                           loss_decode=dict(
                               type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    auxiliary_head = FCNHead(in_channels=768,
                             in_index=2,
                             channels=256,
                             num_convs=1,
                             concat_input=False,
                             dropout_ratio=0.1,
                             num_classes=1,
                             norm_cfg=dict(type='BN2d', requires_grad=True),
                             align_corners=False,
                             loss_decode=dict(
                                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))

    return Segmentor(backbone, decode_head, auxiliary_head=auxiliary_head, pretrained=pretrained)
