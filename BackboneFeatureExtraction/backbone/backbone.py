# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
import os
from typing import Dict, List
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from groundingdino.util.misc import NestedTensor, clean_state_dict, is_main_process
from groundingdino.datasets import transforms as T

from .position_encoding import build_position_encoding
from .swin_transformer import build_swin_transformer
import BackboneFeatureExtraction.FacebookDino.vision_transformer as ViT
import BackboneFeatureExtraction.FacebookDino.utils as ViT_utils
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_indices: list,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
                print('false')
            else:
                parameter.requires_grad_(True)
                print('true')

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update(
                {"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)}
            )

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        # import ipdb; ipdb.set_trace()
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        dilation: bool,
        return_interm_indices: list,
        batch_norm=FrozenBatchNorm2d,
    ):
        if name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(),
                norm_layer=batch_norm,
            )
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ("resnet18", "resnet34"), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices) :]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding,args):
        super().__init__(backbone, position_embedding)
        self.args = args
        args_dict = dict(
            arch='vit_small', batch_size_per_gpu=128, checkpoint_key='teacher',
            data_path='/path/to/imagenet', dist_url='env://', dump_features=None,
            load_features=None, local_rank=0, nb_knn=[10, 20, 100, 200],
            num_workers=10, patch_size=16,  # 16
            pretrained_weights=self.args.dino_pretrained_path,
            temperature=0.07, use_cuda=False)
        args_ViT = argparse.Namespace(**args_dict)
        self.backbone_support = ViT.__dict__[args_ViT.arch](patch_size=args_ViT.patch_size, num_classes=0)
        self.backbone_support.to(device = self.args.device)
        ViT_utils.load_pretrained_weights(self.backbone_support, args_ViT.pretrained_weights, args_ViT.checkpoint_key, args_ViT.arch,
                                      args_ViT.patch_size)
        if self.args.freeze_support_backbone:
            self.backbone_support.eval()
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

    def forward_supp_branch(self, tensor_list: NestedTensor, support_out_indices=None):
        backbone_support = self.backbone_support
        #xs = backbone_support(tensor_list, support_out_indices,supp=True)
        out = []
        for support_tensors in tensor_list.tensors.permute(4,0,1,2,3):
            xs = backbone_support(support_tensors)
            out.append(xs)
        out = torch.stack(out,dim=-1)
        return out
        # out_tensors,out_masks = [],[]
        # for index in range(len(xs)):
        #     out_tensors.append(xs[index][0].tensors)
        #     out_masks.append(xs[index][0].mask)
        #     #print(xs[index][0].tensors.shape,xs[index][0].mask.shape)
        # out = {'tensors':torch.stack(out_tensors,dim=-1),'mask':torch.stack(out_masks,dim=-1)}
        # return out

def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone:
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords:
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = True
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    args.backbone_freeze_keywords
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in ["resnet50", "resnet101"]:
        backbone = Backbone(
            args.backbone,
            train_backbone,
            args.dilation,
            return_interm_indices,
            batch_norm=FrozenBatchNorm2d,
        )
        bb_num_channels = backbone.num_channels
    elif args.backbone in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            dilation=False,
            out_indices=tuple([1, 2, 3]),
            use_checkpoint=True,
            frozen_stages = args.frozen_stages,
        )
        if "backbone_dir" in args:
            checkpoint = torch.load(args.backbone_dir, map_location='cpu')['model']
            from collections import OrderedDict
            def key_select_function(keyname):
                if 'head' in keyname:
                    return False
                if backbone.dilation and 'layers.3' in keyname:
                    return False
                return True
            _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices
    ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding,args)
    model.num_channels = bb_num_channels
    assert isinstance(
        bb_num_channels, List
    ), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))
    # import ipdb; ipdb.set_trace()
    return model

def process_image(img_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)

    return image_transformed
def nested_tensor(samples):
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    return samples


if __name__ == "__main__":
    #from groundingdino.util.slconfig import SLConfig
    #from groundingdino.util.misc import nested_tensor_from_tensor_list
    from util.misc import nested_tensor_from_tensor_list
    from util.slconfig import SLConfig
    config_args = SLConfig.fromfile('../../config/train_config.py')

    backbone = build_backbone(config_args)

    img_a = cv2.imread('/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/dataset/support/1_shot_support/train2017/000000051191/0000.jpg')
    img_b = cv2.imread('/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/dataset/support/1_shot_support/train2017/000000052527/0000.jpg')

    img_a = process_image(img_bgr=img_a)[None]
    img_b = process_image(img_bgr=img_b)[None]

    img_x = nested_tensor(img_a)
    img_y = nested_tensor(img_b)

    out_x, pos_x = backbone(img_x)
    out_y, pos_y = backbone.forward_supp_branch(img_y, support_out_indices=config_args.support_out_indices)
    print(len(out_x), len(pos_x))
    print(len(out_y), len(pos_y))
