from torch import nn
import torch
import torch.nn.functional as F
import copy
from util.misc import NestedTensor,inverse_sigmoid
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss
from config import train_config
from torchvision import transforms as pth_transforms
import numpy as np

from PIL import Image
import cv2
from util.misc import nested_tensor_from_tensor_list
from BackboneFeatureExtraction.backbone.backbone import build_backbone
transform = pth_transforms.Compose([
    #pth_transforms.Resize(256, interpolation=3),
    #pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# def load_image(img_bgr: np.ndarray) -> torch.Tensor:
#     transform = T.Compose(
#         [
#             #T.RandomResize([1000], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image_pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#     image_transformed, _ = transform(image_pillow, None)
#     return image_transformed
class TIDE(nn.Module):
    def __init__(self,position_embedding,transformer,args):
        super().__init__()
        self.device = torch.device('cuda')
        self.pos_emb = position_embedding
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = train_config.num_feature_levels
        self.args = args
        self.backbone = build_backbone(self.args).to(device=args.device)
        self.backbone_num_channels = [192, 384, 768][4-self.num_feature_levels:]
        self.input_proj = self.__build_input_proj__()
        self.batch_size = train_config.batch_size
        #特征降维384->256
        self.feat_map = nn.Linear(train_config.prompt_feat_dim, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.args = args
        #head子模块
        if args.two_stage_type == 'standard':
            _class_embed = ContrastiveEmbed(args.max_support_len)
        elif args.two_stage_type == 'no':
            _class_embed = nn.Linear(self.hidden_dim,2)
        _bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
        self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        self._reset_parameters()
        self.max_support_dim = args.max_support_len
    # def get_support_feats(self, support_np):
    #     backbone = build_backbone(self.args).to(self.device)
    #     image_s = load_image(support_np)
    #     images_s = image_s.to(self.device)[None]
    #     samples_s = nested_tensor_from_tensor_list(images_s)
    #     features_s = backbone.forward_supp_branch(samples_s, self.args.support_out_indices)
    #     features_s = features_s[0].tensors
    #     #把torch.Size([1, 384, 50, 67])变换成torch.Size([1, 384])
    #     #features_s = torch.mean(features_s, dim=(2,3))
    #     norms = torch.linalg.norm(features_s, axis=1)
    #     for n in range(len(features_s)):
    #         features_s[n] /= norms[n]
    #     return features_s
    #
    # def get_query_feats(self, query_np):
    #     backbone = build_backbone(self.args).to(self.device)
    #     image_q = load_image(query_np)
    #     images_q = image_q.to(self.device)[None]
    #     samples_q = nested_tensor_from_tensor_list(images_q)
    #     features_q = backbone(samples_q)
    #     pos = features_q[1]
    #     feature_tensors = features_q[0][2].tensors[0]#取最后一层
    #     return feature_tensors,pos
    def __build_input_proj__(self):
        num_backbone_outs = len(self.backbone_num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = self.backbone_num_channels[_]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
        for _ in range(self.num_feature_levels - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
            in_channels = self.hidden_dim

        return nn.ModuleList(input_proj_list)

    def forward(self, samples_q, samples_s):
        #print(samples_q.shape)
        query_feats,_ = self.backbone(samples_q)
        query_feats = tuple([query_feats[-1]])

        query_pos = []
        for feat in query_feats:
            query_pos.append(self.pos_emb(feat))
        # 特征图、massk、pos 增加1层（更低分辨率） add one more layer (lower resolution) for feature map, mask and pos
        srcs, masks, query_pos = self.extend_featlayer(query_feats, query_pos)
        support_feats = self.backbone.forward_supp_branch(samples_s, self.args.support_out_indices)
        prompt_tensors = support_feats
        nested_prompt_tensrors = nested_tensor_from_tensor_list(prompt_tensors)
        prompt_tensors, prompt_masks = nested_prompt_tensrors.decompose()
        #从(batchsize,h,w,cls)->(batchsize,cls)(swin as support backbone)
        # prompt_tensors = torch.max(torch.max(prompt_tensors,dim=3).values,dim=2).values
        # prompt_masks = torch.max(torch.max(prompt_masks,dim=2).values,dim=1).values
        prompt_tensors = prompt_tensors.permute(0,2,1)
        prompt_tensors = self.feat_map(prompt_tensors)
        prompt_masks = ~prompt_masks
        #print(prompt_tensors.shape,prompt_masks.shape)
        prompt_dict = {
            'encoded_support':prompt_tensors,
            'support_token_mask':prompt_masks,
            'position_ids':None,
            'support_self_attention_masks':None}



        # 核心模型 CORE OF THE MODEL
        hs, reference = self.transformer(srcs, masks, query_pos, prompt_dict)

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        if self.args.two_stage_type == 'standard':
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, prompt_dict)for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                    #layer_cls_embed(F.normalize(layer_hs,dim=-1), prompt_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
        elif self.args.two_stage_type == 'no':
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs)for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
        else:
            raise NotImplementedError

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        return out

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def extend_featlayer(self, features, poss):

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])

                #m = samples.mask
                #mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # todo
                mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                #pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                #todo
                pos_l = self.pos_emb(NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        return srcs, masks, poss

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]