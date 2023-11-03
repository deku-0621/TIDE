from torch import nn
import torch
import torch.nn.functional as F
import copy
from util.misc import NestedTensor,inverse_sigmoid
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss
from config import train_config

class TIDE(nn.Module):
    def __init__(self,position_embedding,transformer,args):
        super().__init__()

        self.pos_emb = position_embedding
        self.transformer = transformer

        self.hidden_dim = transformer.d_model
        self.num_feature_levels = train_config.num_feature_levels

        self.backbone_num_channels = [192, 384, 768][4-self.num_feature_levels:]
        self.input_proj = self.__build_input_proj__()

        #在ground-dino中用于对文本特征降维，现用于对dino特征降维，都是384->256
        self.feat_map = nn.Linear(train_config.prompt_feat_dim, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        self.args = args
        #head子模块
        if self.args.two_stage_type == 'standard':
            _class_embed = ContrastiveEmbed(args.max_support_len)
        else:
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

    def forward(self, image_feats, prompt_feats):

        features = image_feats

        #特征图位置编码
        poss = []
        for feat in image_feats:
            poss.append(self.pos_emb(feat))

        #准备提示字典
        prompt_tensors,prompt_masks = prompt_feats.decompose()
        prompt_tensors = prompt_tensors.permute(0, 2, 1)
        prompt_tensors = self.feat_map(prompt_tensors)
        prompt_masks = ~prompt_masks
        prompt_dict = {
            'encoded_support':prompt_tensors,
            'support_token_mask':prompt_masks,
            'position_ids':None,
            'support_self_attention_masks':None}

        #特征图、massk、pos 增加1层（更低分辨率）
        srcs, masks, poss = self.extend_featlayer(features, poss)

        # 核心模型
        hs, reference = self.transformer(srcs, masks, poss, prompt_dict)

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
                    #todo 6.6
                    #layer_cls_embed(F.normalize(layer_hs,dim=-1), prompt_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )
        else:
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                    # todo 6.6
                    # layer_cls_embed(F.normalize(layer_hs,dim=-1), prompt_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )


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