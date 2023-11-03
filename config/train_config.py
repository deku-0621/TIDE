modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
support_out_indices = [1]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048 #2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900#900#900
query_dim = 4
num_patterns = 0
num_feature_levels = 2 #1:3
assert num_feature_levels in [2,3,4]
enc_n_points = 4
dec_n_points = 4
frozen_stages = -1
freeze_support_backbone = True
backbone_dir = '/media/pc/works/grounding-dino-modified/weights/swin_T_224_1k.pth'
dino_pretrained_path = '/media/pc/works/grounding-dino-modified/weights/dino_deitsmall16_pretrain_full_checkpoint.pth'
two_stage_type = "standard"#"standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_support_len = 256 #256
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_support_image_cross_attention =True
support_dropout = 0.01
fusion_dropout = 0.5
fusion_droppath = 0.1
sub_sentence_present = False
device = 'cuda'

import os
VERSION = '20230910_N2_CLS5BOX3GIOU1_unfrozen_backbone_fixed_diff_cls'
def run_as_local():
    # name = os.popen("hostname").read()
    # if 'lyn-laptop\n' == name:
    #     return True
    # else:
    #     return False
    return True

def get_dino_feat_fetcher():
    '''
    构建dino特征提取器
    '''
    from BackboneFeatureExtraction.FacebookDino.feat_fetcher import ViTFeatFetcher
    dino_path = os.path.join(pretrained_model_folder, 'dino_deitsmall16_pretrain_full_checkpoint.pth')
    fetcher = ViTFeatFetcher(dino_path, 'cuda')

    return fetcher

def get_swin_feat_fetcher():
    '''
    构建swin特征提取器
    '''
    from BackboneFeatureExtraction.swin.swin_feat_fetcher import SwinFeatFetcher
    swin_model_path = os.path.join(pretrained_model_folder, 'swin_T_224_1k.pth')
    fetcher = SwinFeatFetcher(swin_model_path)

    return fetcher


def get_swin_feat_fetcher_new():
    """
    构建swin特征提取器
    """
    from BackboneFeatureExtraction.backbone.backbone import build_backbone
    from util.slconfig import SLConfig

    config_args = SLConfig.fromfile('/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/config/train_config.py')
    fetcher = build_backbone(config_args)

    return fetcher

if run_as_local():
    pretrained_model_folder = '/media/pc/works/grounding-dino-modified/weights/'
    support_feats_folder = '/media/pc/works/grounding-dino-modified/main/coco_support_feats_val2017'
    dino_feats_folder = '/media/pc/works/grounding-dino-modified/main/coco_dino_feats_val2017'
    dino_feats_train_folder = '/media/pc/works/grounding-dino-modified/main/coco_dino_feats_train2017'
    swin_feats_folder = '/media/pc/works/grounding-dino-modified/main/coco_swin_feats_val2017'
    swin_feats_train_folder = '/media/pc/works/grounding-dino-modified/main/coco_swin_feats_train2017'
    query_feats_folder = '/media/pc/works/grounding-dino-modified/main/coco_query_feats_val2017'
    train_ann_path = '/media/pc/works/grounding-dino-modified/main/coco/annotations/instances_train2017.json'
    train_image_folder = '/media/pc/works/grounding-dino-modified/main/coco/train2017/'
    train_ann_path_train2017 = '/media/pc/works/grounding-dino-modified/main/coco/annotations/instances_train2017.json'
    train_image_folder_train2017 = '/media/pc/works/grounding-dino-modified/main/coco/train2017/'
    test_ann_path = '/media/pc/works/grounding-dino-modified/main/coco/annotations/instances_val2017.json'
    test_model_path = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/train_output/epoch1_numlevel4_para_changed_checkpointnumlevel4_embedding.pth'
    batch_size = 4

#小目标的dino特征降低相似比对意义，48是实验值，目前尚未考虑输入图像经过尺寸对齐后的影响。
min_obj_box_size = 48*48
coco_use_all_class = False



#feat_encode_way = 'onehot'
feat_encode_way = 'embedding'
if feat_encode_way == 'embedding':
    #num_classes = 255
    num_classes = 80
    prompt_feat_dim = 384#384

if feat_encode_way == 'onehot':
    num_classes = 80
    prompt_feat_dim = num_classes+1


max_support_len = num_classes+1

train_resume_model = 'train_output/checkpoint_20230903_N2_CLS5BOX3GIOU1_unfrozen_backbone_embedding.pth'#train_output/checkpoint_2023810_N2_DIFF_SUP_FROM_QUERY_embedding.pth'
#train_resume_model = None






