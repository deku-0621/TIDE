import torch
import os
from torch.utils.data import DataLoader
from models.TIDE_with_backbone import TIDE
from config import train_config
from dataset.mydataset import MyDataset
from util.misc import nested_tensor_from_tensor_list
import config
from util.slconfig import SLConfig
from models.position_encoding import build_position_encoding
from models.transformer import build_transformer
from models.matcher import HungarianMatcher
from models.loss import SetCriterion

def build_model():
    model_config_path = os.path.join(os.path.dirname(config.__file__), 'train_config.py')
    args = SLConfig.fromfile(model_config_path)


    args.use_checkpoint = False #True速度太慢
    args.num_feature_levels = train_config.num_feature_levels
    args.max_support_len = train_config.max_support_len
    #args.num_queries = 100


    position_embedding = build_position_encoding(args)
    transformer = build_transformer(args)
    model = TIDE(position_embedding, transformer,args)
    return model

def build_loss():

    dec_layers = 6 #解码器层数
    #weight_dict = {'loss_bbox': 5, 'loss_ce': 3, 'loss_giou': 1}
    weight_dict = {'loss_bbox': 2, 'loss_ce': 5, 'loss_giou': 1}

    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    matcher = HungarianMatcher(cost_class=weight_dict['loss_ce'],
                               cost_bbox=weight_dict['loss_bbox'],
                               cost_giou=weight_dict['loss_giou'])

    losses = ['labels', 'boxes']
    eos_coef = 0.1
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=eos_coef, losses=losses)

    return criterion
if __name__ == '__main__':

    #数据集
    ds = MyDataset(train_config.train_ann_path, train_config.train_image_folder, train_config.dino_feats_folder,train_config.swin_feats_folder, train=False)

    #样本生成器
    sampler_train = torch.utils.data.RandomSampler(ds)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, train_config.batch_size, drop_last=True)
    data_loader = DataLoader(ds, batch_sampler=batch_sampler_train, collate_fn=collate_fn_TIDE)

    #构建模型
    device = 'cuda'
    model = build_model().to(device)
    criterion = build_loss().to(device)

    #批量推理
    for image_feats,prompt_feats,targets in data_loader:

        image_feats = [featmap.to(device) for featmap in image_feats]
        prompt_feats = prompt_feats.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(image_feats, prompt_feats)

        loss_dict = criterion(outputs, targets)
        print(loss_dict)
        #break