# -*- coding:utf-8 -*-
#2023.4.6 为了在loss中添加“置信度与相似度的距离“，需要用dino提取coco数据集中所有目标的特征向量，存储到文件。

import os
import pickle
import cv2
import numpy as np
import itertools
from BackboneFeatureExtraction.FacebookDino.feat_fetcher import ViTFeatFetcher
from dataset.mycoco  import CocoAnn
from util.myutils import get_feats
from config import train_config
from torchvision import transforms as T
#dino特征文件所在路径
dino_feats_folder = train_config.dino_feats_train_folder

#构建特征提取器
feat_fetcher = train_config.get_dino_feat_fetcher()

def box_min_size_norm(xyxy,w,h,min_size):
    '''
    保证box的宽高都不小于min_size
    '''
    assert w>min_size and h>min_size

    box_h = xyxy[3] - xyxy[1]
    box_w = xyxy[2] - xyxy[0]

    if box_h < min_size:
        if xyxy[1] + min_size < h:
            xyxy[3] = xyxy[1] + min_size
        else:
            xyxy[1] = xyxy[3] - min_size

    if box_w < min_size:
        if xyxy[0] + min_size < w:
            xyxy[2] = xyxy[0] + min_size
        else:
            xyxy[0] = xyxy[2] - min_size

    return xyxy

def gen_feat_by_cate():
    '''
    按coco类别保存dino特征
    '''

    # 构建mycoco
    ann_path = train_config.train_ann_path_train2017
    data_folder = train_config.train_image_folder_train2017
    coco = CocoAnn(ann_path, True)

    ann_count = len(coco.anns)
    counter = 0

    for catid,anns in coco.dict_catid_anns.items():

        #为按类别保存特征，创建和类别对应的文件夹
        cat_folder = os.path.join(dino_feats_folder,str(catid))
        if not os.path.exists(cat_folder):
            os.mkdir(cat_folder)

        for ann in anns:

            img_path = os.path.join(data_folder, ann['filename'])
            bbox = ann['bbox']
            id = ann['id']

            img_np = cv2.imread(img_path)
            h,w,c = img_np.shape
            #为满足dino的patch对齐，如果box宽或高小于16，调整其尺寸为16。
            bbox = box_min_size_norm(bbox,w,h,16)
            #按边框截取目标
            obj_np = img_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # 数据增强
            # obj_np = T.ToPILImage()(obj_np)
            # obj_np = T.RandomHorizontalFlip()(obj_np)
            # obj_np = T.RandomVerticalFlip()(obj_np)
            # obj_np = T.RandomRotation(90)(obj_np)
            # obj_np = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(obj_np)
            # obj_np = T.ToTensor()(obj_np)
            # obj_np = obj_np.transpose(2,0)
            # obj_np = obj_np.numpy()
            # cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.imshow('img_np', obj_np)
            cv2.waitKey(0)



            #feat = get_feats(feat_fetcher, [obj_np])[0]
            #feat_save_path = os.path.join(cat_folder,'{}.npy'.format(id))
            #np.save(feat_save_path,feat)

            counter += 1
            print('{}/{}'.format(counter,ann_count))

def list_feats_by_cate(catid):

    feats = []
    cat_folder = os.path.join(dino_feats_folder, str(catid))
    for file in os.listdir(cat_folder):
        path = os.path.join(cat_folder,file)
        feat = np.load(path)
        feats.append(feat)

    return feats

def test_feat_compre():
    '''
    试验同类、异类特征之间的相似度
    '''
    cat_a_feats = list_feats_by_cate(6)
    cat_b_feats = list_feats_by_cate(6)

    feats_comb_lst = list(itertools.product(cat_a_feats,cat_b_feats))

    sim_lst = []
    for a,b in feats_comb_lst:
        sim = np.dot(a,b)
        sim_lst.append(sim)

    sim_max = np.max(sim_lst)
    sim_min = np.min(sim_lst)
    sim_mean = np.mean(sim_lst)

    print(sim_max,sim_min,sim_mean)

    a = np.where(np.array(sim_lst) > 0.5)

    pass

def test_for_sim():

    folder = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/tests/input'

    path_a = os.path.join(folder, 'gc2.jpg')
    img_a = cv2.imread(path_a)
    feat_a = get_feats(feat_fetcher, [img_a])[0]

    # path_b = os.path.join(folder, '2.jpg')
    # img_b = cv2.imread(path_b)
    # feat_b = get_feats(feat_fetcher, [img_b])[0]
    no_obj_feat = np.ones(384)/np.linalg.norm(np.ones(384))
    #random noise
    feat_b = no_obj_feat+np.random.normal(0,0.1,384)
    sim = np.dot(feat_a,feat_b)
    print(sim)

if __name__ == '__main__':
    #gen_feat_by_cate()
    #test_feat_compre()
    test_for_sim()
    pass

