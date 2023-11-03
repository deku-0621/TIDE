# -*- coding:utf-8 -*-
#2023.4.6 从coco生成swin feats,保存文件。

import pickle
import os
import cv2
from dataset.mycoco  import CocoAnn
from config import train_config


def gen_feats_from_coco():
    '''
    按coco类别保存swin特征
    '''

    swin_fetcher = train_config.get_swin_feat_fetcher()

    # 构建mycoco
    ann_path = train_config.train_ann_path_train2017
    data_folder = train_config.train_image_folder_train2017
    coco = CocoAnn(ann_path, True)

    image_count = len(coco.dict_imgid_filename)
    counter = 0

    for imgid,filename in coco.dict_imgid_filename.items():

        img_path = os.path.join(data_folder, filename)

        query_np = cv2.imread(img_path)
        feats = swin_fetcher.get_feats(query_np)

        #提取器得到的是多层特征tensor组成的列表，需要转为numpy列表，再保存。
        np_feats = []
        for feat in feats:
            np_feats.append(feat.cpu().numpy())

        #保存文件名和原图像名相同
        save_filename = filename.split('.')[0] + '.pkl'
        save_path = os.path.join(train_config.swin_feats_train_folder,save_filename)
        with open(save_path,'wb') as f:
            pickle.dump(np_feats,f)

        counter += 1
        print('{}/{}'.format(counter,image_count))



if __name__ == '__main__':

    gen_feats_from_coco()
    pass

