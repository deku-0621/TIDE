import random

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import shutil
import argparse
import sys
def main(shot,poly,seed):
    dataDir = ''

    #root_path = '/home/fanqi/data/COCO'
    root_path = ''
    support_path = os.path.join('train_supports', '{}_shot_support'.format(shot))
    #support_path = '10_shot_support'
    if not isdir(support_path): 
        mkdir(support_path)
    #else:
    #    shutil.rmtree(support_path)

    support_dict = {}
    
    support_dict['support_box'] = []
    support_dict['category_id'] = []
    support_dict['image_id'] = []
    support_dict['id'] = []
    support_dict['file_path'] = []
    support_dict['segmentation'] = []

    for dataType in ['train2017']: #, 'train2017']:
        set_crop_base_path = join(support_path, dataType)
        set_img_base_path = join(dataDir, dataType)

        # other information
        #annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
        #annFile = './new_annotations/final_split_voc_10_shot_instances_train2017.json'
        
        annFile = os.path.join(root_path, 'new_annotations/final_split_novel_{}_shot_instances_train2017.json'.format(shot))
        
        with open(annFile,'r') as load_f:
            dataset = json.load(load_f)
            print(dataset.keys())
            save_info = dataset['info']
            save_licenses = dataset['licenses']
            save_images = dataset['images']
            save_categories = dataset['categories']

        coco = COCO(annFile)
        import random
        random.seed(seed)
        random.shuffle(dataset['annotations'])
        for img_id, id in enumerate(coco.imgs):
            img = coco.loadImgs(id)[0]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))

            if len(anns) == 0:
                continue

            #print(img['file_name'])
            frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])
            if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)
            im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))
            #cv2.imshow('im', im)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            # if im is None:
            #     print('{}/{}'.format(set_img_base_path, img['file_name']))
            #     continue
            from math import floor
            for item_id, ann in enumerate(anns):
                #print(ann)
                rect = ann['bbox']
                #non crowd
                if ann['iscrowd'] == 1:
                    continue
                bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
                support_img = im[floor(bbox[1]):floor(bbox[3]), floor(bbox[0]):floor(bbox[2]), :]
                if poly:
                    w,h  = im.shape[0], im.shape[1]
                    segmentation_mask = np.zeros((w, h), dtype=np.uint8)
                    for coords in ann['segmentation']:
                        points = np.array(coords, np.int32).reshape(-1, 2)
                        cv2.fillPoly(segmentation_mask, [points], color=255)  # 设置分割颜色
                    support_img = cv2.bitwise_and(im, im, mask=segmentation_mask)
                    support_img = cv2.resize(support_img, (320, 320), interpolation=cv2.INTER_LINEAR) if support_img.shape[0] > 700 or support_img.shape[1] >700 else support_img
                # cv2.imshow('support_img', segmented_target)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                if bbox_xywh[2]<48 or bbox_xywh[3]<48:
                    support_img = cv2.resize(support_img, (64, 64), interpolation=cv2.INTER_LINEAR)
                #im_name = img['file_name'].split('.')[0] + '_' + str(item_id) + '.jpg'
                #output_dir = './fig'
                #vis_image(support_img[:, :, ::-1], support_box, join(frame_crop_base_path, '{:04d}.jpg'.format(item_id)))
                if rect[2] <= 0 or rect[3] <=0:
                    print(rect)
                    continue
                file_path = join(frame_crop_base_path, '{:04d}.jpg'.format(item_id))
                cv2.imwrite(file_path, support_img)
                #print(file_path)
                support_dict['support_box'].append(bbox_xywh)
                support_dict['category_id'].append(ann['category_id'])
                support_dict['image_id'].append(ann['image_id'])
                support_dict['id'].append(ann['id'])
                support_dict['file_path'].append(file_path)
                support_dict['segmentation'].append(ann['segmentation'])

        support_df = pd.DataFrame.from_dict(support_dict)
        
    return support_df
if __name__ == '__main__':
    since = time.time()
    poly = False
    #random number
    seed1 = random.randint(0,1000)
    seed2 = random.randint(1000,2000)
    seed3 = random.randint(2000,3000)
    seed4 = random.randint(3000,4000)
    seed5 = random.randint(4000,5000)
    seed6 = random.randint(5000,6000)
    for shot,seed in zip([1,2,3,5,10],[seed1,seed2,seed3,seed4,seed5]):
        support_df = main(shot,poly,seed)
        support_df.to_pickle("./train_seed1/{}_shot_support_df.pkl".format(shot))
        time_elapsed = time.time() - since
        print('Total complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

