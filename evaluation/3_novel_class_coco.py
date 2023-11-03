#ZladWu

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
import sys
def calculate_iou(box1,box2):
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[0]+box1[2],box2[0]+box2[2])
    y2 = min(box1[1]+box1[3],box2[1]+box2[3])
    if x1 > x2 or y1 > y2:
        return 0
    else:
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        area = (x2-x1) * (y2-y1)
        return area / (area1 + area2 - area)
def main(shot):
    ann_file = 'new_annotations/final_split_novel_instances_train2017.json'
    with open(ann_file,'r') as load_f:
        dataset = json.load(load_f)
        save_info = dataset['info']
        save_licenses = dataset['licenses']
        save_images = dataset['images']
        save_categories = dataset['categories']
    save_annotations = []
    #打乱dataset['annotations']的顺序
    import random

    #过滤掉小目标
    dataset['annotations'] = [ann for ann in dataset['annotations'] if ann['bbox'][2] > 150 and ann['bbox'][3] > 150]
    #过滤掉iscrowd
    dataset['annotations'] = [ann for ann in dataset['annotations'] if ann['iscrowd'] == 0]
    random.seed(2023819)
    random.shuffle(dataset['annotations'])
    for s in range(shot):
        #novel_class_ids = [9, 8, 22, 15, 23, 24, 25, 33, 41, 42, 43, 51, 59, 60, 61, 70, 77, 82, 89, 90]
        novel_class_ids = [1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 17, 18, 19, 20, 40, 57, 58, 59, 61, 63]
        for each in dataset['annotations']:
            if each['category_id'] in novel_class_ids and each not in save_annotations:
                box1 = each['bbox']
                has_over_lap = False
                for each2 in dataset['annotations']:
                    if each2 != each:
                        box2 = each2['bbox']
                        iou = calculate_iou(box1,box2)
                        if iou>0.9:
                            has_over_lap = True
                            break
                if not has_over_lap:
                    save_annotations.append(each)
                    #print('w,h:',each['bbox'][2],each['bbox'][3])
                    novel_class_ids.remove(each['category_id'])
    print(len(save_annotations))


    dataset_split = {
        'info': save_info,
        'licenses': save_licenses,
        'images': save_images,
        'annotations': save_annotations,
        'categories': save_categories
    }
    split_file = './new_annotations/final_split_novel_{}_shot_instances_train2017.json'.format(shot)
    with open(split_file, 'w') as f:
        json.dump(dataset_split, f)
if __name__ == '__main__':
    for shot in [1,2,3,5,10]:
        main(shot)
    pass

