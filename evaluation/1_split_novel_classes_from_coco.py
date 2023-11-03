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
def filter_coco(coco, cls_split):
    new_anns = []
    all_cls_dict = {}
    for img_id, id in enumerate(coco.imgs):
        img = coco.loadImgs(id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        skip_flag = False
        img_cls_dict = {}
        if len(anns) == 0:
            continue
        for ann in anns:
            segmentation = ann['segmentation']
            area = ann['area']
            iscrowd = ann['iscrowd']
            image_id = ann['image_id']
            bbox = ann['bbox']
            category_id = ann['category_id']
            id = ann['id']
            bbox_area = bbox[2] * bbox[3]

            # filter images with small boxes
            if category_id in cls_split:
                if iscrowd == 1:
                    skip_flag = True
                if bbox_area < 32 * 32:
                    skip_flag = True

        if skip_flag:
            continue
        else:
            for ann in anns:
                category_id = ann['category_id']
                if category_id in cls_split:
                    new_anns.append(ann)

                    if category_id in all_cls_dict.keys():
                        all_cls_dict[category_id] += 1
                    else:
                        all_cls_dict[category_id] = 1

    print(len(new_anns))
    print(sorted(all_cls_dict.items(), key=lambda kv: (kv[1], kv[0])))
    return new_anns

def main(instances_name):
    root_path = ''
    print(root_path)
    dataDir = './'
    support_dict = {}

    support_dict['support_box'] = []
    support_dict['category_id'] = []
    support_dict['image_id'] = []
    support_dict['id'] = []
    support_dict['file_path'] = []

    #novel_class_ids = [8,9,15,22,23,24,25,33,41,42,43,51,59,60,61,70,77,82,89,90]
    novel_class_ids = [1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 17, 18, 19, 20, 40, 57, 58, 59, 61, 63]
    #novel_class_ids = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
    for dataType in [instances_name]:  # , 'split_voc_instances_train2017.json']:
        annFile = join(dataDir, dataType)

        with open(annFile, 'r') as load_f:
            dataset = json.load(load_f)
            print(dataset.keys())
            #save_info = dataset['info']
            #save_licenses = dataset['licenses']
            save_images = dataset['images']
            save_categories = dataset['categories']
            save_annotations = dataset['annotations']

        novel_split = novel_class_ids
        #根据类别split annotations
        categories_novel_split = []
        for each in save_categories:
            if each['id'] in novel_split:
                categories_novel_split.append(each)
        id_novel_split = [c['id'] for c in categories_novel_split]
        print('Novel Split : {} classes'.format(len(categories_novel_split)))
        for c in categories_novel_split:
            print('\t', c['name'])
        coco = COCO(annFile)

        annotations_novel_split = filter_coco(coco, id_novel_split)

        dataset_novel_split = {
            #'info': save_info,
            #'licenses': save_licenses,
            'images': save_images,
            'annotations': annotations_novel_split,
            'categories': save_categories}
        new_annotations_path = os.path.join(root_path, 'new_annotations')

        if not os.path.exists(new_annotations_path):
            os.makedirs(new_annotations_path)
        novel_split_json = os.path.join(root_path, f'new_annotations/final_split_novel_{instances_name}')

        with open(novel_split_json, 'w') as f:
            json.dump(dataset_novel_split, f)
if __name__ == '__main__':
    main(instances_name='voc2012_val.json')
