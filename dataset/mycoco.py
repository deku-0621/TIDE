# -*- coding:utf-8 -*-
import json
from config import train_config

#min_bk_ratio = 0.5

'''
00 = {dict: 3} {'id': 1, 'name': 'person', 'supercategory': 'person'}
01 = {dict: 3} {'id': 2, 'name': 'bicycle', 'supercategory': 'vehicle'}
02 = {dict: 3} {'id': 3, 'name': 'car', 'supercategory': 'vehicle'}
03 = {dict: 3} {'id': 4, 'name': 'motorcycle', 'supercategory': 'vehicle'}
04 = {dict: 3} {'id': 5, 'name': 'airplane', 'supercategory': 'vehicle'}
05 = {dict: 3} {'id': 6, 'name': 'bus', 'supercategory': 'vehicle'}
06 = {dict: 3} {'id': 7, 'name': 'train', 'supercategory': 'vehicle'}
07 = {dict: 3} {'id': 8, 'name': 'truck', 'supercategory': 'vehicle'}
08 = {dict: 3} {'id': 9, 'name': 'boat', 'supercategory': 'vehicle'}
09 = {dict: 3} {'id': 10, 'name': 'traffic light', 'supercategory': 'outdoor'}
10 = {dict: 3} {'id': 11, 'name': 'fire hydrant', 'supercategory': 'outdoor'}
11 = {dict: 3} {'id': 13, 'name': 'stop sign', 'supercategory': 'outdoor'}
12 = {dict: 3} {'id': 14, 'name': 'parking meter', 'supercategory': 'outdoor'}
13 = {dict: 3} {'id': 15, 'name': 'bench', 'supercategory': 'outdoor'}
14 = {dict: 3} {'id': 16, 'name': 'bird', 'supercategory': 'animal'}
15 = {dict: 3} {'id': 17, 'name': 'cat', 'supercategory': 'animal'}
16 = {dict: 3} {'id': 18, 'name': 'dog', 'supercategory': 'animal'}
17 = {dict: 3} {'id': 19, 'name': 'horse', 'supercategory': 'animal'}
18 = {dict: 3} {'id': 20, 'name': 'sheep', 'supercategory': 'animal'}
19 = {dict: 3} {'id': 21, 'name': 'cow', 'supercategory': 'animal'}
20 = {dict: 3} {'id': 22, 'name': 'elephant', 'supercategory': 'animal'}
21 = {dict: 3} {'id': 23, 'name': 'bear', 'supercategory': 'animal'}
22 = {dict: 3} {'id': 24, 'name': 'zebra', 'supercategory': 'animal'}
23 = {dict: 3} {'id': 25, 'name': 'giraffe', 'supercategory': 'animal'}
24 = {dict: 3} {'id': 27, 'name': 'backpack', 'supercategory': 'accessory'}
25 = {dict: 3} {'id': 28, 'name': 'umbrella', 'supercategory': 'accessory'}
26 = {dict: 3} {'id': 31, 'name': 'handbag', 'supercategory': 'accessory'}
27 = {dict: 3} {'id': 32, 'name': 'tie', 'supercategory': 'accessory'}
28 = {dict: 3} {'id': 33, 'name': 'suitcase', 'supercategory': 'accessory'}
29 = {dict: 3} {'id': 34, 'name': 'frisbee', 'supercategory': 'sports'}
30 = {dict: 3} {'id': 35, 'name': 'skis', 'supercategory': 'sports'}
31 = {dict: 3} {'id': 36, 'name': 'snowboard', 'supercategory': 'sports'}
32 = {dict: 3} {'id': 37, 'name': 'sports ball', 'supercategory': 'sports'}
33 = {dict: 3} {'id': 38, 'name': 'kite', 'supercategory': 'sports'}
34 = {dict: 3} {'id': 39, 'name': 'baseball bat', 'supercategory': 'sports'}
35 = {dict: 3} {'id': 40, 'name': 'baseball glove', 'supercategory': 'sports'}
36 = {dict: 3} {'id': 41, 'name': 'skateboard', 'supercategory': 'sports'}
37 = {dict: 3} {'id': 42, 'name': 'surfboard', 'supercategory': 'sports'}
38 = {dict: 3} {'id': 43, 'name': 'tennis racket', 'supercategory': 'sports'}
39 = {dict: 3} {'id': 44, 'name': 'bottle', 'supercategory': 'kitchen'}
40 = {dict: 3} {'id': 46, 'name': 'wine glass', 'supercategory': 'kitchen'}
41 = {dict: 3} {'id': 47, 'name': 'cup', 'supercategory': 'kitchen'}
42 = {dict: 3} {'id': 48, 'name': 'fork', 'supercategory': 'kitchen'}
43 = {dict: 3} {'id': 49, 'name': 'knife', 'supercategory': 'kitchen'}
44 = {dict: 3} {'id': 50, 'name': 'spoon', 'supercategory': 'kitchen'}
45 = {dict: 3} {'id': 51, 'name': 'bowl', 'supercategory': 'kitchen'}
46 = {dict: 3} {'id': 52, 'name': 'banana', 'supercategory': 'food'}
47 = {dict: 3} {'id': 53, 'name': 'apple', 'supercategory': 'food'}
48 = {dict: 3} {'id': 54, 'name': 'sandwich', 'supercategory': 'food'}
49 = {dict: 3} {'id': 55, 'name': 'orange', 'supercategory': 'food'}
50 = {dict: 3} {'id': 56, 'name': 'broccoli', 'supercategory': 'food'}
51 = {dict: 3} {'id': 57, 'name': 'carrot', 'supercategory': 'food'}
52 = {dict: 3} {'id': 58, 'name': 'hot dog', 'supercategory': 'food'}
53 = {dict: 3} {'id': 59, 'name': 'pizza', 'supercategory': 'food'}
54 = {dict: 3} {'id': 60, 'name': 'donut', 'supercategory': 'food'}
55 = {dict: 3} {'id': 61, 'name': 'cake', 'supercategory': 'food'}
56 = {dict: 3} {'id': 62, 'name': 'chair', 'supercategory': 'furniture'}
57 = {dict: 3} {'id': 63, 'name': 'couch', 'supercategory': 'furniture'}
58 = {dict: 3} {'id': 64, 'name': 'potted plant', 'supercategory': 'furniture'}
59 = {dict: 3} {'id': 65, 'name': 'bed', 'supercategory': 'furniture'}
60 = {dict: 3} {'id': 67, 'name': 'dining table', 'supercategory': 'furniture'}
61 = {dict: 3} {'id': 70, 'name': 'toilet', 'supercategory': 'furniture'}
62 = {dict: 3} {'id': 72, 'name': 'tv', 'supercategory': 'electronic'}
63 = {dict: 3} {'id': 73, 'name': 'laptop', 'supercategory': 'electronic'}
64 = {dict: 3} {'id': 74, 'name': 'mouse', 'supercategory': 'electronic'}
65 = {dict: 3} {'id': 75, 'name': 'remote', 'supercategory': 'electronic'}
66 = {dict: 3} {'id': 76, 'name': 'keyboard', 'supercategory': 'electronic'}
67 = {dict: 3} {'id': 77, 'name': 'cell phone', 'supercategory': 'electronic'}
68 = {dict: 3} {'id': 78, 'name': 'microwave', 'supercategory': 'appliance'}
69 = {dict: 3} {'id': 79, 'name': 'oven', 'supercategory': 'appliance'}
70 = {dict: 3} {'id': 80, 'name': 'toaster', 'supercategory': 'appliance'}
71 = {dict: 3} {'id': 81, 'name': 'sink', 'supercategory': 'appliance'}
72 = {dict: 3} {'id': 82, 'name': 'refrigerator', 'supercategory': 'appliance'}
73 = {dict: 3} {'id': 84, 'name': 'book', 'supercategory': 'indoor'}
74 = {dict: 3} {'id': 85, 'name': 'clock', 'supercategory': 'indoor'}
75 = {dict: 3} {'id': 86, 'name': 'vase', 'supercategory': 'indoor'}
76 = {dict: 3} {'id': 87, 'name': 'scissors', 'supercategory': 'indoor'}
77 = {dict: 3} {'id': 88, 'name': 'teddy bear', 'supercategory': 'indoor'}
78 = {dict: 3} {'id': 89, 'name': 'hair drier', 'supercategory': 'indoor'}
79 = {dict: 3} {'id': 90, 'name': 'toothbrush', 'supercategory': 'indoor'}

'''

unseen_clses= [8,9,15,22,23,24,25,33,41,42,43,51,59,60,61,70,77,82,89,90]
#unseen_clses = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 40, 57, 58, 59, 61, 63]
class CocoAnn:
    def __init__(self,ann_path,be_train):

        self.dict_catid_anns = {}
        self.anns = []
        self.dict_imgid_anns = {} #每个图像所含objs
        #self.image_ids = [] #2023.5.16 新增
        self.dict_imgid_filename = {}

        #从标注文件填充 ...
        with open(ann_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)

        #5.25 cateid对应clsid
        cat_ids = [x['id'] for x in json_data['categories']]
        self.dict_catid_clsid = {}
        for n,cat_id in enumerate (cat_ids):
            self.dict_catid_clsid[cat_id] = n
        #clsid对应cateid


        # 5.26 clsid 对应catename
        self.dict_clsid_catename = {}
        cat_names = [x['name'] for x in json_data['categories']]
        for n,cat_name in enumerate (cat_names):
            self.dict_clsid_catename[n] = cat_name

        for n,item in enumerate(json_data['images']):
            imgid = item['id']
            self.dict_imgid_filename[imgid] = item['file_name']
            #self.image_ids.append(imgid)

        annotations = json_data['annotations']
        print('annotations count:',len(annotations))

        for n,ann in enumerate(annotations):

            if ann['iscrowd'] != 0: continue

            if n % 10000==0:print('{}/{}'.format(n,len(annotations)))

            bbox = list(map(int, ann['bbox']))

            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            polygons = ann['segmentation']
            catid = ann['category_id']
            imgid = ann['image_id']

            filename = self.dict_imgid_filename[imgid]

            if box_w < 16 or box_h < 16:
                continue

            if box_w*box_h < train_config.min_obj_box_size:
                continue

            if not train_config.coco_use_all_class:
                # todo 区分训练类和测试类
                if be_train:
                    if catid in unseen_clses:
                        continue
                else:
                    if catid not in unseen_clses:
                        continue


            #登记
            obj_ann = {'filename': filename, 'bbox': bbox, 'polygons': polygons, 'catid': catid, 'id': ann['id'],'imgid':imgid}
            self.anns.append(obj_ann)

            if catid not in self.dict_catid_anns:
                self.dict_catid_anns[catid] = [obj_ann]
            else:
                self.dict_catid_anns[catid].append(obj_ann)

            if imgid not in self.dict_imgid_anns:
                self.dict_imgid_anns[imgid] = [obj_ann]
            else:
                self.dict_imgid_anns[imgid].append(obj_ann)


if __name__ == '__main__':
    ann_path = train_config.train_ann_path
    ann = CocoAnn(ann_path, True)
    print('obj count:',len(ann.anns),'img count:',len(ann.dict_imgid_anns))