import pickle

import cv2
import numpy as np
from torch.utils.data import dataset
import os
from dataset.mycoco import CocoAnn
from config import train_config
from util.myutils import *
import random
import util.transforms as T
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
import glob
def load_image(img_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            #T.RandomResize([1000], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed
def load_image_support(img_bgr: np.ndarray) -> torch.Tensor:
    #b,c,s,h = random.uniform(0,0.5),random.uniform(0,0.5),random.uniform(0,0.5),random.uniform(0,0.5)
    transform = T.Compose(
        [
            T.RandomColorJitter(0.7),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed
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
class MyDataset(dataset.Dataset):

    def __init__(self,ann_path,image_folder,train):
        super(MyDataset, self).__init__()
        self.same_class_folder = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/dataset/coco_split'
        self.image_folder = image_folder
        self.coco = CocoAnn(ann_path, train)
        self.image_ids = [k for k,v in self.coco.dict_imgid_anns.items()]
        self.same_class = True
        self.be_train = train
        self.model = 'pos_nag'
        if not train:
            self.index_count = 0
        self.resample_count = 0
    def __test_index__(self):

        #测试模式下，使用确定的采样顺序
        index = self.index_count
        anns_count = self.__len__()
        if self.resample_count == 0:
            step = 1
        else:
            step = anns_count // self.resample_count
        self.index_count += step
        self.index_count = self.index_count % anns_count
        return index
    def __len__(self):
        return len(self.image_ids)

    def build_targets(self, ann, w, h):
        tag_bboxes = []
        pos_catids = []
        # 收集目标box和特征
        for n, ann_obj in enumerate(ann):
            tag_bboxes.append(xyxy_2_xywh(ann_obj['bbox'], w, h))
            pos_catids.append(ann_obj['catid'])
        #根据GLIP:Prompt design for detection data.
        if random.randint(0, 1) == 0:
            num_nag_cat = train_config.max_support_len - len(tag_bboxes)
        else:
            num_nag_cat = random.randint(1, train_config.max_support_len - len(tag_bboxes))
        if self.model == 'pos_nag':
            num_all_cat = num_nag_cat+len(tag_bboxes)
        else:
            num_all_cat = len(tag_bboxes)+1
        tag_labels = random.sample(range(1, num_all_cat), len(tag_bboxes))
        return tag_labels, tag_bboxes,num_all_cat,pos_catids
    def __getitem__(self,index):

        if not self.be_train:
            #测试模式下，使用确定的采样顺序
            index = self.__test_index__()

        #索引定位一个图像
        imgid = self.image_ids[index]
        # if self.same_class:
        #     cat_ann_all_for_query = [self.coco.dict_catid_anns[int(self.coco.dict_imgid_anns[imgid][index]['catid'])] for index in range(len(self.coco.dict_imgid_anns[imgid]))]
        #     ann = self.coco.dict_imgid_anns[imgid]
        #     filename = self.coco.dict_imgid_filename[imgid]
        # else:
        #定位当前图像的目标标注和图像文件名
        ann = self.coco.dict_imgid_anns[imgid]
        filename = self.coco.dict_imgid_filename[imgid]
        #根据图像文件名，读取特征文件。
        query_np = cv2.imread(os.path.join(self.image_folder, filename))
        h, w, c = query_np.shape
        # =====AUGMENTATION=====
        ann_diff = random.choice(self.coco.anns)
        query_np_diff = cv2.imread(os.path.join(self.image_folder, ann_diff['filename']))
        h_diff,w_diff,c_diff = query_np_diff.shape
        if h_diff <h or w_diff < w:
            query_np_diff = cv2.resize(query_np_diff,(w,h))
        # CUT RANDOM REGION
        cut_ratio = 0.3
        cut_x = random.randint(0, int(w * (1 - cut_ratio)))
        cut_y = random.randint(0, int(h * (1 - cut_ratio)))
        cut_width = int(w * cut_ratio)
        cut_height = int(h * cut_ratio)
        # EXTRACT REGION
        cut_region = query_np_diff[cut_y:cut_y+cut_height, cut_x:cut_x+cut_width]
        # MIX
        query_np[cut_y:cut_y+cut_height, cut_x:cut_x+cut_width] = cut_region
        cv2.imshow('query',query_np)
        cv2.waitKey(0)
        query_img = load_image(query_np)
        #todo
        tag_labels, tag_bboxes,num_all_cate,pos_cate_ids = self.build_targets(ann, w, h)
        tag_bboxes = torch.Tensor(tag_bboxes)
        tag_labels = torch.LongTensor(tag_labels)
        target = {'boxes': tag_bboxes, 'labels': tag_labels}
        prompt_np = []
        pos_prompt_np = []
        no_object_img_np = np.zeros((128,128,3),dtype=np.uint8)
        prompt_np.append(load_image(no_object_img_np))
        if self.same_class:
            for each in ann:
                same_class_support_filename_path = os.path.join(self.same_class_folder, str(each['catid']))
                #randomly choose a support image
                support_img_path = random.choice(glob.glob(os.path.join(same_class_support_filename_path,'*')))
                CV2_support = cv2.imread(support_img_path)
                # cv2.imshow('query', query_np)
                # cv2.imshow('support',CV2_support)
                # cv2.waitKey(0)8
                # cv2.destroyAllWindows()
                PIL_support = load_image_support(cv2.resize(CV2_support,(128,128)))
                pos_prompt_np.append(PIL_support)
        else:
            for each in ann:
                box = each['bbox']
                box = box_min_size_norm(box, w, h, 16)
                x1, y1, x2, y2 = box
                temp_support = load_image_support(cv2.resize(query_np[y1:y2, x1:x2], (128, 128)))
                # print('support size:',temp.shape)
                # support_sample = nested_tensor_from_tensor_list(temp_support)
                pos_prompt_np.append(temp_support)
                # cv2.imshow('query_np', query_np)
                # cv2.imshow('prompt_np', support_img[y1:y2, x1:x2])
                #
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        while len(prompt_np)<num_all_cate:
            ann = random.choice(self.coco.anns)
            if ann['catid'] in pos_cate_ids:
                continue
            nag_img = cv2.imread(os.path.join(self.image_folder, ann['filename']))
            nag_prompt_img = load_image(cv2.resize(nag_img[ann['bbox'][1]:ann['bbox'][3],ann['bbox'][0]:ann['bbox'][2]],(128,128)))
            prompt_np.append(nag_prompt_img)
        for n,npy in enumerate(pos_prompt_np):
            prompt_np[tag_labels[n]] = npy
        sample = []
        sample.append(query_img)
        prompt_tensor = torch.stack(prompt_np).permute(1,2,3,0)
        sample.append(prompt_tensor)
        #sample.append(prompt_tensor)
        sample.append(target)
        #info = {'height': h, 'width': w, 'filename': filename, 'image_id': imgid, 'ann': ann}
        #sample.append(info)
        return tuple(sample)
def collate_fn_TIDE(batch):
    batch = list(zip(*batch))
    query_sample,support_sample,targets = batch
    query_sample = nested_tensor_from_tensor_list(query_sample)
    support_samples = nested_tensor_from_tensor_list(support_sample)
    return tuple([query_sample,support_samples,targets])
if __name__ == '__main__':
    ds = MyDataset(train_config.train_ann_path, train_config.train_image_folder,train=True)
    sampler_train = torch.utils.data.RandomSampler(ds)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 8, drop_last=True)
    data_loader = DataLoader(ds, batch_sampler=batch_sampler_train,collate_fn=collate_fn_TIDE,num_workers=0)
    for a,b,c in data_loader:
        print(a,b,c)
    for sample in data_loader:
        print(len(sample))



