import pickle
import numpy as np
from torch.utils.data import dataset
import os
from dataset.mycoco import CocoAnn
try:
   from config import train_config
except:
   import train_config
from util.myutils import *
import random
from torch.utils.data import DataLoader
import glob

class MyDataset(dataset.Dataset):

    def __init__(self,ann_path,image_folder,dino_feats_folder,swin_feats_folder,train):

        super(MyDataset, self).__init__()



        self.image_folder = image_folder
        self.support_feats_folder = dino_feats_folder
        self.swin_feats_folder = swin_feats_folder
        self.coco = CocoAnn(ann_path, train)
        self.image_ids = [k for k,v in self.coco.dict_imgid_anns.items()]

        self.be_train = train
        if not train:
            self.index_count = 0

        self.resample_count = 0
        self.cates = os.listdir(self.support_feats_folder)
        self.CateNpyDict = {}
        for each in self.cates:
            self.CateNpyDict[each] = os.listdir(os.path.join(self.support_feats_folder, each))
        print('cate-npy dict ceated')




    def set_resample_count(self,count):
        self.resample_count = count

    def get_boxes_from_ann(self,ann):
        boxes = [ann_obj['bbox'] for ann_obj in ann]
        return boxes

    def get_boxes_random(self, img_h,img_w):

        boxes = []

        box_count = 10
        counter = 0
        while counter < box_count:

            x = random.randint(0, img_w - 1)
            y = random.randint(0,img_h-1)
            x0 = random.randint(0, x)
            y0 = random.randint(0, y)
            x1 = random.randint(x, img_w - 1)
            y1 = random.randint(y, img_h-1)

            if (x1-x0)*(y1-y0) < train_config.min_obj_box_size:
                continue

            if x1-x0 < 16 or y1-y0 < 16:
                continue

            boxes.append([x0,y0,x1,y1])
            counter += 1

        return boxes

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

    def get_support_feat_from_category(self, ann_id, cat_id):
        #acquire all files in the folder
        #randomly choose one
        feat_path = random.choice(self.CateNpyDict[str(cat_id)])
        path = os.path.join(self.support_feats_folder, str(cat_id), feat_path)
        feat = np.load(path)
        return feat
    def get_support_feat_from_query(self, ann_id, cat_id):
        feat_path = os.path.join(self.support_feats_folder, str(cat_id), '{}.npy'.format(ann_id))
        feat = np.load(feat_path)
        return feat

    def get_swin_feats(self,img_filename):

        # if myconfig.run_as_local():
        #     img_filename = '000000397133.jpg'

        feat_filename = img_filename.split('.')[0] + '.pkl'
        feat_path = os.path.join(self.swin_feats_folder, feat_filename)
        with open(feat_path,'rb') as f:
            feats = pickle.load(f)

        #为了降低算力消耗，去掉第一层特征图
        if train_config.num_feature_levels == 3: feats = feats[1:]
        if train_config.num_feature_levels == 2: feats = feats[2:]

        return feats

    def __len__(self):
        return len(self.image_ids)

    def get_onehot_vec(self,id):
        vec = np.eye(train_config.num_classes + 1)[id]
        return vec
    def build_prompts_from_category(self, ann, w, h):
        pos_catids = []
        pos_prompt_feats = []
        prompt_feats = []
        tag_bboxes = []

        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        prompt_feats.append(no_obj_feat)
        # 收集目标box和特征
        for n, ann_obj in enumerate(ann):
            tag_bboxes.append(xyxy_2_xywh(ann_obj['bbox'], w, h))
            pos_prompt_feats.append(self.get_support_feat_from_category(ann_obj['id'], ann_obj['catid']))
            pos_catids.append(ann_obj['catid'])

        # 根据GLIP:Prompt design for detection data.
        if random.randint(0, 1) == 0:
            num_nag_cat = train_config.max_support_len - len(tag_bboxes)
        else:
            num_nag_cat = random.randint(1, train_config.max_support_len - len(tag_bboxes))

        num_all_cat = num_nag_cat + len(tag_bboxes)

        # 选取非目标类的特征做为背景特征提示
        while len(prompt_feats) < num_all_cat:

            ann = random.choice(self.coco.anns)
            if ann['catid'] in pos_catids:
                continue
            nag_prompt_feat = self.get_support_feat_from_category(ann['id'], ann['catid'])

            sim_max = 0
            for pos_prompt_feat in pos_prompt_feats:
                sim = np.dot(nag_prompt_feat, pos_prompt_feat)
                sim_max = max(sim, sim_max)
            if sim_max < 0.4:  # 0.4
                prompt_feats.append(nag_prompt_feat)
            # else:
            #     #add random noise to the no_obj_feat
            #     no_obj_feat_with_noise = no_obj_feat + np.random.normal(0, 0.1, 384)
            #     prompt_feats.append(no_obj_feat_with_noise)

        tag_labels = random.sample(range(1, num_all_cat), len(tag_bboxes))
        for n, feats in enumerate(pos_prompt_feats):
            prompt_feats[tag_labels[n]] = feats

        return prompt_feats, tag_labels, tag_bboxes
    def build_prompts_from_query(self, ann, w, h):

        pos_catids = []
        pos_prompt_feats = []
        prompt_feats = []
        tag_bboxes = []

        no_obj_feat = np.ones(384)/np.linalg.norm(np.ones(384))
        prompt_feats.append(no_obj_feat)

        #收集目标box和特征
        for n, ann_obj in enumerate(ann):
            tag_bboxes.append(xyxy_2_xywh(ann_obj['bbox'], w, h))
            pos_prompt_feats.append(self.get_support_feat_from_query(ann_obj['id'], ann_obj['catid']))
            pos_catids.append(ann_obj['catid'])

        #根据GLIP:Prompt design for detection data.
        if random.randint(0, 1) == 0:
            num_nag_cat = train_config.max_support_len - len(tag_bboxes)
        else:
            num_nag_cat = random.randint(1, train_config.max_support_len - len(tag_bboxes))

        num_all_cat = num_nag_cat + len(tag_bboxes)

        #选取非目标类的特征做为背景特征提示
        while len(prompt_feats) < num_all_cat:

            ann = random.choice(self.coco.anns)
            if ann['catid'] in pos_catids:
                continue
            nag_prompt_feat = self.get_support_feat_from_query(ann['id'], ann['catid'])

            sim_max = 0
            for pos_prompt_feat in pos_prompt_feats:
                sim = np.dot(nag_prompt_feat,pos_prompt_feat)
                sim_max = max(sim,sim_max)
            if sim_max < 0.4:#0.4
                prompt_feats.append(nag_prompt_feat)

        tag_labels = random.sample(range(1, num_all_cat), len(tag_bboxes))
        for n,feats in enumerate(pos_prompt_feats):
            prompt_feats[tag_labels[n]] = feats

        return prompt_feats, tag_labels, tag_bboxes

    def build_lables_onehot_prompts(self,ann,w,h):

        all_clsids = [x for x in range(train_config.num_classes)]
        pos_clsids = []
        tag_bboxes = []

        for n, ann_obj in enumerate(ann):

            bbox = ann_obj['bbox']
            clsid = self.coco.dict_catid_clsid[ann_obj['catid']]
            pos_clsids.append(clsid)
            if clsid in all_clsids:
                all_clsids.remove(clsid)

            tag_bboxes.append(xyxy_2_xywh(bbox, w, h))
            #cv2.rectangle(query_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        assert len(pos_clsids) <= train_config.num_classes

        # todo 根据GLIP:Prompt design for detection data.
        if random.randint(0, 1) == 0:
            num_nag_cat = train_config.num_classes - len(pos_clsids)
        else:
            num_nag_cat = random.randint(1, train_config.num_classes - len(pos_clsids))

        nag_clsids = random.sample(all_clsids, num_nag_cat)
        prompt_clsids = pos_clsids + nag_clsids

        # todo clsid和feats捆绑
        prompt_clsidfeats = []
        for clsid in prompt_clsids:
            prompt_clsidfeats.append((clsid, self.get_onehot_vec(clsid)))
        random.shuffle(prompt_clsidfeats)
        # todo 背景类插在第0个位置
        prompt_clsidfeats.insert(0, (train_config.num_classes, self.get_onehot_vec(train_config.num_classes)))

        # todo 分解：clsid、feats
        prompt_clsids = [x[0] for x in prompt_clsidfeats]
        prompt_feats = [x[1] for x in prompt_clsidfeats]
        tag_labels = [prompt_clsids.index(id) for id in pos_clsids]

        return prompt_feats,tag_labels,tag_bboxes
    # def data_aug(self,image):
    #     import torchvision.transforms as T
    #     transform = T.Compose([
    #         T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #         T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
    #         T.ToTensor(),
    #     ])
    #     return transform(image)



    def __getitem__(self,index):

        if not self.be_train:
            #测试模式下，使用确定的采样顺序
            index = self.__test_index__()

        #索引定位一个图像
        imgid = self.image_ids[index]
        #定位当前图像的目标标注和图像文件名
        ann = self.coco.dict_imgid_anns[imgid]
        filename = self.coco.dict_imgid_filename[imgid]
        #根据图像文件名，读取特征文件。
        image_feats = self.get_swin_feats(filename)

        query_np = cv2.imread(os.path.join(self.image_folder, filename))
        #todo 数据增强
        if self.be_train:
            pass
        h,w,c = query_np.shape

        #todo
        if train_config.feat_encode_way == 'embedding':
            prompt_feats, tag_labels, tag_bboxes = self.build_prompts_from_category(ann, w, h)

        if train_config.feat_encode_way == 'onehot':
            prompt_feats, tag_labels, tag_bboxes = self.build_lables_onehot_prompts(ann, w, h)

        #输出转tensor
        image_feats = [torch.Tensor(feat) for feat in image_feats]
        #交换维度是为了便于批量对齐统一处理
        prompt_feats = torch.Tensor(prompt_feats).permute(1,0)
        tag_bboxes = torch.Tensor(tag_bboxes)
        tag_labels = torch.LongTensor(tag_labels)
        target = {'boxes': tag_bboxes, 'labels': tag_labels}

        #cv2.imshow('query_np', query_np)
        #cv2.waitKey(0)

        sample = []
        sample.extend(image_feats)
        sample.append(prompt_feats)
        sample.append(target)
        #info = {'height': h, 'width': w, 'filename': filename, 'image_id': imgid, 'ann': ann}
        #sample.append(info)
        return tuple(sample)
if __name__ == '__main__':
    ds = MyDataset(train_config.train_ann_path, train_config.train_image_folder, train_config.dino_feats_folder,train_config.swin_feats_folder,train=True)
    data_loader = DataLoader(dataset=ds, shuffle=True)
    for sample in data_loader:
        print(len(sample))



