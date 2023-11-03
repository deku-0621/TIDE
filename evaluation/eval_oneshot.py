import random
from time import time

import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import glob
import cv2
import torch
import numpy as np
from config import train_config
from config.train_config import get_dino_feat_fetcher, get_swin_feat_fetcher
import json
from util.myutils import get_feats


def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)

    return [x0, y0, x1, y1]


def getfromcategories(categoryIndex, category_id):
    index = random.randint(0, len(categoryIndex[category_id]) - 1)


def main():
    model_path = '../train_output/checkpoint2_embedding_loss5_para_changed.pth'
    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()
    start = time()
    with open('./instances_val2017.json', 'r') as f:
        instances = json.load(f)
    # Initialize COCO API
    coco = COCO('./instances_val2017.json')

    # Get the category information
    categories = coco.loadCats(coco.getCatIds())

    # Create a dictionary to store the counts
    category_counts = {}

    # Iterate through each category and count the number of images
    for category in categories:
        category_id = category['id']
        category_name = category['name']
        image_ids = coco.getImgIds(catIds=category_id)
        num_images = len(image_ids)
        category_counts[category_name] = num_images
    # Print the result
    for category_name, num_images in category_counts.items():
        print(f"{category_name}: {num_images} images")

    with open('./instances_train2017.json') as f:
        train_instances = json.load(f)
    end = time()
    print(f"Loaded json successfully! Cost time {end-start}s")
    ##########################################
    # Preprocess Category Support Bbox & images
    category_ids = np.unique([each['category_id'] for each in train_instances['annotations']])
    categoryIndex = {category_id: [] for category_id in category_ids}
    for each in train_instances['annotations']:
        categoryIndex[each['category_id']].append([each['image_id'], each['bbox']])
    ###########################################
    new_instances = {}
    new_instances['images'] = instances['images']
    new_instances['annotations'] = []
    new_instances['categories'] = instances['categories']
    # 去除instances['annotations']中的重复image_id
    ann_imgs = []
    instances['annotations'] = list({each['image_id']: each for each in instances['annotations']}.values())
    for each in instances['annotations']:
        index = random.randint(0, len(categoryIndex[each['category_id']]) - 1)
        ann_imgs.append(dict(image_id=each['image_id'],
                             bbox=each['bbox'],
                             path='./val2017/' + str(each['image_id']).zfill(12) + '.jpg',
                             crop_box=categoryIndex[each['category_id']][index][1],
                             s_path='./train2017/' + str(categoryIndex[each['category_id']][index][0]).zfill(12) + '.jpg',
                             #support image path
                             category_id=each['category_id']))
        new_instances['annotations'].append(each)
        # 只取20张图片
        if len(ann_imgs) == 20:
            with open('./new_instances.json', 'w') as f:
                json.dump(new_instances, f)
            break
    # support处理开始============================
    prompt_feats = []

    for ann in ann_imgs:
        p_feat = []
        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        p_feat.append(no_obj_feat)
        prompt_img_np = cv2.imread(ann['s_path'])

        # xywh to xyxy
        x1 = ann['crop_box'][0]
        y1 = ann['crop_box'][1]
        x2 = ann['crop_box'][0] + ann['crop_box'][2]
        y2 = ann['crop_box'][1] + ann['crop_box'][3]
        #print(x1, y1, x2, y2)
        # 裁减为bbox范围内的图像
        prompt_img_np = prompt_img_np[int(y1):int(y2), int(x1):int(x2)]
        h, w, c = prompt_img_np.shape
        if h < 16 or w < 16: prompt_img_np = cv2.resize(prompt_img_np, (16, 16))
        # cv2.imshow('image_id:'+str(ann['image_id']),prompt_img_np)
        # cv2.imshow('bbox:'+str(ann['bbox']), cv2.imread(ann['path']))
        # # print(ann['category_id'])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        feat = get_feats(dino, [prompt_img_np])[0]
        p_feat.append(feat)
        p_feat = torch.Tensor(p_feat).permute(1, 0)
        p_feat = p_feat.to(device='cuda')
        prompt_feats.append(p_feat)
        # 进度条
        print('support处理进度：{}/{}'.format(ann_imgs.index(ann), len(ann_imgs)))
    # support处理结束============================

    # query处理开始=============================
    l = train_config.num_feature_levels
    featmap_lays_list = []
    hwc = []
    for ann in ann_imgs:
        query_img_np = cv2.imread(ann['path'])
        h, w, c = query_img_np.shape
        hwc.append((h, w, c))
        featmap_lays = swin.get_feats(query_img_np)[4 - l:]
        featmap_lays_list.append(featmap_lays)
        # 进度条
        print('query处理进度：{}/{}'.format(ann_imgs.index(ann), len(ann_imgs)))
    print(f'len_featmap_lays_list:{len(featmap_lays_list)}')
    print(f'len_prompt_feats:{len(prompt_feats)}')
    # query处理结束=============================
    # 推理开始==================================
    from model import build_model, collate_fn_food
    predictions = []
    index = 0
    for featmap_layers, oneshot_prompt_feat in zip(featmap_lays_list, prompt_feats):
        sample = []
        sample.extend([featmap for featmap in featmap_layers])
        sample.extend([oneshot_prompt_feat, None])
        batch = collate_fn_food([sample])
        swin_featmap_lst, oneshot_prompt, tag = batch
        # 构建模型
        model = build_model()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device='cuda')
        model = model.eval()
        # 推理
        outputs = model(swin_featmap_lst, oneshot_prompt)
        # 后处理
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
        boxes = outputs["pred_boxes"][0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > 0.5
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        prediction = {}
        query_img = cv2.imread(ann_imgs[index]['path'])
        for logit, box in zip(logits_filt, boxes_filt):
            xywh = [x.item() for x in box.cpu()]
            height, width = hwc[index][0], hwc[index][1]
            xyxy = xywh_2_xyxy(xywh, width, height)
            # xyxy to xywh
            x = xyxy[0]
            y = xyxy[1]
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            xywh = [x, y, width, height]
            prompt_idx = logit.argmax().item()
            score = logit[prompt_idx].item()
            if prompt_idx > 0:
                cv2.rectangle(query_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                # prediction.append({'image_id':ann['image_id'],'bbox': xyxy, 'score': score, 'category_id': ann['category_id']})
                prediction['image_id'] = ann_imgs[index]['image_id']
                prediction['bbox'] = xywh
                prediction['score'] = score
                prediction['category_id'] = ann_imgs[index]['category_id']
        cv2.imshow('support',cv2.imread(ann_imgs[index]['s_path']))
        cv2.imshow('query',query_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        predictions.append(prediction) if prediction else None
        index += 1
    return predictions


if __name__ == '__main__':
    predictions = main()
    print(len(predictions))
    coco = COCO('./new_instances.json')
    cocoDt = coco.loadRes(predictions)
    # 创建COCO评估器对象a
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    # 运行评估
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
