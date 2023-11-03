import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import glob
import cv2
import torch
import numpy as np
from config import train_config
from config.train_config import get_dino_feat_fetcher,get_swin_feat_fetcher
import json
from util.myutils import get_feats
import pickle
from copy import deepcopy
import pandas as pd
from model import build_model, collate_fn_TIDE

def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)

    return [x0, y0, x1, y1]
def main(shots):
    predictions = []
    #model_path= '../train_output/checkpoint2_embedding_loss4.49_para_changed.pth'
    model_path= '../train_output/best_numlevel2_state_dict_model.pth'
    reshape_query = False
    full_model = False
    inference_visualize = False
    support_visualize = False
    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()
    with open ('./instances_val2017.json','r') as f:
        instances = json.load(f)

    new_instances = {}
    new_instances['images'] = instances['images']
    new_instances['annotations'] = []
    new_instances['categories'] = []
    # #去除instances['annotations']中的重复image_id
    ann_imgs = []
    #instances['annotations'] = list({each['image_id']:each for each in instances['annotations']}.values())
    #过滤掉小目标bbox
    instances['annotations'] = [each for each in instances['annotations'] if (each['bbox'][2]*each['bbox'][3])>2500]
    # for each in instances['annotations'][:]:
    #     ann_imgs.append({'image_id':each['image_id'],'bbox':each['bbox'], 'path': './val2017/' + str(each['image_id']).zfill(12) + '.jpg', 'category_id':each['category_id'], 'id':each['id'], 'iscrowd':each['iscrowd'],'area':each['area']})
    #     # 只取20张图片
    #     #new_instances['annotations'].append(each)
    #     if len(ann_imgs) == 20:
    #         # with open('./new_instances.json', 'w') as f:
    #         #     json.dump(new_instances, f)
    #         break
    #只取20个类别的support_image的annotation============================
    #首先读取20个类别的id
    with open(f'{shots}_shot_support_df.pkl','rb') as f:
        oneshot_support_info = pickle.load(f)
        oneshot_info = deepcopy(oneshot_support_info)
    #将这个dataframe里面涉及到的category_id取出来
    oneshot_support_info = oneshot_support_info['category_id'].unique()
    #然后读取instances['annotations']中的所有annotation，仅保留20个类别的annotation
    ann_imgs = []
    for each in instances['categories']:
        if each['id'] in oneshot_support_info:
            new_instances['categories'].append(each)
    for each in instances['annotations'][:]:
        if each['category_id'] in oneshot_support_info:
            ann_imgs.append({'image_id':each['image_id'],'bbox':each['bbox'], 'path': './val2017/' + str(each['image_id']).zfill(12) + '.jpg', 'category_id':each['category_id'], 'id':each['id'], 'iscrowd':each['iscrowd'],'area':each['area']})
            # new_instances['annotations'].append(each)
        #if len(ann_imgs) == 20:
            # with open('./new_instances.json', 'w') as f:
            #     json.dump(new_instances, f)
            #break
    #support处理开始============================
    #根据ann_imgs中的category_id，读取oneshot_support_info中的对应的图片的file_path并存储:
    support_images = []
    if shots == 1:
        for each in ann_imgs:
            support_images.append(oneshot_info[oneshot_info['category_id'] == each['category_id']]['file_path'].values)
    else:
        #K-SHOTS
        for each in ann_imgs:
            k_shots = oneshot_info[oneshot_info['category_id'] == each['category_id']]['file_path'][:].values
            support_images.append(k_shots)
    index = 0
    # 构建模型
    model = build_model()
    checkpoint = torch.load(model_path)
    if full_model:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device='cuda')
    model = model.eval()
    for each,ann in zip(support_images,ann_imgs):
        p_feat = []
        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        p_feat.append(no_obj_feat)
        support_dataset_path = './'
        if shots == 1:
            prompt_img_np = cv2.imread(support_dataset_path+(list(each)[0]))
            w,h = prompt_img_np.shape[1],prompt_img_np.shape[0]
            feat = get_feats(dino, [prompt_img_np])[0]
            p_feat.append(feat)
            p_feat = torch.Tensor(p_feat).permute(1, 0)
            p_feat = p_feat.to(device='cpu')
        else:
            #random choice
            reshape_choice = [(224,224),(166,166),(96,96),(48,48)]
            reshape_choice2 = [(96,96),(224,224),(48,48),(116,116)]
            np.random.seed(42)
            prompt_img_np = cv2.imread(support_dataset_path+np.random.choice(each))
            feat = get_feats(dino, [prompt_img_np])[0]
            p_feat.append(feat)
            w, h = prompt_img_np.shape[1], prompt_img_np.shape[0]
            for shot in range(shots-1):
                rx,ry = reshape_choice2[shot]
                prompt_img_np_reshape = cv2.resize(prompt_img_np, (rx, ry), interpolation=cv2.INTER_CUBIC)
                feat = get_feats(dino, [prompt_img_np_reshape])[0]
                p_feat.append(feat)
            p_feat = torch.Tensor(p_feat).permute(1, 0)
            p_feat = p_feat.to(device='cpu')

        query_img_np = cv2.imread(ann['path'])
        if reshape_query:
            query_img_np = cv2.resize(query_img_np, (640, 480), interpolation=cv2.INTER_CUBIC)
        h, w, c = query_img_np.shape
        l = train_config.num_feature_levels

        featmap_lays = swin.get_feats(query_img_np)[4 - l:]
        for i in range(len(featmap_lays)):
            featmap_lays[i] = featmap_lays[i].to(device='cpu')

        sample = []
        sample.extend([featmap for featmap in featmap_lays])
        sample.extend([p_feat, None])
        batch = collate_fn_TIDE([sample])
        swin_featmap_lst, oneshot_prompt, tag = batch
        # 推理
        swin_featmap_lst = [featmap.to(device='cuda') for featmap in swin_featmap_lst]
        oneshot_prompt = oneshot_prompt.to(device='cuda')
        outputs = model(swin_featmap_lst, oneshot_prompt)

        # 后处理
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
        boxes = outputs["pred_boxes"][0]
        prediction = {}
        query_img = cv2.imread(ann_imgs[index]['path'])
        query_has_bbox = False
        for logit, box in zip(logits, boxes):
            xywh = [x.item() for x in box.cpu()]
            height, width = h,w
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
                # print(f'index:{prompt_idx} score:{score}')
                cv2.rectangle(query_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
                # prediction.append({'image_id':ann['image_id'],'bbox': xyxy, 'score': score, 'category_id': ann['category_id']})
                prediction['image_id'] = ann_imgs[index]['image_id']
                prediction['bbox'] = xywh
                prediction['score'] = score
                prediction['category_id'] = ann_imgs[index]['category_id']
                query_has_bbox = True
                # 保存检测图片
                # cv2.imwrite('./'+str(ann_imgs[index]['image_id'])+'.jpg',query_img)
                predictions.append(prediction) if prediction and prediction not in predictions else None
            elif score<0.8:
                cv2.rectangle(query_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 3)
                prediction['image_id'] = ann_imgs[index]['image_id']
                prediction['bbox'] = xywh
                prediction['score'] = 1-score
                prediction['category_id'] = ann_imgs[index]['category_id']
                query_has_bbox = True
                predictions.append(prediction) if prediction and prediction not in predictions else None
        new_instances['annotations'].append(ann_imgs[index])
        if query_has_bbox:
            # show support
            #new_instances['annotations'].append(ann_imgs[index])
            if inference_visualize:
                #s_image = cv2.imread(support_dataset_path+(support_images[index])[0])
                #cv2.imshow('support',s_image)
                cv2.imshow('bbox:'+str(ann_imgs[index]['bbox']), query_img)
                # # # #crop original box and show
                x,y,w,h = ann_imgs[index]['bbox']
                x,y,w,h = int(x),int(y),int(w),int(h)
                cv2.imshow('gt',query_img[y:y+h,x:x+w])
                if support_visualize:
                    for i, s in enumerate(each):
                        s_image = cv2.resize(prompt_img_np,(112,112))
                        cv2.imshow(f'support-{i}', s_image)
                #move windows to a row
                cv2.moveWindow('support-0', 0, 0)
                cv2.moveWindow('support-1', 112+250, 0)
                cv2.moveWindow('support-2', 224+500, 0)
                cv2.moveWindow('support-3', 336+750, 0)
                cv2.moveWindow('support-4', 0, 224)
                cv2.imshow('groundtruth',query_img[y:y+h,x:x+w])
                cv2.moveWindow('groundtruth', 112+250, 224)
                cv2.moveWindow('expected-bbox:'+str(ann_imgs[index]['bbox']), 224+500, 224)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            pass
            #predictions.append({'image_id':ann_imgs[index]['image_id'],'bbox': [0,0,0,0], 'score': 0.0001, 'category_id': ann_imgs[index]['category_id']})
        index += 1
        # progress bar
        print('推理进度：{}/{} || SHOTS:{}'.format(index, len(ann_imgs),shots))


    with open('./new_instances.json', 'w') as f:
        json.dump(new_instances, f)
        print(len(new_instances['annotations']))
    return predictions
if __name__ == '__main__':
    for shots in [1,2,3,4,5]:
        predictions = main(shots)
        print(len(predictions))
        coco = COCO('./new_instances.json')
        cocoDt = coco.loadRes(predictions)
        # 创建COCO评估器对象a
        cocoEval = COCOeval(coco, cocoDt, 'bbox')
        # 运行评估
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        res = cocoEval.stats.tolist()
        res = np.array(res) * 100
        res = str(res)
        #以表格形式保存评估结果
        with open('./eval_result_shots[{}].json'.format(shots), 'w') as f:
            json.dump(res,f)
        print("{}shots检测结果已保存为eval_result_{}.json".format(shots,shots))


