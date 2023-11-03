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
from pandas import DataFrame as df
def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)

    return [x0, y0, x1, y1]
def main(shots,instance):
    predictions = []
    #model_path= '../train_output/checkpoint2_embedding_loss5_para_changed.pth'
    #model_path= '../train_output/best_numlevel2_state_dict_model.pth'
    model_path = '../train_output/checkpoint_20230901_N2_DIFF_SUP_FROM_QUERY_BOX3CLS5GIOU1_FIXED_SUP_SIZE_ALL_CLS_EASY_MODE_embedding.pth'

    reshape = False
    reshape_support = False
    reshape_128 = True
    full_model = True
    inference_visualize = False
    support_visualize = False
    self_train = False
    self_train_visualize = False

    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()
    with open (instance,'r') as f:
        instances = json.load(f)

    new_instances = {}
    new_instances['images'] = instances['images']
    new_instances['annotations'] = []
    new_instances['categories'] = instances['categories']
    # #去除instances['annotations']中的重复image_id
    ann_imgs = []
    #instances['annotations'] = list({each['image_id']:each for each in instances['annotations']}.values())
    #过滤掉特别小的目标bbox
    instances['annotations'] = [each for each in instances['annotations'] if (each['bbox'][2]*each['bbox'][3])>32*32]
    # for each in instances['annotations'][:]:
    #     ann_imgs.append({'image_id':each['image_id'],'bbox':each['bbox'], 'path': './val2017/' + str(each['image_id']).zfill(12) + '.jpg', 'category_id':each['category_id'], 'id':each['id'], 'iscrowd':each['iscrowd'],'area':each['area']})
    #     # 只取20张图片
    #     #new_instances['annotations'].append(each)
    #     if len(ann_imgs) == 20:
    #         # with open('./new_instances.json', 'w') as f:
     #     json.dump(new_instances, f)
    #
    #         break
    #只取20个类别的support_image的annotation============================
    #首先读取20个类别的id
    with open(f'seed4/{shots}_shot_support_df.pkl','rb') as f:
        oneshot_support_info = pickle.load(f)
        oneshot_info = deepcopy(oneshot_support_info)
    #将这个dataframe里面涉及到的category_id取出来
    oneshot_support_info = oneshot_support_info['category_id'].unique()
    #打印ID对应的类别名,并保存为字典
    dict_cate = {}
    for each in oneshot_support_info:
        dict_cate[each] = instances['categories'][each]['name']
    print(dict_cate)


    #然后读取instances['annotations']中的所有annotation，仅保留20个类别的annotation
    ann_imgs = []
    for each in instances['annotations'][:]:
        if each['category_id'] in oneshot_support_info:
            if 'voc' in instance:
                for image in instances['images']:
                    if image['id'] == each['image_id']:
                        file_name = image['file_name']
                ann_imgs.append({'image_id':each['image_id'],'bbox':each['bbox'], 'path': './VOC2012_COCO/' + file_name, 'category_id':each['category_id'], 'id':each['id'], 'iscrowd':each['iscrowd'],'area':each['area']})
            else:
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
            if 'voc' in instance:
                support_images.append(oneshot_info[oneshot_info['category_id'] == each['category_id']]['file_path'].values)
            else:
                support_images.append(oneshot_info[oneshot_info['category_id'] == each['category_id']]['file_path'].values)

    else:
        #K-SHOTS
        for each in ann_imgs:
            if 'voc' in instance:
                k_shots = oneshot_info[oneshot_info['category_id'] == each['category_id']]['file_path'][:shots].values
            else:
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
    self_train_imgs = []
    for each,ann in zip(support_images,ann_imgs):
        p_feat = []
        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        p_feat.append(no_obj_feat)
        support_dataset_path = './'
        if shots == 1:
            reshape_hack = False
            prompt_img_np = cv2.imread(support_dataset_path+(list(each)[0]))
            w,h = prompt_img_np.shape[1],prompt_img_np.shape[0]
            if reshape_128:
                prompt_img_np = cv2.resize(prompt_img_np, (128, 128), interpolation=cv2.INTER_CUBIC)
            if reshape_support:
                if ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] > 370 * 220:
                    reshape_choice = None
                elif ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] > 48 * 48:
                    reshape_choice = [(260, 180),(140, 230)]#[(200, 140),(120,96),(48,96),(60,90)]
                    if ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] < 96 * 96:
                        reshape_choice = [(60,90),(48,48)]
                        reshape_hack = False
                else:
                    reshape_choice = None
                if reshape_choice:
                    for reshape_size in reshape_choice:
                        if reshape_hack:
                            prompt_img_np_reshape = cv2.resize(cv2.resize(prompt_img_np, reshape_size),(68,68))
                            feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
                            p_feat.append(feat_reshape)
                            reshape_hack = False
                        else:
                            prompt_img_np_reshape = cv2.resize(prompt_img_np, reshape_size)
                            feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
                            p_feat.append(feat_reshape)


            feat = get_feats(dino, [prompt_img_np])[0]
            p_feat.append(feat)
            if self_train and len(self_train_imgs) > 30:
                self_train_imgs = []
            if self_train and self_train_imgs is not None:
                for self_train_img,category_id in self_train_imgs:
                    if category_id == ann_imgs[index]['category_id']:
                        feat = get_feats(dino, [self_train_img])[0]
                        p_feat.append(feat)
            p_feat = torch.Tensor(p_feat).permute(1, 0)
            p_feat = p_feat.to(device='cpu')
        else:
            reshape_hack = False
            for s in each:
                prompt_img_np = cv2.imread(support_dataset_path+s)
                if reshape_128:
                    prompt_img_np = cv2.resize(prompt_img_np, (128, 128), interpolation=cv2.INTER_CUBIC)
                w,h = prompt_img_np.shape[1],prompt_img_np.shape[0]
                if reshape_support:
                    if ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] > 370 * 220:
                        reshape_choice = None
                    elif ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] > 48 * 48:
                        #reshape_choice = [(180, 140),(120,96),(60,90),(200,320)]
                        reshape_choice = [(200, 140)]
                        if ann_imgs[index]['bbox'][2] * ann_imgs[index]['bbox'][3] < 96 * 96:
                            reshape_choice = [(60, 90)]
                            reshape_hack = False
                    else:
                        reshape_choice = None
                    if reshape_choice:
                        for reshape_size in reshape_choice:
                            if reshape_hack:
                                prompt_img_np_reshape = cv2.resize(cv2.resize(prompt_img_np, reshape_size), (68, 96))
                                feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
                                p_feat.append(feat_reshape)
                                reshape_hack = False
                            else:
                                prompt_img_np_reshape = cv2.resize(prompt_img_np, reshape_size)
                                feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
                                p_feat.append(feat_reshape)

                feat = get_feats(dino, [prompt_img_np])[0]
                p_feat.append(feat)
            if self_train and len(self_train_imgs) > 30:
                self_train_imgs = []
            if self_train and self_train_imgs is not None:

                for self_train_img, category_id in self_train_imgs:
                    if category_id == ann_imgs[index]['category_id']:
                        feat = get_feats(dino, [self_train_img])[0]
                        p_feat.append(feat)
            p_feat = torch.Tensor(p_feat).permute(1, 0)
            p_feat = p_feat.to(device='cpu')

        query_img_np = cv2.imread(ann['path'])
        if reshape:
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
            if prompt_idx > 0.0:
                # print(f'index:{prompt_idx} score:{score}')
                cv2.rectangle(query_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 5)
                # prediction.append({'image_id':ann['image_id'],'bbox': xyxy, 'score': score, 'category_id': ann['category_id']})
                prediction['image_id'] = ann_imgs[index]['image_id']
                prediction['bbox'] = xywh
                prediction['score'] = score
                prediction['category_id'] = ann_imgs[index]['category_id']
                query_has_bbox = True
                if score>=0.99 and self_train and xywh[2]*xywh[3]<200*200 and xyxy[0]>0 and xyxy[1]>0:
                    self_train_img = query_img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    if self_train_visualize:
                        cv2.imshow('self_train',self_train_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    self_train_imgs.append((self_train_img,ann_imgs[index]['category_id']))
                    print(len(self_train_imgs))

                # 保存检测图片
                # cv2.imwrite('./'+str(ann_imgs[index]['image_id'])+'.jpg',query_img)
                predictions.append(prediction) if prediction and prediction not in predictions else None
            elif score<0.3:
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
                x, y, w, h = ann_imgs[index]['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(query_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(query_img, 'EXPECTED', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (223, 123, 231), 2)
                cv2.imshow('expected-bbox:'+str(ann_imgs[index]['bbox']), query_img)
                #rectangle expected bbox with the color red,dtype int
                if support_visualize:
                    for i, s in enumerate(each):
                        s_image = cv2.resize(cv2.imread(support_dataset_path + s),(112,112))
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
        index += 1
        # progress bar
        print('推理进度：{}/{} || SHOTS:{}'.format(index, len(ann_imgs),shots))


    with open('./new_instances.json', 'w') as f:
        json.dump(new_instances, f)
        print(len(new_instances['annotations']))
    return predictions
if __name__ == '__main__':
    for shots in [1,2,3,5,10]:
        instance = './instances_val2017.json'#'./instances_val2017.json'#'./instances_val2017.json'#'./voc2012_val.json'#
        predictions = main(shots,instance)
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
        res = np.insert(res, 0, shots, axis=0)
        res = res.reshape(1,13)
        df_res = df(res,columns=['shots','AP@[IoU=0.50:0.95]','AP@[IoU=0.50]','AP@[IoU=0.75]','AP@[S]','AP@[M]','AP@[L]','AR@[IoU=0.50:0.95]','AR@[IoU=0.50:0.95]','AR@[IoU=0.50:0.95]','AR@[S]','AR@[M]','AR@[L]'])
        # Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100] =
        # Average Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 100] =
        # Average Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 100] =
        # Average Precision(AP) @ [IoU = 0.50:0.95 | area = small | maxDets = 100] =
        # Average Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100] =
        # Average Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 100] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 1] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 10] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 100] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = small | maxDets = 100] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100] =
        # Average Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 100] =

        if shots == 1:
            table = pd.DataFrame(res,
                                 columns=['shots', 'AP@[IoU=0.50:0.95]', 'AP@[IoU=0.50]', 'AP@[IoU=0.75]', 'AP@[S]',
                                          'AP@[M]', 'AP@[L]', 'AR@[IoU=0.50:0.95]', 'AR@[IoU=0.50:0.95]',
                                          'AR@[IoU=0.50:0.95]', 'AR@[S]', 'AR@[M]', 'AR@[L]'])
        else:
            table = table.append(df_res)
    #save table
        table.to_csv(f'eval_result_1-10shots_coco_all_cls_trained.csv',index=False)
    print('results saved to eval_result_1-10shots.csv')



