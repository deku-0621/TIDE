'''
推理接口，被测试程序调用
'''
import os
import torch
import random
import cv2
from config import train_config
import sys

from config.train_config import get_dino_feat_fetcher,get_swin_feat_fetcher
import numpy as np
from util.myutils import get_feats
from model import build_model,collate_fn_TIDE
from util.myutils import xywh_2_xyxy
from BackboneFeatureExtraction.backbone.swin_feat_fetcher import SwinFeatFetcher
from torch.nn.functional import normalize as F
device = 'cuda'
def main(SOURCE,SUPPORT):
    prompt_imgs =SUPPORT
    prompt_feats_list = []
   
    from copy import deepcopy
    #构建模型
    model = build_model()
    checkpoint = torch.load(model_path)
    if full_model:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model = model.eval()
    #推理
    import time
    cap = cv2.VideoCapture(SOURCE)
    time_now = time.time()
    #video writer:mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc, 15.0, (640,480))
    #warm up camera
    percent = 0
    for i in range(200):
        ret, frame = cap.read()
        percent_str = str(percent)
        cv2.putText(frame, 'WARMING UP:'+percent_str+'%', (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        cv2.imshow('in_img', frame)
        cv2.waitKey(1)
        percent+=1
        out.write(frame)

    start = False
    no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
    prompt_feats_list.append(no_obj_feat)
    for prompt_img in prompt_imgs:
        if reshape_support:
            prompt_img_np_reshape = cv2.resize(prompt_img, (64, 64))
            feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
            prompt_feats_list.append(feat_reshape)
        feat = get_feats(dino, [prompt_img])[0]
        prompt_feats_list.append(feat)
    while cap.isOpened():
        prompt_feats = torch.Tensor(prompt_feats_list).permute(1,0)
        prompt_feats = prompt_feats.to(device)
        if not start:
            start = True
            print('推理开始')
        ret, frame = cap.read()
        h, w, c = frame.shape
        l = train_config.num_feature_levels
        featmap_lays = swin.get_feats(frame)[4-l:]
        sample = []
        sample.extend([featmap for featmap in featmap_lays])
        sample.extend([prompt_feats, None])
        batch = collate_fn_TIDE([sample])
        swin_featmap_lst, prompt_feats_video,tag = batch
        outputs = model(swin_featmap_lst, prompt_feats_video)
        #输出后处理
        #logits = outputs["pred_logits"].cpu().sigmoid()[0]
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
        boxes = outputs["pred_boxes"][0]
        for logit,box in zip(logits,boxes):
            xywh = [x.item() for x in box.cpu()]
            box = xywh_2_xyxy(xywh,w,h)
            prompt_cls = logit.argmax().item()
            score = logit[prompt_cls].item()
            
            if prompt_cls>0 and score>POS_THRE:
                print('class:',prompt_cls,'score:',score)
                
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                if score>0.999999999999:
                    prompt_img = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                    p_feat = get_feats(dino, [prompt_img])[0]
                    prompt_feats_list.append(p_feat)
                cv2.putText(frame, 'class:'+str(prompt_cls), (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                score_display = float(str(score)[:5])
                cv2.putText(frame, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            elif score<NEG_THRE:
                print('NEG CLASS:', prompt_cls, 'score:', score)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(frame, 'class:' + str(prompt_cls), (int(box[0]), int(box[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                score_display = float(str(score)[:5])
                cv2.putText(frame, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('in_img',frame)
        cv2.waitKey(1)
        #cv2.imwrite(f'output/output_{prompt_class}.jpg',frame)
        #print('time:', time.time() - time_now)
        time_now = time.time()
        #save video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            break

if __name__ == '__main__':
    RSenhance = False
    #reshape  False 
    reshape_support = False
    full_model = True
    iter_num = 1
    assert iter_num>0
    model_path= '../checkpoint_2023810_N2_DIFF_SUP_FROM_QUERY_embedding.pth'
    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()
    POS_THRE = 0.0
    NEG_THRE = 0.1
    test = False
    SOURCE = 'rtsp://admin:admin@192.168.0.104:8554/live'
    crop = False
    #调用本地摄像头截取一帧并使用ROI截取SUPPORT
    if crop:
        cap0 = cv2.VideoCapture(0)
        ret, frame = cap0.read()
        #Region Of Interest
        ROI = cv2.selectROI('in_img',frame,fromCenter=False,showCrosshair=False)
        # 从选择的ROI中截取图像
        frame = frame[int(ROI[1]):int(ROI[1]+ROI[3]), int(ROI[0]):int(ROI[0]+ROI[2])]
        cv2.imshow('your support image looks like this',frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        SUPPORT = frame
        cap0.release()
    else:
        p1 = cv2.imread('bear.png')
        p2 = cv2.imread('bear2.png')
        p3 = cv2.imread('bear3.png')
        p8 = cv2.imread('bear4.png')
        p9 = cv2.imread('guitar.png')
        p10 = cv2.imread('fan.png')
        p4 = cv2.resize(p1,(144,157))
        p5 = cv2.resize(p1,(80,122))
        p6 = cv2.resize(p1,(224,224))
        p7 = cv2.resize(p1,(280,320))
        # p2 = cv2.imread('guitar.png')
        # p3 = cv2.imread('fan.png')
        # p2 = cv2.resize(p1,(224,224))
        # p5 = cv2.resize(p2,(224,224))q
        # p6 = cv2.resize(p3,(224,224))
        # p3 = cv2.resize(p1,(112,112))
        #rotate
        # p4 = cv2.rotate(p1,cv2.ROTATE_90_CLOCKWISE)
        # p5 = cv2.rotate(p1,cv2.ROTATE_180)
        # P6 = cv2.resize(p1,(420,280))
        # p8 = cv2.resize(p2,(112,112))
        # p9 = cv2.resize(p3,(112,112))
        SUPPORT = [p8]

    if test:
        cap = cv2.VideoCapture(SOURCE)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('in_img',frame)
                if cv2.waitKey(1)&0xFF == ord('esc'):
                    break
            else:
                break
    else:
        main(SOURCE,SUPPORT)

    


