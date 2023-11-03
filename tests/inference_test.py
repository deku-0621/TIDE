'''
推理接口，被测试程序调用
'''
import os
#os.chdir('..')
import torch
import random
import cv2
from config import train_config

from config.train_config import get_dino_feat_fetcher,get_swin_feat_fetcher
import numpy as np
from util.myutils import get_feats
from model import build_model,collate_fn_TIDE
from util.myutils import xywh_2_xyxy
from BackboneFeatureExtraction.backbone.swin_feat_fetcher import SwinFeatFetcher
from torch.nn.functional import normalize as F
device = 'cuda'
if __name__ == '__main__':
    RSenhance = False
    reshape =True
    reshape_support = False
    full_model = True
    reshape_128 = True
    iter_num = 1
    assert iter_num>0

    model_path= '../train_output/checkpoint_20230904_N2_CLS2BOX5GIOU1_BINARY_embedding.pth'

    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()
    NEG_THRES = 0.95203
    #in_img_path = './input/101379.jpg'
    in_img_path = './input/lamp_query.jpg'
    prompt_img_paths = ['input/lamp_support.jpg']

    # in_img_path = './input/000000571264.jpg'
    # prompt_img_paths = ['./input/2.jpg']
    # in_img_path = 'input/gt6.jpg'
    # prompt_img_paths = ['input/support_gt6.jpg']
    prompt_class = 'gc'
    #prompt_img_paths = ['input/111.jpg']#头盔
    #prompt_img_paths = ['input/571264.jpg','input/2.jpg']

    #构建输入
    image_feats = []
    prompt_feats_list = []

    #in_img_np = cv2.resize(cv2.imread(in_img_path),(640,480))
    in_img_np = cv2.imread(in_img_path)
    if RSenhance:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,num_grow_ch=32,scale=4)
        upsampler = RealESRGANer(
         scale=4,
         model_path='../evaluation/RealESRGAN_x4plus.pth',
         model=model_4x,
         tile=0,
         tile_pad=10,
         pre_pad=0,
         half=True,
         gpu_id=0
        )
        in_img_np,_ = upsampler.enhance(in_img_np)
    if reshape:
        in_img_np = cv2.resize(in_img_np, (640, 480))
    h, w, c = in_img_np.shape


    if train_config.feat_encode_way == 'embedding':
        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        #no_obj_feat = np.ones((384,50,50)) / np.linalg.norm(np.ones((384,50,50)))
        prompt_feats_list.append(no_obj_feat)
        for path in prompt_img_paths:
            prompt_img_np = cv2.resize(cv2.imread(path),(233,233))
            #from util.slconfig import SLConfig
            #config_args = SLConfig.fromfile(
            #    '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/config/train_config.py')
            #fetcher = SwinFeatFetcher(config_args)
            #feat = fetcher.get_support_feats(prompt_img_np)[0]
            if reshape_128:
                prompt_img_np = cv2.resize(cv2.imread(path),(128,128))
            else:
                prompt_img_np = cv2.imread(path)
            if reshape_support:
                prompt_img_np_reshape = cv2.resize(prompt_img_np, (128, 128))
                feat_reshape = get_feats(dino, [prompt_img_np_reshape])[0]
                prompt_feats_list.append(feat_reshape)
            feat = get_feats(dino, [prompt_img_np])[0]
            prompt_feats_list.append(feat)


    if train_config.feat_encode_way == 'onehot':

        obj_clsids = [0,3]
        #obj_clsids = [30,0,41,43,17,2,74,16,18]

        prompt_clsids = [80,30,0,3,41,43,17,2,74,16,18]
        for clsid in prompt_clsids:
            prompt_feat = np.eye(train_config.num_classes + 1)[clsid]
            noise = [random.uniform(-0.20, 0.20) for n in range(81)]
            #prompt_feat += noise
            prompt_feats_list.append(prompt_feat)


    prompt_feats = torch.Tensor(prompt_feats_list).permute(1,0)
    prompt_feats = prompt_feats.to(device)

    l = train_config.num_feature_levels
    featmap_lays = swin.get_feats(in_img_np)[4-l:]

    sample = []
    sample.extend([featmap for featmap in featmap_lays])
    sample.extend([prompt_feats, None])


    batch = collate_fn_TIDE([sample])
    swin_featmap_lst, prompt_feats,tag = batch

    #构建模型
    model = build_model()
    checkpoint = torch.load(model_path)
    if full_model:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model = model.eval()
    #打印模型参数数量
    print('model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    img_display = np.copy(in_img_np)
    #推理
    #time_now
    import time
    time_now = time.time()
    for i in range(iter_num):
        outputs = model(swin_featmap_lst, prompt_feats)
        print('time:', time.time() - time_now)
        #输出后处理
        #logits = outputs["pred_logits"].cpu().sigmoid()[0]
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
        boxes = outputs["pred_boxes"][0]
        for logit,box in zip(logits,boxes):
            xywh = [x.item() for x in box.cpu()]
            box = xywh_2_xyxy(xywh,w,h)
            prompt_cls = logit.argmax().item()
            score = logit[prompt_cls].item()
            if prompt_cls>0:
                print('class:',prompt_cls,'score:',score)
                cv2.rectangle(in_img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), -1)
                cv2.rectangle(img_display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(img_display, 'class:'+str(prompt_cls), (int(box[0]), int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                score_display = float(str(score)[:5])
                cv2.putText(img_display, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            elif score<NEG_THRES:
                print('NEG CLASS:', prompt_cls, 'score:', score)
                cv2.rectangle(in_img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), -1)
                cv2.rectangle(img_display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_display, 'class:' + str(prompt_cls), (int(box[0]), int(box[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                score_display = float(str(score)[:5])
                cv2.putText(img_display, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0), 2)





        # # logits_filt = logits.clone()
        # # boxes_filt = boxes.clone()
        # # filt_mask_zero = logits_filt.max(dim=1)[0] <= 1.0
        # # filt_mask_pos = logits_filt.max(dim=1)[0] > 0.0
        # # logits_filt_zero = logits_filt[filt_mask_zero]
        # # boxes_filt_zero = boxes_filt[filt_mask_zero]
        # # logits_filt_pos = logits_filt[filt_mask_pos]
        # # boxes_filt_pos = boxes_filt[filt_mask_pos]
        # for logit_zero, box_zero,logit_pos,box_pos in zip(logits_filt_zero, boxes_filt_zero,logits_filt_pos, boxes_filt_pos):
        #
        #     xywh_zero = [x.item() for x in box_zero.cpu()]
        #     box_zero = xywh_2_xyxy(xywh_zero, w, h)
        #     xywh_pos = [x.item() for x in box_pos.cpu()]
        #     box_pos = xywh_2_xyxy(xywh_pos, w, h)
        #     prompt_idx_for_zero = logit_zero.argmax().item()
        #     score_zero = logit_zero[prompt_idx_for_zero].item()
        #     prompt_idx_for_pos = logit_pos.argmax().item()
        #     score_pos = logit_pos[prompt_idx_for_pos].item()
        #     if train_config.feat_encode_way == 'onehot':
        #         pass
        #     #     clsid = prompt_clsids[prompt_idx]
        #     #     if clsid not in obj_clsids:
        #     #         continue
        #     #     print('idx:', prompt_idx, 'clsid:', clsid, 'score:', score)
        #     #     cv2.rectangle(in_img_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 123), 5)
        #     else:
        #         if prompt_idx_for_pos > 0:
        #             print('idx:', prompt_idx_for_pos, 'score:', score_pos)
        #             cv2.rectangle(in_img_np, (box_pos[0], box_pos[1]), (box_pos[2], box_pos[3]), (12, 255, 55),
        #                           thickness=-1)
        #             cv2.rectangle(img_display, (box_pos[0], box_pos[1]), (box_pos[2], box_pos[3]), (0, 255, 123), 3)
        #             cv2.putText(img_display, f'CLASS:{prompt_idx_for_pos}',
        #                         (box_pos[0], box_pos[1] - (h // box_pos[1]) * 2), fontFace=cv2.FONT_HERSHEY_DUPLEX,
        #                         color=(12, 255, 11), fontScale=1, thickness=2)
        #             score = float(str(score_pos)[:5])
        #             cv2.putText(img_display, f'score:{score}', (box_pos[0], box_pos[3] + (h // box_pos[3]) * 50),
        #                         fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255), fontScale=3, thickness=2,
        #                         bottomLeftOrigin=False)
        #         if prompt_idx_for_zero == 0 and score_pos < 0.99:
        #             print('BACKGROUND CLASS','idx:', prompt_idx_for_zero, 'score:', score_zero)
        #             cv2.rectangle(in_img_np, (box_zero[0], box_zero[1]), (box_zero[2], box_zero[3]), (12, 255, 55), thickness=-1)
        #             cv2.rectangle(img_display, (box_zero[0], box_zero[1]), (box_zero[2], box_zero[3]), (0, 255, 123), 3)
        #             cv2.putText(img_display,f'CLASS:{prompt_idx_for_zero}',(box_zero[0], box_zero[1]-(h//box_zero[1])*2),fontFace=cv2.FONT_HERSHEY_DUPLEX,color=(12,255,11),fontScale=1,thickness=2)
        #             score = float(str(score_zero)[:5])
        #             cv2.putText(img_display,f'score:{score}', (box_zero[0], box_zero[3] + (h//box_zero[3])*50),
        #                         fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255), fontScale=3, thickness=2,bottomLeftOrigin=False)




        prompt_feats = torch.Tensor(prompt_feats_list).permute(1, 0)
        prompt_feats = prompt_feats.to(device)
        featmap_lays = swin.get_feats(in_img_np)[4 - l:]
        sample = []
        sample.extend([featmap for featmap in featmap_lays])
        sample.extend([prompt_feats, None])
        batch = collate_fn_TIDE([sample])
        swin_featmap_lst, prompt_feats, tag = batch
    index = 1
    for each in prompt_img_paths:
        prompt_img_np = cv2.imread(each)
        cv2.imshow(f'support_class-{index}',prompt_img_np)
        index+=1
    cv2.imshow('in_img',img_display)
    cv2.waitKey(0)
    cv2.imwrite(f'output/output_{prompt_class}.jpg',img_display)



