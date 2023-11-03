'''
推理接口，被测试程序调用
'''
import torch
import matplotlib.pyplot as plt
import random
import cv2
from config import train_config
from config.train_config import get_dino_feat_fetcher,get_swin_feat_fetcher
import numpy as np
from util.myutils import get_feats
from model import build_model,collate_fn_TIDE
from util.myutils import xywh_2_xyxy
from torch.nn.functional import normalize as F
activation = {}
gradient = {}
def get_activation(name):
    def hook(model, input, output):
        if name not in activation.keys():
            activation[name] = output
        else:
            activation[name]=torch.cat([output,activation[name]],dim=0)
    return hook
def get_gradient(name):
    def hook(model, input, output):
        if name not in gradient.keys():
            gradient[name] = input
        else:
            gradient[name] = torch.cat([input,gradient[name]],dim=0)
    return hook
device = 'cuda'
if __name__ == '__main__':
    RSenhance = False
    reshape = False
    reshape_support = False
    full_model = True
    iter_num = 1
    assert iter_num>0
    model_path= '../train_output/checkpoint2_embedding_loss4.399_para_changed.pth'
    dino = get_dino_feat_fetcher()
    swin = get_swin_feat_fetcher()

    #in_img_path = './tests/input/lamp_query.jpg'
    #prompt_img_paths = ['./tests/input/lamp3.jpg','./tests/input/lamp1.jpg','./tests/input/lamp2.jpg']
    #in_img_path = './tests/input/000000571264.jpg'
    #prompt_img_paths = ['./tests/input/2.jpg']
    #prompt_img_paths = ['./tests/input/571264.jpg']
    in_img_path = '../tests/input/trashman.jpg'
    prompt_img_paths = ['../tests/input/bin1.jpg','../tests/input/support_man.jpg','../tests/input/bin2.jpg']
    #in_img_path = './tests/input/lamp_query.jpg'
    #prompt_img_paths = ['./tests/input/lamp_support.jpg']
    prompt_class = 'unknown'
    #prompt_img_paths = ['input/111.jpg']#头盔
    #prompt_img_paths = ['input/571264.jpg','input/2.jpg']

    #构建输入
    image_feats = []
    prompt_feats_list = []

    #in_img_np = cv2.resize(cv2.imread(in_img_path),(640,480))
    in_img_np = cv2.imread(in_img_path)
    # if RSenhance:
    #     from basicsr.archs.rrdbnet_arch import RRDBNet
    #     from realesrgan import RealESRGANer
    #     model_4x = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,num_grow_ch=32,scale=4)
    #     upsampler = RealESRGANer(
    #      scale=4,
    #      model_path='../evaluation/RealESRGAN_x4plus.pth',
    #      model=model_4x,
    #      tile=0,
    #      tile_pad=10,
    #      pre_pad=0,
    #      half=True,
    #      gpu_id=0
    #     )
    #     in_img_np,_ = upsampler.enhance(in_img_np)
    if reshape:
        in_img_np = cv2.resize(in_img_np, (640, 480))
    h, w, c = in_img_np.shape
    print(h,w,c)


    if train_config.feat_encode_way == 'embedding':
        no_obj_feat = np.ones(384) / np.linalg.norm(np.ones(384))
        prompt_feats_list.append(no_obj_feat)
        for path in prompt_img_paths:
            prompt_img_np = cv2.imread(path)
            if reshape_support:
                prompt_img_np_reshape = cv2.resize(prompt_img_np, (64, 64))
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
    print(model)
    checkpoint = torch.load(model_path)
    if full_model:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.input_proj[0].register_backward_hook(get_gradient('fusion'))
    model.input_proj[0].register_forward_hook(get_activation('fusion'))
    # model.input_proj[1].register_backward_hook(get_gradient('fusion'))
    # model.input_proj[1].register_forward_hook(get_activation('fusion'))
    #model.transformer.encoder.fusion_layers[5].register_backward_hook(get_activation('fusion'))
    #model = model.eval()
    #打印模型参数数量
    print('model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    img_display = np.copy(in_img_np)
    #推理
    #time_now
    import time
    time_now = time.time()
    for i in range(iter_num):
        outputs = model(swin_featmap_lst, prompt_feats)
        print(outputs['pred_logits'].shape)
        #输出后处理
        #logits = outputs["pred_logits"].cpu().sigmoid()[0]
        logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
        boxes = outputs["pred_boxes"][0]
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > 0.0
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        backward_data =[]
        for logit, box in zip(logits_filt, boxes_filt):

            xywh = [x.item() for x in box.cpu()]
            box = xywh_2_xyxy(xywh, w, h)
            prompt_idx = logit.argmax().item()
            score = logit[prompt_idx].item()
            if train_config.feat_encode_way == 'onehot':
                clsid = prompt_clsids[prompt_idx]
                if clsid not in obj_clsids:
                    continue
                print('idx:', prompt_idx, 'clsid:', clsid, 'score:', score)
                cv2.rectangle(in_img_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 123), 5)
            else:
                if prompt_idx > 0:
                    print('idx:', prompt_idx, 'score:', score)
                    backward_data.append(logit[prompt_idx].unsqueeze(0))
                    cv2.rectangle(in_img_np, (box[0], box[1]), (box[2], box[3]), (12, 255, 55), thickness=-1)
                    cv2.rectangle(img_display, (box[0], box[1]), (box[2], box[3]), (0, 255, 123), 3)
                    cv2.putText(img_display,f'{prompt_class}',(box[0], box[1]-(h//box[1])*2),fontFace=cv2.FONT_HERSHEY_DUPLEX,color=(100,11,11),fontScale=1,thickness=2)
                    score = float(str(score)[:5])
                    cv2.putText(img_display,f'score:{score}', (box[0], box[3] + (h//box[3])*50),
                                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255), fontScale=3, thickness=2,bottomLeftOrigin=False)
        prompt_feats = torch.Tensor(prompt_feats_list).permute(1, 0)
        prompt_feats = prompt_feats.to(device)
        featmap_lays = swin.get_feats(in_img_np)[4 - l:]
        sample = []
        sample.extend([featmap for featmap in featmap_lays])
        sample.extend([prompt_feats, None])
        batch = collate_fn_TIDE([sample])
        swin_featmap_lst, prompt_feats, tag = batch
    for each in prompt_img_paths:
        prompt_img_np = cv2.imread(each)
        cv2.imwrite(f'support_{prompt_class}.png',prompt_img_np)
        print("write")
    cv2.imwrite('in_img.png',img_display)
    cv2.imwrite(f'output_{prompt_class}.png',img_display)
    print('time:', time.time() - time_now)
    from torch.nn import functional as F
    print(backward_data)
    torch.cat(backward_data).mean().backward()
    
    
    # 获取模型的梯度
    gradients = gradient['fusion'][0]
    # 计算梯度相应通道的均值
    mean_gradients = torch.mean(gradients, dim=[0,2,3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = activation['fusion']
    # 每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:,i,:,:] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    ratio_optical=0
    heatmap = torch.mean(ratio_optical*F.tanh(activations)+(1-ratio_optical)*F.tanh(gradients), dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().detach().numpy()

    #gradients = F.relu(gradient['fusion'][0])
    #print(gradients.shape)
    #mean_gradients = torch.mean(gradients, dim=[0,2,3])
    #activations = activation['fusion']
    #print(activations.shape)
    #for i in range(len(mean_gradients)):
    #    activations[:,i,:,:] *= mean_gradients[i]
    #heatmap = torch.mean(activations,dim=1,keepdim=True)
    #print(heatmap.shape)
    #heatmap = F.relu(heatmap)
    #heatmap /= torch.max(heatmap)
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    print(heatmap.shape)
    img = cv2.imread(in_img_path)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    grad_cam_img = heatmap * 0.4 + img
    #grad_cam_img = heatmap
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    b,g,r = cv2.split(grad_cam_img)
    grad_cam_img = cv2.merge([r,g,b])
    print(grad_cam_img.shape)
    plt.imsave('heatmap.png',grad_cam_img)
    #plt.imsave('heatmap')


