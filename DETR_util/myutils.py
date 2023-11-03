import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
import DETR_util.mytransforms as T
import random
from torchvision import transforms as tfs
import torchvision.transforms.functional as F
import torch.nn.functional as F2
from PIL import Image
import argparse
import torch
import DETR_util.misc as utils

# if config.measure_use_dino:
#     from dino.feat_fetcher import ViTFeatFetcher
#     dino_path = 'dino/dino_deitsmall16_pretrain_full_checkpoint.pth'
#     feat_fetcher = ViTFeatFetcher(dino_path, 'cuda')



def get_feats(feat_fetcher, img_lst):
    img_tensors = []
    for img in img_lst:
        img_ = Image.fromarray(img)
        img_tensor = tfs.ToTensor()(img_).to('cuda')
        img_tensors.append(img_tensor)

    img_tensors = torch.stack(img_tensors)
    feats = feat_fetcher.get_feats(img_tensors)

    return feats


#todo
def get_featmap_by_dino(feat_fetcher, img_np):

    img_ = Image.fromarray(img_np)
    img_tensor = tfs.ToTensor()(img_).to('cuda')
    featmap = feat_fetcher.get_featmap_pro(img_tensor)
    return featmap

def cut_align_size(img_np):
    height, width, c = img_np.shape
    height = height - height % 16
    width = width - width % 16
    img_np = img_np[:height, :width, :]
    return img_np

def infer_multi(model,feat_fetcher,query_np,support_np,support_feat,device,iter_nums = 1):

    query_np = query_np.copy()

    normalize, _ = make_my_transforms()
    output_objs = []

    for i in range(iter_nums):
        q_tensor = normalize(Image.fromarray(query_np))
        s_tensor = normalize(Image.fromarray(support_np))

        query_wh_tensor = torch.Tensor([q_tensor.shape[2], q_tensor.shape[1]])
        support_wh_tensor = torch.Tensor([s_tensor.shape[2], s_tensor.shape[1]])

        batch = [(q_tensor, s_tensor, query_wh_tensor, support_wh_tensor)]
        batch = utils.collate_fn_poly(batch)
        querys, supports, querys_wh, supports_wh = batch[:4]

        querys = querys.to(device)
        supports = supports.to(device)

        outputs = model(querys, supports)

        p_boxes = outputs['pred_boxes'][0, :, :].detach().cpu().numpy()
        p_box = p_boxes[0]
        c, q_height, q_width = q_tensor.shape
        bbox_pred = xywh_2_xyxy(tuple(p_box), q_width, q_height)

        '''
        probs = torch.nn.functional.softmax(outputs['pred_logits'][0][0]).cpu().detach().numpy()
        clsid = np.argmax(probs)
        prob = probs[clsid]
        '''
        #todo 相似度代替置信度
        obj_cuted = query_np[bbox_pred[1]:bbox_pred[3],bbox_pred[0]:bbox_pred[2],:]
        h,w,c = np.shape(obj_cuted)
        if h>32 and w>32:
            #cv2.imshow('obj_cuted',obj_cuted)
            #cv2.imshow('support', support_np)
            #cv2.waitKey(0)

            q_feat = get_feats(feat_fetcher,[obj_cuted])[0]
            #s_feat = get_feats(feat_fetcher, [support_np])[0]
            #prob = np.dot(q_feat,s_feat)
            prob = np.dot(q_feat, support_feat)

            #print('prob:',prob)
            output_objs.append((tuple(p_box), prob))

            #遮盖已检测到的
            cv2.rectangle(query_np, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (122, 215, 155),
                          thickness=-1)


            # cv2.rectangle(query_np, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (122,215,155), thickness=1)
            # cv2.imshow('query_np',query_np)
            # cv2.imshow('support_np', support_np)
            # cv2.waitKey(0)

    return output_objs

def infer_once(model,query_np,support_np,device):

    normalize, _ = make_my_transforms()

    q_tensor = normalize(Image.fromarray(query_np))
    s_tensor = normalize(Image.fromarray(support_np))

    query_wh_tensor = torch.Tensor([q_tensor.shape[2], q_tensor.shape[1]])
    support_wh_tensor = torch.Tensor([s_tensor.shape[2], s_tensor.shape[1]])

    batch = [(q_tensor, s_tensor, query_wh_tensor, support_wh_tensor)]
    batch = utils.collate_fn_poly(batch)
    querys, supports, querys_wh, supports_wh = batch[:4]

    querys = querys.to(device)
    supports = supports.to(device)

    if config.DETR_VERSION ==8:
        outputs = model(querys, supports, querys_wh)
    elif config.DETR_VERSION in [9,10]:
        outputs = model(querys, supports, querys_wh, supports_wh)
    else:
        outputs = model(querys, supports)

    p_boxes = outputs['pred_boxes'][0, :, :].detach().cpu().numpy()
    c, q_height, q_width = q_tensor.shape
    p_sims = outputs['pred_sims'][0, :, 0].detach().cpu().numpy()

    output_objs = []
    for n in range(len(p_boxes)):
        probs = torch.nn.functional.softmax(outputs['pred_logits'][0][n]).cpu().detach().numpy()
        clsid = np.argmax(probs)
        if clsid == 0:
            #prob = probs[clsid]
            prob = p_sims[n]

            #bbox_pred = xywh_2_xyxy(tuple(p_boxes[n]), q_width, q_height)
            bbox_pred = p_boxes[n]
            output_objs.append((bbox_pred,prob))

    return output_objs

def get_featmap(model,im_np,device='cuda'):
    #对输入图像经过resnet提取特征图
    h,w,c = im_np.shape

    normalize, _ = make_my_transforms()

    tensor = normalize(Image.fromarray(im_np))
    tensors = tensor.unsqueeze(0)
    masks = torch.zeros((1,h,w), dtype=torch.bool, device=tensor.device)
    tensor_mask = utils.NestedTensor(tensors, masks)
    featmaps = model.fetch_featmap(tensor_mask.to(device))[0]
    featmap = featmaps[0]
    featmap = F2.normalize(featmap, dim=0)
    return featmap

def make_my_transforms():

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return normalize,T.Compose([
        tfs.RandomHorizontalFlip(0.5),
        tfs.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        tfs.RandomGrayscale(0.2),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        tfs.RandomRotation(15),
        tfs.RandomAffine(degrees=0, translate=(0, 0), shear=10),
        T.RandomAspect(config.min_cut_size,config.max_cut_size),
    ])

def xyxy_2_xywh(box, w, h):
    cent_x = round((box[0] + box[2]) / 2 / w, 2)
    cent_y = round((box[1] + box[3]) / 2 / h, 2)
    box_w = round((box[2] - box[0]) / w, 2)
    box_h = round((box[3] - box[1]) / h, 2)

    return [cent_x, cent_y, box_w, box_h]


def xywh_2_xyxy(box, w, h):
    cent_x, cent_y, box_w, box_h = box
    x0 = int((cent_x - box_w / 2) * w)
    y0 = int((cent_y - box_h / 2) * h)
    x1 = int((cent_x + box_w / 2) * w)
    y1 = int((cent_y + box_h / 2) * h)

    return [x0, y0, x1, y1]

def mask_denose(im_np):
    gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, thresh=2, maxval=1, type=cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    im_np = cv2.bitwise_and(im_np, im_np, mask=mask)
    return im_np

def img_size_clamp(img_np,std_height,std_width):
    h,w,c = img_np.shape
    r_hw = h/w
    r_h = h/std_height
    r_w = w/std_width

    if r_h >= r_w:
        dst_h = std_height
        dst_w = dst_h/r_hw
        dst_w = min(dst_w,std_width)
    else:
        dst_w = std_width
        dst_h = dst_w*r_hw
        dst_h = min(dst_h, std_height)

    dst_w = int(dst_w)
    dst_h = int(dst_h)

    img_np = cv2.resize(img_np, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    return img_np

def img_size_align(img_np,std_height,std_width):

    h,w,c = img_np.shape

    dst_w = std_width
    dst_h = dst_w*h/w

    if dst_h > std_height:
        dst_h = std_height
        dst_w = dst_h*w/h

    dst_w = int(dst_w)
    dst_h = int(dst_h)

    assert dst_w <= std_width and dst_h <= std_height
    assert dst_w == std_width or dst_h == std_height

    img_np = cv2.resize(img_np, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    return img_np

def size_nomalize(img_np, std_height, std_width):

    img_np = img_size_clamp(img_np, std_height, std_width)
    if config.image_size_align:
        img_np = img_size_align(img_np, std_height, std_width)

    return img_np

def gen_my_randint(min,max):
    #自定义指定范围随机整数生成，样本概率和数值成反比
    init_bound = np.random.randint(min+1, max)
    r = np.random.randint(min, init_bound)
    return r

def test_gen_my_randint():
    samples = []
    for n in range(5000):
        samples.append(gen_my_randint(32,320))
    samples = np.array(samples)
    count, bins, ignored = plt.hist(samples, 20, density=True)
    plt.show()

def get_cut_range(img_size):

    cut_size = random.randint(config.min_cut_size, min(config.max_cut_size,img_size)-1)
    #todo 自定义概率分布:尺寸和概率成反比
    #cut_size = gen_my_randint(config.min_cut_size, min(config.max_cut_size,img_size)-1)

    x0 = random.randint(0, img_size-cut_size-1)
    x1 = x0 + cut_size
    assert x1<img_size

    return x0, x1


def create_random_polygon_pts(h,w):

    pt_count = 20

    #弧度序列
    radus = 200
    rads = np.linspace(0, 2 * np.pi, pt_count)
    #半径因子序列
    facts = [random.random() for n in range(pt_count)]
    #facts = [1 for n in range(pt_count)]
    #因子平滑
    alpha = 0.6
    for n in range(1, pt_count):
        facts[n] = alpha * facts[n] + (1 - alpha) * facts[n - 1]

    xs = []
    ys = []

    x_min,x_max = w,0
    y_min,y_max = h,0

    # 按弧度序列，生成坐标序列
    for n in range(pt_count):
        rad = rads[n]
        fact = 1-0.7*facts[n]
        x = fact * radus * np.cos(rad)
        y = fact * radus * np.sin(rad)

        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)

        xs.append(x)
        ys.append(y)

    #点列坐标整体伸缩，铺满全图
    pts = []
    x_span = x_max - x_min
    y_span = y_max - y_min
    for x, y in zip(xs, ys):

        x += -(x_min)
        y += -(y_min)

        x *= (w-1)/x_span
        y *= (h-1)/y_span

        x = int(x)
        y = int(y)

        #print(x, y)

        assert (x >= 0 and x < w)
        assert (y >= 0 and y < h)

        pts.append([x, y])

    pts = np.array(pts)

    return pts

def get_random_strip(n_range,max_strip_size,min_strip_size):

    assert max_strip_size >= min_strip_size

    strip_size = random.randint(min_strip_size, max_strip_size)
    #pos = random.randint(0,n_range-strip_size-1)
    #todo 2022.12.31 拒绝产生边缘条带
    assert n_range > (strip_size+2*min_strip_size)
    pos = random.randint(min_strip_size, n_range - strip_size - min_strip_size -1)

    return pos,pos+strip_size

def random_mask(support_np):
    #随机遮盖
    h, w, c = support_np.shape

    #if random.randint(0, 1) == 0:
    if False:

        y0, y1 = get_random_strip(h,h//4,h//8)
        x0, x1 = get_random_strip(w,w//4,w//8)

        support_np[y0:y1, :, :] = 0
        support_np[:, x0:x1, :] = 0

    else:
        #创建随机蒙板
        pts = create_random_polygon_pts(h,w)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], (255))

        #应用蒙版到原图
        support_np = cv2.bitwise_and(support_np, support_np, mask=mask)

        #cv2.imshow("support_np", support_np)
        #cv2.imshow("mask", np.array(mask))
        #cv2.waitKey(0)

    return support_np,mask


'''
def norm_sup_size(img):
    # 伸缩后满足最小尺寸要求
    sw, sh = img.size
    if sw < config.min_sup_size:
        ratio = config.min_sup_size / sw
        sw = config.min_sup_size
        sh = int(sh * ratio)
    if sh < config.min_sup_size:
        ratio = config.min_sup_size / sh
        sh = config.min_sup_size
        sw = int(sw * ratio)

    img_cut = F.resize(img, (sh, sw))

    return img_cut
'''


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def video_extract(video_path,frame_save_folder):

    cap = cv2.VideoCapture(video_path)

    frame_no = 0
    file_no = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame_no % 10 == 0:

            h,w,c=frame.shape
            ratio = 0.65
            frame = cv2.resize(frame, (int(w*ratio), int(h*ratio)))

            #对医废高清视频截取重要部分
            #frame = frame[325:925,300:1100,:]
            #offset_y = 100
            #frame = frame[325+offset_y:925+offset_y, 300:1100, :]

            #tools
            offset_x = 130
            frame = frame[:, offset_x:, :]

            save_path = os.path.join(frame_save_folder,'{}.jpg'.format(file_no))
            cv2.imwrite(save_path, frame)
            print(save_path)
            file_no += 1
        print(frame_no)
        frame_no += 1

def compute_iou(rec1, rec2):

    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        ret = (intersect / (sum_area - intersect)) * 1.0
        return ret

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img


def resize_img_keep_ratio(img,target_size,pad_color):
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,pad_color)
    return img_new