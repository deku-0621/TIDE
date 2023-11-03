import torch
import cv2
import numpy as np
from PIL import Image



#todo 2023.5.16
from torchvision import transforms as pth_transforms
transform = pth_transforms.Compose([
    #pth_transforms.Resize(256, interpolation=3),
    #pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def get_feats(feat_fetcher, img_lst):
    img_tensors = []
    for img in img_lst:
        img_ = Image.fromarray(img)
        #img_tensor = tfs.ToTensor()(img_).to('cuda')
        img_tensor = transform(img_).to('cuda') #todo 2023.5.16
        img_tensors.append(img_tensor)

    img_tensors = torch.stack(img_tensors)
    feats = feat_fetcher.get_feats(img_tensors)

    return feats

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