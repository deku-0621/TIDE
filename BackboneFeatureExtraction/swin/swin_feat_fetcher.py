import os.path

import torch
from BackboneFeatureExtraction.swin.swin_transformer import build_swin_transformer
from PIL import Image
import util.transforms as T
from util.misc import nested_tensor_from_tensor_list

def load_image(img_np):

    image_pil = Image.fromarray(img_np).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            # T.RandomColorJitter(p=0.3333),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # T.RandomHorizontalFlip(0.5),
            # T.RandomSizeCrop(384, 600),
            #cutmix
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


class SwinFeatFetcher:
    def __init__(self,pretrained_path,device='cuda'):

        '''
        模型保存代码在 GroundingDINO-reduce\test\ground_dino_detector.py 的 load_model方法:
        torch.save({'model': model.backbone[0].state_dict()}, swin_model_path)
        '''

        self.device = device
        self.model = build_swin_transformer("swin_T_224_1k", 224, out_indices=tuple([1,2,3]),dilation=False,use_checkpoint=True)

        state_dict = torch.load(pretrained_path, map_location="cpu")
        state_dict = state_dict['model']
        msg = self.model.load_state_dict(state_dict, strict=False)
        print('SwinFeatFetcher load model:',msg)
        self.model.to(self.device)
        self.model.eval()

    def get_feats(self, query_np):

        image_pil, image = load_image(query_np)
        images = image.to(self.device)[None]
        samples = nested_tensor_from_tensor_list(images)

        with torch.no_grad():
            features = self.model(samples)

        feature_tensors = []
        for lay,feat in features.items():
            feature_tensors.append(feat.tensors[0])

        return feature_tensors

if __name__ == '__main__':

    import cv2
    from config import train_config

    fetcher = train_config.get_swin_feat_fetcher()

    image_path = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/tests/input/2.jpg'
    query_np = cv2.imread(image_path)

    features = fetcher.get_feats(query_np)

    a  =1