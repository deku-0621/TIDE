import os.path

import torch
from BackboneFeatureExtraction.backbone import build_backbone
from PIL import Image
import util.transforms as T
from util.misc import nested_tensor_from_tensor_list
import numpy as np
import cv2
from torchvision import transforms as pth_transforms
transform = pth_transforms.Compose([
    #pth_transforms.Resize(256, interpolation=3),
    #pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class SwinFeatFetcher:
    def __init__(self,config_args,device='cuda'):

        '''
        模型保存代码在 GroundingDINO-reduce\test\ground_dino_detector.py 的 load_model方法:
        torch.save({'model': model.backbone[0].state_dict()}, swin_model_path)
        '''

        self.device = device
        self.config_args = config_args
        self.model = build_backbone(config_args)
        self.model.to(self.device)
        self.model.eval()

    def get_support_feats(self, support_np):
        image_s = load_image(support_np)
        images_s = image_s.to(self.device)[None]
        samples_s = nested_tensor_from_tensor_list(images_s)
        with torch.no_grad():
            features_s = self.model.forward_supp_branch(samples_s, self.config_args.support_out_indices)[0]
            features_s = features_s[0].tensors
            #把torch.Size([1, 384, 50, 67])变换成torch.Size([1, 384])
            features_s = torch.mean(features_s, dim=(2,3))
            features_s = features_s.detach().cpu().numpy()
            norms = np.linalg.norm(features_s, axis=1)
            for n in range(len(features_s)):
                features_s[n] /= norms[n]
        return features_s

    def get_query_feats(self, query_np):
        image_q = load_image(query_np)
        images_q = image_q.to(self.device)[None]
        samples_q = nested_tensor_from_tensor_list(images_q)
        #with torch.no_grad():
        features_q = self.model(samples_q)
        feature_tensors = []
        for feat in features_q[0]:
            feature_tensors.append(feat.tensors[0])
        # np_feats = []
        # for feat in feature_tensors:
        #     np_feats.append(feat.cpu().numpy())
        return feature_tensors
def load_image(img_bgr: np.ndarray) -> torch.Tensor:
    transform = T.Compose(
        [
            #T.RandomResize([1000], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pillow = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    image_transformed, _ = transform(image_pillow, None)
    return image_transformed
if __name__ == '__main__':

    import cv2
    from util.slconfig import SLConfig

    config_args = SLConfig.fromfile(
        '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/config/train_config.py')
    fetcher = SwinFeatFetcher(config_args)

    image_path_q = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/tests/input/2.jpg'
    image_path_s = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/tests/input/111.jpg'

    query_np = cv2.imread(image_path_q)
    support_np = cv2.imread(image_path_s)

    features_q = fetcher.get_query_feats(query_np)
    features_s = fetcher.get_support_feats(support_np)[0]
    fetcher = SwinFeatFetcher(config_args)
    features_q2 = fetcher.get_query_feats(query_np)
    features_s2 = fetcher.get_support_feats(support_np)[0]
    fetcher = SwinFeatFetcher(config_args)
    features_q3 = fetcher.get_query_feats(query_np)
    features_s3 = fetcher.get_support_feats(support_np)[0]
    print('end')
