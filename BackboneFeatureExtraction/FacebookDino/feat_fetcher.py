
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import BackboneFeatureExtraction.FacebookDino.utils as utils
import os
import BackboneFeatureExtraction.FacebookDino.vision_transformer as vits

def ForServer():
    name = os.popen("hostname").read()
    if 'lyn-laptop\n' == name:
        return False
    else:
        return True

class ViTFeatFetcher:
    def __init__(self,pretrained_path,device):

        self.init(pretrained_path,device)


    def init(self,pretrained_path,device):
        args_dict = dict(
            arch='vit_small', batch_size_per_gpu=128, checkpoint_key='teacher',
            data_path='/path/to/imagenet', dist_url='env://', dump_features=None,
            load_features=None, local_rank=0, nb_knn=[10, 20, 100, 200],
            num_workers=10, patch_size=16,#16
            pretrained_weights=pretrained_path,
            temperature=0.07, use_cuda=False)

        args = argparse.Namespace(**args_dict)

        #utils.init_distributed_mode(args)
        cudnn.benchmark = True

        self.model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        self.model.to(device)
        utils.load_pretrained_weights(self.model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        self.model.eval()

        self.patch_size = args.patch_size


    def size_align(self,size):
        remain = size % self.patch_size
        if remain > 0:
            size += (self.patch_size - remain)
        return size

    '''
    #得到的特征图映射效果很差
    def get_alll_feat(self,img):
        img = img[None]
        with torch.no_grad():
            feats = self.model.get_allfeat(img)[0]
            feat = feats[0].detach().cpu().numpy()
            feat = feat / np.linalg.norm(feat)

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size
            featmap = feats[1:].reshape(feats.shape[1],feat_h, feat_w)

            return feat,featmap
    '''


    def get_feat(self, img):

        img = img[None]

        #imgs = []
        #for n in range(100):
        #    imgs.append(img)
        #img = torch.cat(imgs,0)

        with torch.no_grad():

            feat = self.model(img).clone()[0]
            feat = feat.detach().cpu().numpy()
            feat = feat / np.linalg.norm(feat)

            return feat

    def get_feats(self, imgs):

        with torch.no_grad():

            feats = self.model(imgs).clone()
            feats = feats.detach().cpu().numpy()
            norms = np.linalg.norm(feats, axis=1)

            #feats = feats / norms
            for n in range(len(feats)):
                feats[n] /= norms[n]

            return feats


    def get_featmap_pro(self, img, n=1):

        img = img[None]
        with torch.no_grad():

            # get selected layer activations
            feat= self.model.get_intermediate_feat(img, n=n)
            feat= feat[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

            return image_feat[0]

    def get_last_selfattention(self,x):
        attentions = self.model.get_last_selfattention(x)
        return attentions
