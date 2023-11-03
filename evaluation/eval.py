import os
import time
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json

from config import train_config as config
from model import build_model,collate_fn_TIDE

from dataset.mydataset import *

from util.myutils import *
#from eval_util import Evaluate
from util import utils
#eval = Evaluate()
def draw_pr_curve(precisions,recalls):
    import matplotlib.pyplot as plt
    X = recalls
    Y = precisions

    plt.figure(figsize=(8, 8))  # 定义图的大小
    plt.xlabel("recalls")  # X轴标签
    plt.ylabel("precisions")  # Y轴坐标标签
    plt.title("P-R")  # 曲线图的标题
    plt.plot(X, Y)  # 绘制曲线图

    X = np.linspace(0, 1, 10)
    Y = np.linspace(0, 1, 10)
    plt.plot(X, Y)

    plt.show()

#precisions = [0.48, 0.38, 0.33, 0.31, 0.28, 0.28, 0.28, 0.28, 0.28, 0.28]
#recalls = [0.27, 0.36, 0.39, 0.43, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44]
#draw_pr_curve(precisions,recalls)
#exit(0)

def main(args):

    # todo for cocoapi
    coco_ann_path = config.test_ann_path


    with open(coco_ann_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    categories = [{'supercategory': 'object', 'id': 0, 'name': 'object'}]
    dict_ds_tg = {'annotations':[],'images':json_data['images'],'categories':categories}
    dict_ds_pred = []

    #todo for low recall study
    dict_cateid_name = {}
    for item in json_data['categories']:
        dict_cateid_name[item['id']] = item['name']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model()
    model.to(device)

    model_path = config.test_model_path
    if model_path != None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    model.eval()

    ds = MyDataset(config.test_ann_path, config.train_image_folder, config.dino_feats_folder,config.swin_feats_folder,train=False)
    #todo 并行取样
    if config.run_as_local():
        batch_size = 1
    else:
        batch_size = 1
    sampler = torch.utils.data.RandomSampler(ds)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=True)
    data_loader_val = DataLoader(ds,batch_sampler=batch_sampler, collate_fn=collate_fn_TIDE, num_workers=0)

    resample_count = 100 #todo 实测采样数（远小于样本总数）
    ds.set_resample_count(resample_count)

    ann_obj_id = 0#todo for cocoapi
    n_count = 0
    dscount = len(data_loader_val)*batch_size
    for samples in data_loader_val:
        querys, supports,targets,info= samples
        #querys todevice 3 nested tensor in a tuple
        query_list = []
        for each in querys:
            query_list.append(each.to(device))
        supports = supports.to(device)
        if resample_count !=0 and n_count == resample_count:
            break

        #querys = querys.to(device)
        #supports = supports.to(device)
        #querys_wh = querys_wh.to(device)
        #supports_wh = supports_wh.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        start_time = time.time()

        # if config.DETR_VERSION != 2:
        #     if config.DETR_VERSION >= 8 and config.DETR_VERSION < 11:
        #         outputs = model(querys, supports, querys_wh,supports_wh)
        #     else:
        outputs = model(query_list, supports)
        # else:
        #     # todo
        #     outputs = infer_multi(model, infos[0]['query_np'], infos[0]['support_np'], device)

        total_time = time.time() - start_time
        print('total_time:', total_time)

        b = 0
        assert batch_size==1
        #if len(targets[b]['labels']) != 1:
        #    continue

        #c, h, w = querys.tensors[b].shape
        h, w = info[0]['height'],info[0]['width']

        #收集正目标标注bbox
        objs_t = []
        boxes_t = targets[b]['boxes'].cpu().numpy()
        clses_t = targets[b]['labels'].cpu().numpy()
        for n in range(len(clses_t)):
            if clses_t[n] == 0:
                box_t = tuple(xywh_2_xyxy(tuple(boxes_t[n]), w, h))
                objs_t.append((box_t,1))


        #收集正目标预测bbox
        objs_p = []
        boxes_p = outputs['pred_boxes'][b, :, :].detach().cpu().numpy()
        prob_densities = torch.nn.functional.softmax(outputs['pred_logits'][b]).cpu().detach().numpy()
        clses_p = np.argmax(prob_densities,axis=1)

        # for n in range(len(clses_p)):
        #     cls_p = clses_p[n]
        #     box_p = boxes_p[n]
        #     prob = prob_densities[n][cls_p]
        #     if cls_p == 0: #只关心正类
        #         box_p = tuple(xywh_2_xyxy(tuple(box_p), w, h))
        #         objs_p.append((box_p,prob))
        # else:
        objs_p = outputs
        clses_p = [0]*len(outputs)



        #按样本匹配多目标
        #recall = eval.append_sample(objs_p,objs_t)

        #todo for cocoapi
        image_id = info[0]['image_id']
        for obj,cls in zip(objs_t,clses_t):
            box = obj[0]
            obj_ann = {'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]],
                   'category_id':int(cls),'id':ann_obj_id,'image_id':image_id,'iscrowd':0,'area':1}
            dict_ds_tg['annotations'].append(obj_ann)
            ann_obj_id += 1

        for obj,cls in zip(objs_p,clses_p):
            if cls != 0:continue
            box,prob = obj
            obj_ann = {'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]],
                   'category_id':int(cls),'id':ann_obj_id,'image_id':image_id,'score':np.float64(prob)}
            dict_ds_pred.append(obj_ann)
            ann_obj_id += 1

        '''
        #todo 收集图片和类别信息，分析低召回率的原因
        log_home = 'D:/home/data/detr_img/recall_low'
        if recall == 0:
            query_np = infos[b]['query_np']
            support_np = infos[b]['support_np']
            for obj_t,obj_p in zip(objs_t,objs_p):
                bbox_t, _ = obj_t
                bbox_p, _ = obj_p
                cv2.rectangle(query_np, (bbox_t[0], bbox_t[1]), (bbox_t[2], bbox_t[3]),(0, 255, 0), 2)
                cv2.rectangle(query_np, (bbox_p[0], bbox_p[1]), (bbox_p[2], bbox_p[3]), (0, 0, 255), 2)
            cate_name = dict_cateid_name[infos[b]['catid']]

            log_folder = os.path.join(log_home,cate_name)
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            support_path = os.path.join(log_folder,'{}S.jpg'.format(n_count))
            query_path = os.path.join(log_folder,'{}Q.jpg'.format(n_count))
            cv2.imwrite(support_path,support_np)
            cv2.imwrite(query_path,query_np)
        '''


        n_count += 1
        print('n_count:{}/{}'.format(n_count, dscount))


    #precisions,recalls = eval.compute_mAP()
    #draw_pr_curve(precisions,recalls)

    # todo for cocoapi
    ann_tg_path = 'ann_tg.json'
    ann_pred_path = 'ann_pred.json'
    with open(ann_tg_path, "w",encoding='utf8') as f:
        json.dump(dict_ds_tg, f)
    with open(ann_pred_path, "w",encoding='utf8') as f:
        json.dump(dict_ds_pred, f)

    cocoGt = COCO(ann_tg_path)
    cocoDt = cocoGt.loadRes(ann_pred_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    import os
    import argparse
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser('TIDE evaluation script')
    args = parser.parse_args()
    args.coco_path = '/media/pc/works/grounding-dino-modified/main/coco'
    args.output_dir = 'train_output/'
    args.num_queries = config.num_queries
    main(args)
