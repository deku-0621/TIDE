import torch
from model_with_backbone import build_model
from util.myutils import xywh_2_xyxy
import cv2
import numpy as np
from dataset.mydataset_backbone import load_image
from util.misc import nested_tensor_from_tensor_list
def main(POS_THRES,NEG_THRES,support_img_paths,query_img_path,reshape_query = False):
    model = build_model()
    model_path = 'train_output/checkpoint_20230912_N2_CLS5BOX3GIOU1_unfrozen_backbone_support_self_attention_embedding.pth'#'train_output/checkpoint_20230905_N2_CLS5BOX2GIOU1_NO_SUPPORT_CROSS_ATTENTION_Q500_embedding.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device='cuda')
    support_list = []
    query_img = load_image(cv2.imread(query_img_path)) #if not reshape_query else load_image(cv2.resize(cv2.imread(query_img_path),(640,480)))
    query_img_ = cv2.imread(query_img_path) if not reshape_query else cv2.resize(cv2.imread(query_img_path),(640,480))
    h, w, c = query_img_.shape
    no_object_img_np = np.zeros((128,128,3),dtype=np.uint8)
    support_list.append(load_image(no_object_img_np))
    for path in support_img_paths:
        support_img = cv2.resize(cv2.imread(path),(128,128))
        support_img = load_image(support_img)
        support_list.append(support_img)
    support_sample = torch.stack(support_list).permute(1, 2, 3, 0)
    query_sample = nested_tensor_from_tensor_list([query_img]).to(device='cuda')
    support_samples = nested_tensor_from_tensor_list([support_sample]).to(device='cuda')
    outputs = model(query_sample,support_samples)
    #post process
    logits = torch.nn.functional.softmax(outputs["pred_logits"][0])
    boxes = outputs["pred_boxes"][0]
    for logit, box in zip(logits, boxes):
        xywh = [x.item() for x in box.cpu()]
        box = xywh_2_xyxy(xywh, w, h)
        prompt_cls = logit.argmax().item()
        score = logit[prompt_cls].item()
        if prompt_cls > 0 and score>POS_THRES:
            print('class:', prompt_cls, 'score:', score)
            cv2.rectangle(query_img_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(query_img_, 'class:' + str(prompt_cls), (int(box[0]), int(box[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            score_display = float(str(score)[:5])
            cv2.putText(query_img_, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
        elif score < NEG_THRES:
            print('NEG CLASS:', prompt_cls, 'score:', score)
            cv2.rectangle(query_img_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(query_img_, 'class:' + str(prompt_cls), (int(box[0]), int(box[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            score_display = float(str(score)[:5])
            cv2.putText(query_img_, str(score_display), (int(box[0]), int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0), 2)
    for index,support_img in enumerate(support_img_paths):
        support_img = cv2.resize(cv2.imread(support_img),(128,128))
        cv2.imshow(f'support_img_cls_{index+1}',support_img)
    cv2.imshow('query_img', query_img_)
    cv2.waitKey(0)
if __name__ == '__main__':
    # s_paths = ['tests/input/helmet_support.jpg','tests/input/helmet_support2.jpg','tests/input/helmet_support3.jpg','tests/input/helmet_support4.jpg','tests/input/helmet_support5.jpg','tests/input/helmet_support6.jpg']
    # q_path = 'tests/input/safety_helmet.jpg'

    s_paths = ['tests/input/2.jpg','tests/input/support_man.jpg','tests/input/93406_s.jpg']
    q_path = 'tests/input/trashman.jpg'
    main(POS_THRES=0.0,NEG_THRES=1.0,support_img_paths=s_paths,query_img_path=q_path,reshape_query = True)



