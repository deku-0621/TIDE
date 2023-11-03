import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

# COCO数据集的路径
data_dir = '/media/pc/works/grounding-dino-modified/main/coco'

# 初始化COCO实例
coco = COCO(os.path.join(data_dir, 'annotations', 'instances_train2017.json'))

# 获取所有类别
categories = coco.loadCats(coco.getCatIds())

# 输出根目录
output_root_dir = '/media/pc/works/TIDE/TEST-TIME-FEW-SHOT-OBJECT-DETECTION-IN-THE-WILD/dataset/coco_split'
os.makedirs(output_root_dir, exist_ok=True)

for category in categories:
    break_flag = False
    category_id = category['id']
    category_name = category['name']

    # 创建以类别catid命名的子文件夹
    output_category_dir = os.path.join(output_root_dir, str(category_id))
    os.makedirs(output_category_dir, exist_ok=True)
    save_cnt = 0
    # 获取该类别的所有图像ID
    image_ids = coco.getImgIds(catIds=[category_id])

    for n,image_id in enumerate(image_ids):
        if n>1000 or break_flag:
            break
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(data_dir, 'train2017', image_info['file_name'])

        # 打开图像
        image = Image.open(image_path)

        # 获取图像上的标注框
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=[category_id]))
        for n,annotation in enumerate(annotations):
            if annotation['iscrowd'] == 1:
                continue                  
            if save_cnt == 10:
                save_cnt = 0
                break_flag = True
                break


            bbox = annotation['bbox']
            # 提取标注框范围内的图像
            cropped_image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            # 保存图像到对应类别文件夹
            output_file_path = os.path.join(output_category_dir, f"{image_info['file_name'][:-4]}_bbox{annotation['id']}.jpg")
            if cropped_image.size[0]>200 and cropped_image.size[1]>200:
                save_cnt+=1
                cropped_image.save(output_file_path)

print("图像提取完成。")
