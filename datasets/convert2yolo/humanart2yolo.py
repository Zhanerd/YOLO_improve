import os
import json
import cv2
from tqdm import tqdm

def convert_humanart_to_yolo(coco_json_path, image_dir, output_label_dir, output_classes_file="classes.txt"):
    os.makedirs(output_label_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    categories = data["categories"]
    images = data["images"]
    annotations = data["annotations"]

    # 类别映射（默认 HuMAn-Art 是只有 person）
    class_map = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    # 保存 classes.txt
    with open(output_classes_file, 'w') as f:
        for name in class_names:
            f.write(name + "\n")

    # 图像 ID → 文件名 映射
    imgid_to_file = {img["id"]: img["file_name"] for img in images}
    imgid_to_size = {img["id"]: (img["width"], img["height"]) for img in images}

    # 收集每张图像的所有标注
    img_ann_map = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in imgid_to_file:
            continue
        if ann.get("iscrowd", 0) == 1:
            continue
        if img_id not in img_ann_map:
            img_ann_map[img_id] = []
        img_ann_map[img_id].append(ann)

    for img_id, ann_list in tqdm(img_ann_map.items()):
        # file_name = os.path.splitext(imgid_to_file[img_id])[0]
        file_name = os.path.splitext(os.path.basename(imgid_to_file[img_id]))[0]
        txt_path = os.path.join(output_label_dir, file_name + ".txt")
        img_w, img_h = imgid_to_size[img_id]

        with open(txt_path, 'w') as f:
            for ann in ann_list:
                cat_id = ann["category_id"]
                class_id = class_map[cat_id]
                x, y, w, h = ann["bbox"]
                # 转换为 yolo 格式 (x_center, y_center, w, h)，归一化
                x_c = (x + w / 2) / img_w
                y_c = (y + h / 2) / img_h
                w /= img_w
                h /= img_h
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ 转换完成，共 {len(img_ann_map)} 张图生成了标签。")

if __name__ == "__main__":
    convert_humanart_to_yolo(
        coco_json_path=r'C:\Users\84728\Documents\HumanArt\annotations\training_humanart_dance.json',
        image_dir=r'C:\Users\84728\Documents\HumanArt\dance',
        output_label_dir=r'D:\coco_humart\dance\labels',
        output_classes_file=r'D:\coco_humart\dance\classes.txt'
    )
