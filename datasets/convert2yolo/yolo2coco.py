import os
import json
import cv2
from tqdm import tqdm

class Yolo2CocoConverter:
    def __init__(self, root_dir, classes_path):
        """
        :param root_dir: YOLO 数据集根目录，需包含 train/val/test 各子文件夹，每个子文件夹中包含 images/ 和 labels/
        :param classes_path: YOLO 格式的 classes.txt 文件路径
        """
        self.root_dir = root_dir
        self.classes_path = classes_path
        self.class_names = self._load_class_names()
        self.categories = self._build_categories()

    def _load_class_names(self):
        with open(self.classes_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _build_categories(self):
        return [
            {"id": idx, "name": name, "supercategory": "none"}
            for idx, name in enumerate(self.class_names)
        ]

    def _convert_subset(self, subset):
        """
        处理 train / val / test
        """
        image_dir = os.path.join(self.root_dir, subset, "images")
        label_dir = os.path.join(self.root_dir, subset, "labels")

        images = []
        annotations = []
        img_id = 1
        ann_id = 1


        for img_name in tqdm(os.listdir(image_dir), desc=f"Converting {subset}"):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]

            images.append({
                "id": img_id,
                "file_name": img_name,
                "height": height,
                "width": width
            })

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id = int(parts[0])
                        x_center, y_center, w_box, h_box = map(float, parts[1:])
                        x_min = (x_center - w_box / 2) * width
                        y_min = (y_center - h_box / 2) * height
                        w = w_box * width
                        h = h_box * height

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                            "segmentation": [],
                        })
                        ann_id += 1

            img_id += 1

        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": self.categories
        }
        return coco_dict

    def convert_all(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for subset in ["train", "val", "test"]:
            subset_dir = os.path.join(self.root_dir, subset)
            if not os.path.isdir(subset_dir):
                print(f"⚠️ 子目录不存在：{subset_dir}，跳过")
                continue

            coco_data = self._convert_subset(subset)
            out_path = os.path.join(output_dir, f"instances_{subset}.json")
            with open(out_path, 'w') as f:
                json.dump(coco_data, f, indent=4)
            print(f"✅ {subset} 转换完成 → {out_path}")

if __name__ == "__main__":
    '''
    your/yolo_dataset_root/
    ├── classes.txt
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    '''
    converter = Yolo2CocoConverter(
        root_dir=r"D:\coco_humart",  # 包含 train/val/test 子目录
        classes_path=r"D:\coco_humart\classes.txt"
    )
    converter.convert_all(output_dir="output/coco_jsons")