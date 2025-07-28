import os
import json
import cv2
import base64
import tqdm

class YoloLabelmeConverter:
    def __init__(self, class_file, version='5.6.0'):
        self.class_file = class_file
        self.version = version
        self.class_names = self.load_class_names()
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def load_class_names(self):
        with open(self.class_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def yolo_to_bbox(self, yolo_bbox, img_w, img_h):
        x_c, y_c, w, h = yolo_bbox
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        return [[x1, y1], [x2, y2]]

    def bbox_to_yolo(self, points, img_w, img_h):
        x1, y1 = points[0]
        x2, y2 = points[1]
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        return round(x_center, 6), round(y_center, 6), round(width, 6), round(height, 6)

    def encode_image_base64(self, image_path):
        with open(image_path, 'rb') as img_f:
            return base64.b64encode(img_f.read()).decode('utf-8')

    def convert_yolo_to_labelme(self, image_dir, label_dir, output_dir, embed_image=False):
        os.makedirs(output_dir, exist_ok=True)
        for img_name in tqdm(os.listdir(image_dir), desc=f"Converting {image_dir}"):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(image_dir, img_name)
            label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')
            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 无法读取图像: {img_path}")
                continue
            h, w = img.shape[:2]
            shapes = []

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    points = self.yolo_to_bbox(bbox, w, h)
                    shapes.append({
                        "label": self.class_names[class_id],
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None
                    })

            labelme_json = {
                "version": self.version,
                "flags": {},
                "shapes": shapes,
                "imagePath": img_name,
                "imageHeight": h,
                "imageWidth": w,
                "imageData": self.encode_image_base64(img_path) if embed_image else None
            }

            out_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.json')
            with open(out_path, 'w') as f:
                json.dump(labelme_json, f, indent=4)

    def convert_labelme_to_yolo(self, json_dir, output_txt_dir, output_class_file=None):
        os.makedirs(output_txt_dir, exist_ok=True)
        for file in tqdm(os.listdir(json_dir), desc=f"Converting {json_dir}"):
            if not file.endswith('.json'):
                continue
            with open(os.path.join(json_dir, file), 'r') as f:
                data = json.load(f)

            img_w = data.get('imageWidth')
            img_h = data.get('imageHeight')
            shapes = data.get('shapes', [])
            txt_name = os.path.splitext(file)[0] + '.txt'

            with open(os.path.join(output_txt_dir, txt_name), 'w') as out_f:
                for shape in shapes:
                    if shape['shape_type'] != 'rectangle':
                        continue
                    label = shape['label']
                    class_id = self.class_name_to_id.get(label)
                    if class_id is None:
                        # 新类，动态添加
                        class_id = len(self.class_names)
                        self.class_names.append(label)
                        self.class_name_to_id[label] = class_id
                    x, y, w, h = self.bbox_to_yolo(shape['points'], img_w, img_h)
                    out_f.write(f"{class_id} {x} {y} {w} {h}\n")

        if output_class_file:
            with open(output_class_file, 'w') as f:
                for cls in self.class_names:
                    f.write(cls + '\n')

if __name__ == '__main__':
    converter = YoloLabelmeConverter(class_file=r'D:\coco_humart\classes.txt')
    ### yolo2labelme
    converter.convert_yolo_to_labelme(
        image_dir=r'D:\coco_humart\train1\images',
        label_dir=r'D:\coco_humart\train1\labels',
        output_dir=r'D:\coco_humart\train1\labelme_json',
        embed_image=False
    )
    ### labelme2yolo
    converter.convert_labelme_to_yolo(
        json_dir=r'D:\coco_humart\train1\labelme_json',
        output_txt_dir=r'D:\coco_humart\train1\yolo_txt'
    )